from __future__ import annotations
import torch
from .sequence import Sequence
from typing import Optional
from numpy import pi
import numpy as np
import time
from . import util
from auxutil.cg_batch import cg_batch
# import torchkbnufft as tkbn


def reconstruct(signal: torch.Tensor,
                kspace: torch.Tensor,
                resolution: tuple[int, int, int] | float | None = None,
                FOV: tuple[float, float, float] | float | None = None,
                return_multicoil: bool = False) -> torch.Tensor:
    """Adjoint reconstruction of the signal, based on a provided kspace.

    Parameters
    ----------
    signal : torch.Tensor
        A flat complex tensor containing the signal, shape (sample_count, )
    kspace : torch.Tensor
        A real tensor of shape (sample_count, 4) for the kspace trajectory
    resolution : (int, int, int) | float | None
        The resolution of the reconstruction. Can be either provided directly
        as tuple or set to None, in which case the resolution will be derived
        from the k-space (currently only for cartesian trajectories). A single
        float value will be used as factor for a derived resolution.
    FOV : (float, float, float) | float | None
        Because the adjoint reconstruction adapts to the k-space used
        for measurement, scaling gradients will not directly change the FOV of
        the reconstruction. All SimData phantoms have a normalized size of
        (1, 1, 1). Similar to the resolution, a value of None will
        automatically derive the FOV of the sequence based on the kspace. A
        float value can be used to scale this derived FOV.

    Returns
    -------
    torch.Tensor
        A complex tensor with the reconstructed image, the shape is given by
        the resolution.
    """
    res_scale = 1.0
    fov_scale = 1.0
    if isinstance(resolution, float):
        res_scale = resolution
        resolution = None
    if isinstance(FOV, float):
        fov_scale = FOV
        FOV = None

    # Atomatic detection of FOV - NOTE: only works for cartesian k-spaces
    # we assume that there is a sample at 0, 0 nad calculate the FOV
    # based on the distance on the nearest samples in x, y and z direction
    if FOV is None:
        def fov(t: torch.Tensor) -> float:
            t = t[t > 1e-3]
            return 1.0 if t.numel() == 0 else float(t.min())
        tmp = kspace[:, :3].abs()
        fov_x = fov_scale / fov(tmp[:, 0])
        fov_y = fov_scale / fov(tmp[:, 1])
        fov_z = fov_scale / fov(tmp[:, 2])
        FOV = (fov_x, fov_y, fov_z)
        print(f"Detected FOV: {FOV}")

    # Atomatic detection of resolution
    if resolution is None:
        def res(scale: float, fov: float, t: torch.Tensor) -> int:
            tmp = (scale * (fov * (t.max() - t.min()) + 1)).round()
            return max(int(tmp), 1)
        res_x = res(res_scale, FOV[0], kspace[:, 0])
        res_y = res(res_scale, FOV[1], kspace[:, 1])
        res_z = res(res_scale, FOV[2], kspace[:, 2])
        resolution = (res_x, res_y, res_z)
        print(f"Detected resolution: {resolution}")

    # Same grid as defined in SimData
    pos_x = torch.linspace(-0.5, 0.5, resolution[0] + 1)[:-1] * FOV[0]
    pos_y = torch.linspace(-0.5, 0.5, resolution[1] + 1)[:-1] * FOV[1]
    pos_z = torch.linspace(-0.5, 0.5, resolution[2] + 1)[:-1] * FOV[2]
    pos_x, pos_y, pos_z = torch.meshgrid(pos_x, pos_y, pos_z)

    voxel_pos = util.set_device(torch.stack([
        pos_x.flatten(),
        pos_y.flatten(),
        pos_z.flatten()
    ], dim=1)).t()

    # (Samples, 4)
    kspace = util.set_device(kspace)
    # (Samples, 3) x (3, Voxels)
    phase = kspace[:, :3] @ voxel_pos
    # (Samples, Voxels): Rotation of all voxels at every event
    rot = torch.exp(2j*pi * phase)  # Matches definition of iDFT

    NCoils = signal.shape[1]
    
    
    if return_multicoil:
        return (signal.t() @ rot).view((NCoils, *resolution)).cpu()
    else: 
        return torch.sqrt(((torch.abs(signal.t() @ rot))**2).sum(0)).view(resolution).cpu()


def reconstruct_sense_encop(signal: torch.Tensor,
                kloc: torch.Tensor,
                resolution: tuple[int, int, int] | float | None = None,
                FOV: tuple[float, float, float] | float | None = None,
                coil_sens: torch.Tensor = None,
                mode: str = 'cg',
                lambd: Optional[float] = 0.0,
                maxiter: Optional[int] = 50,
                rtol: Optional[float] = 1e-5,
                B0: Optional[torch.Tensor] = None,
                T2dash: Optional[float] = None,
                return_gfactor: bool = False) :
    """SENSE reconstruction, supports multiple ways of solving the linear equation
    system: adjoint, pinv, CG

    Parameters
    ----------
    signal : torch.Tensor
        A flat complex tensor containing the signal, shape (sample_count, )
    kloc : torch.Tensor
        A real tensor of shape (sample_count, 4) for the kspace trajectory
    resolution : (int, int, int) | float | None
        The resolution of the reconstruction. Can be either provided directly
        as tuple or set to None, in which case the resolution will be derived
        from the k-space (currently only for cartesian trajectories). A single
        float value will be used as factor for a derived resolution.
    FOV : (float, float, float) | float | None
        Because the adjoint reconstruction adapts to the k-space used
        for measurement, scaling gradients will not directly change the FOV of
        the reconstruction. All SimData phantoms have a normalized size of
        (1, 1, 1). Similar to the resolution, a value of None will
        automatically derive the FOV of the sequence based on the kspace. A
        float value can be used to scale this derived FOV.
    coil_sens : torch.Tensor
        image space coil sensitivity maps [NCoils, Nx,Ny,Nz]
    mode : str
        'adjoint', 'pinv' or cg

    Returns
    -------
    torch.Tensor
        A complex tensor with the reconstructed image, the shape is given by
        the resolution.
    """
    
    res_scale = 1.0
    fov_scale = 1.0
    if isinstance(resolution, float):
        res_scale = resolution
        resolution = None
    if isinstance(FOV, float):
        fov_scale = FOV
        FOV = None

    # Atomatic detection of FOV - NOTE: only works for cartesian k-spaces
    # we assume that there is a sample at 0, 0 nad calculate the FOV
    # based on the distance on the nearest samples in x, y and z direction
    if FOV is None:
        def fov(t: torch.Tensor) -> float:
            t = t[t > 1e-3]
            return 1.0 if t.numel() == 0 else float(t.min())
        tmp = kloc[:, :3].abs()
        fov_x = fov_scale / fov(tmp[:, 0])
        fov_y = fov_scale / fov(tmp[:, 1])
        fov_z = fov_scale / fov(tmp[:, 2])
        FOV = (fov_x, fov_y, fov_z)
        print(f"Detected FOV: {FOV}")

    # Atomatic detection of resolution
    if resolution is None:
        def res(scale: float, fov: float, t: torch.Tensor) -> int:
            tmp = (scale * (fov * (t.max() - t.min()) + 1)).round()
            return max(int(tmp), 1)
        res_x = res(res_scale, FOV[0], kloc[:, 0])
        res_y = res(res_scale, FOV[1], kloc[:, 1])
        res_z = res(res_scale, FOV[2], kloc[:, 2])
        resolution = (res_x, res_y, res_z)
        print(f"Detected resolution: {resolution}")

    # Same grid as defined in SimData
    pos_x = torch.linspace(-0.5, 0.5, resolution[0] + 1)[:-1] * FOV[0]
    pos_y = torch.linspace(-0.5, 0.5, resolution[1] + 1)[:-1] * FOV[1]
    pos_z = torch.linspace(-0.5, 0.5, resolution[2] + 1)[:-1] * FOV[2]
    pos_x, pos_y, pos_z = torch.meshgrid(pos_x, pos_y, pos_z)

    voxel_pos = util.set_device(torch.stack([
        pos_x.flatten(),
        pos_y.flatten(),
        pos_z.flatten()
    ], dim=1)).t()

    # (Samples, 4)
    kloc = util.set_device(kloc)
    # (Samples, 3) x (3, Voxels)
    phase = kloc[:, :3] @ voxel_pos
    # (Samples, Voxels): Rotation of all voxels at every event
    F = torch.exp(-2j*pi * phase)  # Matches definition of iDFT
    
    # reconstruct only voxels where there is sensitivity > 0
    sens_mask = torch.sum(torch.abs(coil_sens)**2, dim=0) > .5
    # sens_mask[:] = True
    
    # form Pruessmann's SENSE operator: E_{gr} = s_g(vox_r) * exp(i ksp_k @ vox_r)
    # [dotwise/Hadamard product of sensitivity and spatial exponential part, tensor product of kspace and spatial exponential part]
    NCoils = signal.shape[1]
    S = coil_sens.reshape(NCoils,-1) # sensitivities
    E = torch.einsum('gr,kr->gkr', S, F).reshape(NCoils*kloc.shape[0],-1) # Ncoils * Nk x Nvox
    E = E[:,sens_mask.flatten()] # exclude locations where sensitivities are 0 anyway
    EHE = E.conj().t() @ E # normal operator, only if this is identity (or very close to it), adjoint gives good reco
    
    # helpful for analyzing reco properties: Eigenvalue spectrum of EHE, but takes long to calculate...
    # ratio of max(eigenvals)/min(eigenvals) is condition number
    # eigenvals, _ = torch.linalg.eigh(EHE) 
    # eigenvals = torch.abs(eigenvals) # should be real and positive already, but numerical imprecision...
    # eigenvals, _ = torch.sort(eigenvals)
    

    # gather signal that should be reconstructed   
    # signal_adc = (torch.cat([signal[i].squeeze()[:, torch.abs(seq[i].adc_usage) > 0].unsqueeze(0)
    #                         for i in range(len(signal))])
    #               .permute(1,0,2) # quite confusing, but works that way (at least for 2D I hope...)
    #               ).flatten()
    signal_adc = signal.t().flatten()
    

    reco  = util.set_device(torch.zeros(*resolution, dtype=torch.cfloat))
    
    if mode.lower() == 'adjoint':
        reco[sens_mask] = (E.conj().t() @ signal_adc)
    
    if mode.lower() == 'pinv':
        # Tikhonov (=L2) regularization to counteract ill-conditioning
        # should be chosen according to Eigenvalue spectrum of E^HE I think
        # lambd = 0 # 100 seemed good now for all trajectories
        EHE_reg = EHE + util.set_device(lambd*torch.eye(EHE.shape[0]))
       
        t = time.time()
        reco[sens_mask] = torch.linalg.inv(EHE_reg) @ (E.conj().t() @ signal_adc) # explicit (regularized) pseudoinversion
        # reco_pinv = (torch.linalg.pinv(E) @ signal_full).reshape(resolution)
        elapsed = time.time() - t
        print(f'pinv took {elapsed:.4f}s')
        
    if mode.lower() == 'cg':
        # lambd = 10 # Tikhonov regularization
        # maxiter = 100
        
        # rtol = 1e-5 # convergence criterion
        EHE_reg = EHE + util.set_device(lambd*torch.eye(EHE.shape[0]))
        # scalefactor = 1/max(eigenvals) 
        scalefactor = 1e-5 # something like preconditioner to counteract possible numerical scaling issues
        cgmm = lambda x: (EHE_reg*scalefactor @ x)
        rhs = (E.conj().t() @ signal_adc).unsqueeze(0).unsqueeze(-1)*scalefactor # right-hand side of normal equation system
        
        # def cg_callback(k, X_k, R_k): #for plotting intermediate CG iterations
        #     if k in showiters:
        #         reco_cg = torch.zeros(resolution, dtype=torch.complex64).squeeze()
        #         reco_cg[sens_mask] = X_k.squeeze()
        #         plot_complex_image(reco_cg)
        #         plt.suptitle(f'CG reco, iter {k}, lambda={lambd}')
        
        
        
        t = time.time()
        reco[sens_mask] = cg_batch(cgmm, rhs, maxiter=maxiter, rtol=rtol, verbose=False)[0].squeeze()
        elapsed = time.time() - t
        # print(f'CG ENCOP took {elapsed:.4f}s')
    
    
    
    # plt.figure(), plt.imshow(torch.abs(gfactor)), plt.colorbar()
    # plt.suptitle('gfactor')
    
    if return_gfactor:
        gfactor = torch.zeros(resolution, dtype=torch.complex64)
        # invEHE_reg = torch.linalg.inv((E.conj().t()) @ E) @ E.conj().t()
        invEHE_reg = torch.linalg.inv(EHE_reg)
        gfactor[sens_mask] = torch.sqrt(torch.diag(invEHE_reg) * torch.diag(EHE_reg))
        
        return reco, gfactor
    else:
        return reco

def reconstruct_sense_nufft(seq: Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int],
                size: tuple[int, int, int],
                coil_sens: torch.Tensor,
                mode: str = 'cg',
                lambd: Optional[float] = 0.0,
                maxiter: Optional[int] = 50,
                rtol: Optional[float] = 1e-5,
                B0: Optional[torch.Tensor] = None,
                T2dash: Optional[float] = None) -> torch.Tensor:
    
    im_size = resolution[:-1] # 2D for now
    grid_size = (im_size[0]*2,)*2 # twice as much as image size looks good
    dtype = torch.complex64
    
    NCoils = signal[0].shape[0]

    # create NUFFT objects
    nufft_ob = tkbn.KbNufft(
        im_size=im_size,
        grid_size=grid_size,
        n_shift = (ii//2-1 for ii in im_size), # needed to stay in agreement with encoding operator / adjoint reco, otherwise image shifts by one pixel
    ).to(dtype)
    adjnufft_ob = tkbn.KbNufftAdjoint(
        im_size=im_size,
        grid_size=grid_size,
        n_shift = (ii//2-1 for ii in im_size),
    ).to(dtype)
    
    # rescale trajectory to [-pi, pi)
    # get kspace coordinates 
    traj = seq.get_kspace_trajectory()
    traj_adc = torch.cat([traj[i][torch.abs(seq[i].adc) > 0] for i in range(len(traj))]) #...only at ADC sampling locations
    traj_nufft = traj_adc[:,:-1].t() * (torch.tensor(size[:-1]) / torch.tensor(resolution[:-1])).unsqueeze(-1) * 1
    
    
    # gather signal that should be reconstructed   
    signal_adc = (torch.cat([signal[i].squeeze()[torch.abs(seq[i].adc) > 0, :].unsqueeze(0)
                            for i in range(1,len(signal))])
                  .permute(1,0,2) # quite confusing, but works that way (at least for 2D I hope...)
                  ).flatten()
    kdata = signal_adc.reshape(NCoils,-1).unsqueeze(0) # add batch and coil dims: batch x coils x ksamples
    
    #  method 3: conjugate gradient inversion of forward NUFFT (type 2)
    # sens_maps_nufft = coil_sens.flip(1,2).roll(1,1).roll(1,2) # not needed if NUFFT is defined with correct n_shift
    sens_maps_nufft = coil_sens.flip(1,2)


    
    def EH_nufft(kdata): # adjoint
        res = adjnufft_ob(kdata.reshape(1,NCoils,-1), traj_nufft)
        res = torch.sum(res * sens_maps_nufft.unsqueeze(0).conj(), dim=1)
        # import pdb; pdb.set_trace()
        return res
    
    def E_nufft(x): # forward
        # import pdb; pdb.set_trace()
        x = x.flatten()
        res = x.unsqueeze(0) * sens_maps_nufft.reshape(NCoils,-1)
        res = nufft_ob(res.reshape((NCoils,*resolution)).squeeze().unsqueeze(0),
                 traj_nufft)
        return res
    
    scale_factor = 1e-5
    # maxiter = 100
    # showiters = [1, 5, 10, 25, 50, 75]
    # rtol = 1e-5 # convergence criterion
    # lambd = 0

    def cgmm(x): # apply E^H E to x (left hand side operator of normal equation system)
        ret = EH_nufft(E_nufft(x)).flatten()
        # ret = ret + lambd*torch.eye(ret.shape[0])
        return ret.unsqueeze(-1)  * scale_factor
    
    rhs = EH_nufft(kdata).flatten().unsqueeze(0).unsqueeze(-1) * scale_factor # E^H kdata (right hand side of normal equation system)
    
    # def cg_callback(k, X_k, R_k): #for plotting intermediate CG iterations
    #     if k in showiters:
    #         plot_complex_image(X_k.reshape(resolution))
    #         plt.suptitle(f'CG reco with NUFFT, iter {k}, lambda={lambd}')
        
    t = time.time()
    reco_cg_nufft = cg_batch(cgmm, rhs, maxiter=maxiter, rtol=rtol, verbose=True)[0].reshape(resolution)
    elapsed = time.time() - t
    print(f'CG NUFFT took {elapsed:.4f}s')

    reco_cg_nufft = reco_cg_nufft.flip(0,1) 
    
    return reco_cg_nufft

def get_kmatrix(seq: Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], contrast = 0,
                kspace_scaling: torch.Tensor | torch.Tensor | None = None
                ) -> torch.Tensor:
    '''
    reorder scanner signal according to kspace trajectory, works only for
    cartesian (under)sampling (where kspace grid points are hit exactly)
    '''
    # import pdb; pdb.set_trace()
    NCoils = signal.shape[1]
    
    if kspace_scaling is None:
        kmax = torch.round(torch.max(torch.abs(seq.get_kspace()[:,:3]),0).values)
        kspace_scaling = kmax*2/torch.tensor(resolution)
    
        kspace_scaling[kspace_scaling==0] = 1
    traj = seq.get_kspace()[:,:3]/kspace_scaling
    kindices = (traj + torch.floor(util.set_device(torch.tensor(resolution)) / 2)).round().to(int)
    if contrast:
        mask = seq.get_contrast_mask(contrast)
        signal = signal[mask]
        kindices = kindices[mask]
    # import pdb; pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(kindices[:,0], kindices[:,1], '.', ms=1)
    # plt.axis('equal')
    
    kmatrix = util.set_device(torch.zeros(*resolution, NCoils, dtype=torch.complex64))
    
    for jj in range(kindices.shape[0]): # I'm sure there must be a way of doing this without any loop...
        ix, iy, iz = kindices[jj,:]
        kmatrix[ix,iy,iz,:] = signal[jj,:]
        
        
    return kmatrix.permute([3,0,1,2]) # Dim (NCoils x resolution)


def reconstruct_cartesian_fft(seq: Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], contrast = 0
                ) -> torch.Tensor:
    '''
    do fft reco for Cartesian kspace grid
    '''
    
    ksp = get_kmatrix(seq, signal, resolution, contrast)
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ksp,dim=dim),dim=dim),dim=dim)
    
    return reco_fft # coils first

def reconstruct_cartesian_fft_naive(seq: Sequence, signal: list[torch.Tensor], 
                resolution: tuple[int, int, int], contrast = 0
                ) -> torch.Tensor:
    '''
    do naive fft reco for any kind of signal,
    naive = just assume rectangular kspace matrix (no matter which trajectory was acutally there)
    
    TBD: handle partitions / 3D!
    '''
    
    NRep = len(seq)
    NCol = torch.sum(seq[0].adc_usage > 0) # assume same number of ADC points in each rep
    
    ksp = signal.reshape([NRep,NCol,1,-1]).permute(3,1,0,2) # coils first,  compensate xy flip
    dim = (1,2,3)
    reco_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ksp,dim=dim),dim=dim),dim=dim)
    
    return reco_fft # coils first


def adaptive_combine(im, bs=None, modeSVD=True, modeSliBySli=False, donorm=0):
    '''
    Adaptive recon based on Walsh et al.
    Walsh DO, Gmitro AF, Marcellin MW.
    Adaptive reconstruction of phased array MR imagery. 
    Magn Reson Med. 2000 May;43(5):682-90.
    
    and
    
    Mark Griswold, David Walsh, Robin Heidemann, Axel Haase, Peter Jakob. 
    The Use of an Adaptive Reconstruction for Array Coil Sensitivity Mapping and Intensity Normalization, 
    Proceedings of the Tenth  Scientific Meeting of the International Society for Magnetic Resonance in Medicine pg 2410 (2002)
    
    implementation as in adaptiveCombine.m, which is part of recoVBVD
    
    -----------------------------------------------------------------
    
    im: Ncoils x Nx x Ny x Nz
    bs: optional (block size for sliding window svd)
    modeSVD: do some form of coil compression before getting weights (seems to be good according to recoVBVD)
    modeSliBySli: only 2D kernels
    donorm: empirical intensity normalization
    
    -----------------------------------------------------------------
    outputs:
        reco: Nx x Ny x Nz coil combined image
        weights: Ncoils x Nx x Ny x Nz Walsh weights
        norm: only if donorm=True, intensity normalization image
    
    
    ignores noise correlation for now!
    '''
    
    sz = im.shape
    nc = sz[0] # coils first!
    n = torch.tensor(sz[1:])
    
    weights = 1j*torch.zeros([nc, n.prod()])
    
    if bs is None: # automatic determination of block size
        bs = n.clone()
        bs[bs>7] = 7
        if n[2] > 1:
            bs[2] = 3 if n[2] > 3 else n[2]
            
    if modeSliBySli:
        bs[2] = 1
        
    if modeSVD:
        # intuitively: if more then 12 coils, use low rank approximation of coil images to determine coil weights
        nc_svd = int(min(min(12,max(9,np.floor(nc/2))),nc))
    else:
        nc_svd = nc
        
    cnt = 0
    maxcoil = 0
    if not modeSliBySli:
        if modeSVD:
            imcorr = im.reshape((nc,-1)) @ im.reshape((nc,-1)).conj().t()
            _, _, Vh = torch.linalg.svd(imcorr)
            V = Vh.conj().t()
            V = V[:,:nc_svd]
        else:
            V = torch.eye(nc, dtype=torch.complex64)
            _, maxcoil = torch.max(torch.sum(torch.abs(im), dim=(1,2,3)),0)
            
    # sliding window SVD for coil combination weights
    
    for z in range(n[2]): 
        if modeSliBySli:
            if modeSVD:
                tmp = im[:,:,:,z].reshape((nc,-1))
                _, _, Vh = torch.linalg.svd(tmp @ tmp.conj().t())
                V = Vh.conj().t()
                V = V[:,:nc_svd]
            else:
                V = torch.eye(nc, dtype=torch.complex64)
                _, maxcoil = torch.max(torch.sum(torch.abs(im[:,:,:,z]), dim=(1,2)),0)

        for y in range(n[1]): 
            for x in range(n[0]): 
                # current position of sliding window
                ix = torch.tensor([x,y,z])
                imin = torch.max(ix - torch.floor(bs.float()/2), torch.tensor([0.])).int()
                imax = torch.min(ix + torch.ceil(bs.float()/2) -1, (n-1.)).int() + 1
                
                # import pdb; pdb.set_trace()
                m1 = im[:, imin[0]:imax[0], imin[1]:imax[1], imin[2]:imax[2]].reshape((nc,-1))
                m1 = V.conj().t() @ m1
                m = m1 @ m1.conj().t() # signal covariance
                
                # d, v = torch.linalg.eigh(m) 
                # tmp = v[:,-1] # last eigenvalue is always largest
                d, v = torch.linalg.eig(m) 
                _, ind = torch.max(torch.abs(d),0)
                tmp = v[:,ind]
                tmp = V @ tmp # transform back to original coil space
                
                # Correct phase based on coil with max intensity
                tmp = tmp * torch.exp(-1j*torch.angle(tmp[maxcoil]));
                
                weights[:, cnt] = tmp.conj() / (tmp.conj().t() @ tmp)
                
                cnt += 1
    
    # now combine coils
    weights = weights.reshape(sz).permute([0,2,1,3]) # permute is neccessary due to inverted row/column major flattening order between Matlab and python
    recon = torch.sum(weights * im, dim=0).reshape((*n, ))
    
    if donorm:
        norm = torch.sum(torch.abs(weights)**2, dim=0).reshape(n)
        recon = recon * norm
        return recon, weights, norm
    else:
        return recon, weights
    
def sos(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.abs(x)**2,0))