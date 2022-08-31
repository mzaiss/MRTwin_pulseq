# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:21:11 2021

@author: fmglang
"""

import torch
import numpy as np

def pinv_reg(A, lambd=0.01):
        '''
        regularized pseudo-inverse of matrix A with reg parameter lambd
        (similar to Matlab implementation)
        '''

        m,n = A.shape
        if n > m:
            At = A
            A = A.conj().t()
            n = m
            finalTranspose = True
        else:
            At = A.conj().t()
            finalTranspose = False
        
        AtA = At @ A
        S,_ = torch.linalg.eigh(AtA)
        lambda_sq = lambd**2 * torch.abs(torch.max(S))
        
        identity = torch.eye(n).to(AtA.device) # potentially gpu
        
        X = torch.linalg.inv(AtA + identity*lambda_sq) @ At # invert does not work, use solve instead
        
        if finalTranspose: X = X.conj().t()
        
        return X

def get_source_and_target_cell(afy, afz, delta):
        '''
        3D implementation of Felix Breuer's grappa_imspace, as included in recoVBVD
        ---------
        get indices of source and target kspace points
        
        not sure if it really works for 3D
        '''
        af = afy *afz
        
        # define elementary sampling pattern cell
        pat = torch.zeros([af, af])
        
        cnt = 0
        for k in range(0, af, afy):
            pat[k:af:af, (cnt % afz):af:afz] = 1
            cnt += delta
        
        SrcCell = torch.roll(pat, [af//2,af//2], [0,1])
        
        tmp = torch.where(SrcCell.t().flatten()==1)[0] + 1 # to imitate Matlab behavior, there it would be simply tmp=find(SrcCell==1)
        idxy = ((tmp-1) % af) + 1
        idxz = torch.ceil(tmp/af).to(int)
        
        #### calculate distances r from center position (1,1);
        idx = torch.zeros((len(tmp),2))
        idx[:,0] = idxy - np.floor(af/2) - 1
        idx[:,1] = idxz - np.floor(af/2) - 1
        
        r = torch.sqrt(idx[:,0]**2 + idx[:,1]**2)
        
        #### get index of nearest point from center
        _, I = torch.sort(r)
        idxNextPoint = torch.abs((idx[I[1],:])).to(int)
        
        # Get Target block size
        if idxNextPoint[0] < idxNextPoint[1]:
            tblz = idxNextPoint[1]
            if tblz == 0:
                tblz = af
            tbly = torch.ceil(af/tblz).to(int)
        elif idxNextPoint[0] > idxNextPoint[1]:
            tbly = idxNextPoint[0]
            if tbly == 0:
                tbly = af
            tblz = (torch.floor(af/tbly)).to(int)
        else:
            tblz = af - idxNextPoint[0]
            tbly = (torch.floor(af/tblz)).to(int)
        
        

        
        tmp = torch.zeros([af, af])
        TrgCell = torch.zeros([af, af])
        
        #### This is the center index
        IdxCy = torch.floor(torch.tensor([af])/2)+1
        IdxCz = torch.floor(torch.tensor([af])/2)+1
        
        # This is the start index of the target block
        IdxSy = IdxCy - torch.floor(tbly/2)
        IdxSz = IdxCz - torch.floor(tblz/2)
        
        # reorder indices such that we start from the center
        _, ReoY = torch.sort(torch.abs(torch.arange(1,tbly+1) - (torch.floor(tbly/2) + 1)))
        _, ReoZ = torch.sort(torch.abs(torch.arange(1,tblz+1) - (torch.floor(tblz/2) + 1)))
        
        # Shift the kernel within the target block to determine target positions
        # starting from the center
        
        
        for y in range(tbly):
            for z in range(tblz):
                IdxY = IdxSy + ReoY[y] - 1
                IdxZ = IdxSz + ReoZ[z] - 1
                shifty = IdxY - IdxCy + 1 + 1
                shiftz = IdxZ - IdxCz + 1 + 1
                
                tmp += torch.roll(SrcCell, [int(shifty[0]),int(shiftz[0])], [0,1])
                if torch.max(tmp.squeeze()) > 1:
                    # This shift is not allowed --> redo 
                    tmp -= torch.roll(SrcCell,(int(shifty),int(shiftz)), [0,1])
                else:
                    # This is a target point
                    TrgCell[int(IdxY[0]), int(IdxZ[0])] = 1
        
        
        tmp = torch.where(TrgCell.t().flatten()==1)[0] + 1
        idxy = ((tmp - 1) % af) + 1
        idxz = torch.ceil(tmp/af).to(int) 
        
        shift = torch.zeros(len(idxy),2)
        shift[:,0] = idxy - np.floor(af/2) - 1
        shift[:,1] = idxz - np.floor(af/2) - 1
        
        
        return SrcCell, TrgCell, tbly, tblz, shift


def create_grappa_weight_set(kspace, accely, accelz, delta, full_sz, lambd=0.01,
                             kernel_size=None, NACSy=None, NACSz=None):
        '''
        3D implementation of Felix Breuer's grappa_imspace, as included in recoVBVD
        ---------
        create grappa weight set from ACS scan
        
        kspace: (NCoils, NxACS, NyACS, NzACS)
        accely:         acceleration factor in phase direction
        accelz:         acceleration factor in partition direction
        
        full_sz:        full size [nx,ny,nz] of the reconstructed image
        lambd:          regularization factor for pseudoinverse
        kernel_size:    (sblx, sbly, sblz)
        NACSy:           number of ACS lines: optional
                        if this is not given, all non-zero PE lines are used
        
        returns ws, wsKernel, ws_mult
        wsKernel:         grappa weight set (for kspace reco by convolution)
        ws_imspace:   weight set in image space for reco by multiplication
       
        '''
        NCoils = kspace.shape[0]
        device = kspace.device

        # crop calibration kspace (remove zeros in the outer parts)
        # kspace = self.get_kmatrix().numpy()
        
        if NACSy is not None: # manual cropping
            acs = kspace[:,:,int(full_sz[1]//2-NACSy/2):int(full_sz[1]//2+NACSy/2),
                             int(full_sz[2]//2-NACSz/2):int(full_sz[2]//2+NACSz/2)]
        else:
            # keep_indices = torch.sum(torch.abs(kspace),(0,)) > 0 # keep only PE lines with nonzero signal (sum over coils and real/imag)
            # acs = kspace[:,:,np.min(keep_indices):np.max(keep_indices)+1,:]
            acs = kspace
        
        afy = accely
        afz = accelz
        af = afy * afz
        nc, nxacs, nyacs, nzacs = acs.shape  
        
        SrcCell, TrgCell, tbly, tblz, shift = get_source_and_target_cell(afy, afz, delta)

        if kernel_size is None: # nothing provided, choose automatically as Philipp Ehses does
            sblx = np.min([9, np.min([nxacs, full_sz[0]])])
            sbly = np.min([np.max([afy*afz, 2*tbly+1]), np.min([nyacs, full_sz[1]])])
            sblz = np.min([np.max([afy*afz, 2*tblz+1]), np.min([nzacs, full_sz[2]])])
        else:
            sblx, sbly, sblz = kernel_size
        

        
        if full_sz[2] == 1: # fix for 2D sequences
            patSrc = np.tile(SrcCell[:,0].T, (sblx, int(np.ceil(sbly/afy/2)*2+1)))
            patSrc = patSrc[:,int(np.floor((patSrc.shape[1] - sbly)/2)):int(np.floor((patSrc.shape[1] + sbly)/2))]
            patTrg = np.zeros((1, sbly))
            patTrg[0, int(np.ceil((sbly - afy)/2)):int(np.ceil((sbly + afy)/2))] = 1
        else:
            patSrc = np.tile(SrcCell.unsqueeze(0),[sblx, int(np.ceil(sbly/af/2)*2+1), int(np.min([np.ceil(sblz/af/2)*2+1, nzacs]))])
            patSrc = patSrc[:,:,0:np.min([patSrc.shape[2], nzacs])]
            patSrc = patSrc[:,int(np.floor((patSrc.shape[1]-sbly)/2)):int(np.floor((patSrc.shape[1]+sbly)/2+1))-1,
                              int(np.floor((patSrc.shape[2]-sblz)/2)):int(np.floor((patSrc.shape[2]+sblz)/2+1))-1]
            
            
            patTrg = np.zeros((sbly,sblz))
            tmp = np.max([0,af-nzacs])
            TrgCell = TrgCell[:,1+int(np.floor(tmp/2))-1:af-int(np.ceil(tmp/2))]
            patTrg[int(np.ceil((sbly-af)/2))-1+1:int(np.ceil((sbly+af)/2)),
                   np.max([1, int(np.ceil((sblz-af)/2))+1])-1:np.min([sblz, int(np.ceil((sblz+af)/2))])] = TrgCell
        
        
        SrcIdx = patSrc.T.flatten() == 1
        TrgIdx = patTrg.T.flatten() == 1
        
        nsp = SrcIdx[SrcIdx==1].shape[0]
        

        #######################
        
        # Predefine the source (src) and the target (trg) matrix 
        # and the position of the 2D target block in the 3D source block
        src = torch.zeros((nc*nsp, (nxacs-sblx+1)*(nyacs-sbly+1)*(nzacs-sblz+1)), dtype=acs.dtype).to(device)
        trg = torch.zeros((nc*af, (nxacs-sblx+1)*(nyacs-sbly+1)*(nzacs-sblz+1)), dtype=acs.dtype).to(device)
        
        #% Step 1: CALCULATION OF THE GRAPPA WEIGHTS
        #% Collect all the source and target replicates 
        #% within the ACS data in the src and trg-matrix
                
        cnt = 0
        for z in range(nzacs-sblz+1):
            for y in range(nyacs-sbly+1):
                for x in range(nxacs-sblx+1):
                    # source points

                    s = acs[:, x:x+sblx, y:y+sbly, z:z+sblz]
                    s = s.permute((0,3,2,1)).reshape((NCoils,-1))[:, SrcIdx]
                    # s = s[:, TrgIdx]
                    
                    # target points
                    
                    t = acs[:, x+int(np.floor((sblx-1)/2)), y:y+sbly, z:z+sblz]
                    t = t.permute((0,2,1)).reshape((NCoils,-1))[:, TrgIdx]
                    
                    if (0 in torch.sum(s,0)) or (0 in torch.sum(t,0)): # check whether there is missing data (e.g. in case elliptical scanning was used in ref. scan)
                        continue
                    
                    

                    src[:,cnt] = s.flatten()
                    trg[:,cnt] = t.flatten()
                    cnt += 1
    
        src = src[:,0:cnt]
        trg = trg[:,0:cnt]
        
        
        # now solve for weights using regularized pseudo inverse
        ws = trg @ pinv_reg(src, lambd)
        ws = ws.reshape((NCoils, af, NCoils, nsp))
        
        # convert to convolution kernel
        ws_tmp = torch.zeros((NCoils, af, NCoils, sblx, sbly, sblz), dtype=acs.dtype).to(device)
        
        # weird trick because of different indixing in Matlab and python: tranpose everything...
        ws_tmp = ws_tmp.T
        if sblz == 1: # 2D case
            ws_tmp[0,patSrc.T==1,:,:,:] = ws.unsqueeze(-1).T
        else:
            ws_tmp[patSrc.T==1,:,:,:] = ws.T # do indexing with transposed indixing arrays and ...
        ws_tmp = ws_tmp.T # transpose back
        
        
        # extend the kernel by the target block size and make sure that the
        # kernel size is odd because we want to keep the center when we flip
        # later
        
        ksz_y = int(np.min([int(np.floor((sbly + tbly - 1)/2))*2 + 1, full_sz[1]]))
        ksz_z = int(np.min([int(np.floor((sblz + tblz - 1)/2))*2 + 1, full_sz[2]]))
        kernel = torch.zeros((NCoils,af,NCoils,sblx,ksz_y,ksz_z), dtype=acs.dtype).to(device)
        ii1 = int(np.floor((ksz_y - sbly)/2))+np.arange(sbly)[:,np.newaxis] # broadcasting trick for indexing arrays, unfortunately harder compared to Matlab
        ii2 = int(np.floor((ksz_z - sblz)/2))+np.arange(sblz)[np.newaxis,:]
        kernel[:,:,:,:,ii1,ii2] = ws_tmp
        
        # Flip the weights in x,y
        kernel = torch.flip(torch.flip(torch.flip(kernel, (3,)), (4,)), (5,))
        
        wsKernel = torch.zeros((NCoils,NCoils,sblx,ksz_y,ksz_z), dtype=acs.dtype).to(device)
        # TEST = np.zeros((8, 8, 9, 7, 3, 1,af)) + 0j
        for k in range(af):
            wsKernel += torch.roll(kernel[:,[k],:,:,:,:].permute(0,2,3,4,5,1),( int(shift[k,0]), int(shift[k,1])),dims=(3,4)).squeeze(-1)
            # TEST[:,:,:,:,:,:,k] = np.roll(kernel[:,[k],:,:,:,:].permute(0,2,3,4,5,1),( int(shift[k,0]), int(shift[k,1])),axis=(3,4))
        
        # import scipy.io as sio
        # sio.savemat('F:/LOCAL_GIT/grappa_tests/3D/debug2.mat',
        #     {'ws_py': ws.cpu().detach().numpy(),
        #      "ws_tmp_py": ws_tmp.cpu().detach().numpy(),
        #      "patSrc_py": patSrc,
        #      'wsKernel_py': wsKernel.cpu().detach().numpy(),
        #      'kernel_py': kernel.cpu().detach().numpy(),
        #      # 'TEST': TEST,
        #       })
        
        
        # from here on image space weights are formed
        nx, ny, nz = full_sz
       
        _, _, nxw, nyw, nzw = wsKernel.shape
        ws_imspace = torch.zeros((NCoils,NCoils,nx,ny,nz),dtype=acs.dtype).to(device)
        ix_x = [0,1]
        ix_y = [0,1]
        ix_z = [0,1]
        if nx > 1:
            ix_x = (torch.arange(0,nxw) + np.floor((nx - nxw)/2)+1).to(int)
        if ny > 1:
            ix_y = (torch.arange(0,nyw) + np.floor((ny - nyw)/2)+1).to(int)
        if nz > 1:
            ix_z = (torch.arange(0,nzw) + np.floor((nz - nzw)/2)+1).to(int)
        
        
        ws_imspace[:,:,int(ix_x[0]):int(ix_x[-1])+1, int(ix_y[0]):int(ix_y[-1])+1, int(ix_z[0]):int(ix_z[-1])+1] = wsKernel
        
        if nz==1: # 2D
            ws_imspace = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ws_imspace, (-3,-2)), dim=(-3,-2)), (-3,-2))
        else:
            ws_imspace = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ws_imspace, (-3,-2,-1)), dim=(-3,-2,-1)), (-3,-2,-1))
        

        return wsKernel, ws_imspace

def grappa_imspace(kspace, ws_imspace):
        '''
        3D implementation of Felix Breuer's grappa_imspace, as included in recoVBVD
        ---------
        
        do grappa reco of scanner signal with previously calculated weight set ws_imspace
        apply grappa weights in image space (multiplication)
        
        kspace: undersampled kspace (zero-filled); NCoils, NKx, NKy, Nkz
        ws_imspace: grappa weight set in image space; NCoils, NCoils, NKx, NKy, NKz
        
        '''
        # kspace = self.get_kmatrix(extraRep)
        # go to image space
        sig = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(kspace, dim=(1,2,3)), dim=(1,2,3), norm="forward"), (1,2,3))

        # apply weights matrix and complex multiplication matrix simultaneously
        sig = torch.einsum('ijklm,jklm->iklm', ws_imspace, sig)
        # sig = sig * torch.prod(torch.from_numpy(self.sz)) # need to figure out scaling again here...
        
        # FG: depending on planetary constellation (Jupiter orthogonal to Mars and so on), there will be random permutes and flips compared to fully sampled reco, which one can try to undo here
        # sig = sig.flip([1,2]).permute([0,2,1,3])
        
        sig = sig * kspace.shape[1]* kspace.shape[2]* kspace.shape[3] * np.sqrt(2) * 2 
        
        return sig