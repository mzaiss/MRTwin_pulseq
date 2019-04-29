import numpy as np
import torch

# HOW we measure
class Scanner():
    def __init__(self,sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu):
        
        self.sz = sz                                             # image size
        self.NVox = sz[0]*sz[1]                                 # voxel count
        self.NSpins = NSpins              # number of spin sims in each voxel
        self.NRep = NRep                              # number of repetitions
        self.T = T                     # number of "actions" within a readout
        self.NCoils = NCoils                # number of receive coil elements
        self.noise_std = noise_std              # additive Gaussian noise std
        
        self.adc_mask = None         # ADC signal acquisition event mask (T,)
        self.rampX = None        # spatial encoding linear gradient ramp (sz)
        self.rampY = None
        self.F = None                              # flip tensor (T,NRep,3,3)
        self.R = None                          # relaxation tensor (NVox,3,3)
        self.P = None                   # free precession tensor (NSpins,3,3)
        self.G = None            # gradient precession tensor (NRep,NVox,3,3)
        self.G_adj = None         # adjoint gradient operator (NRep,NVox,3,3)

        self.B0_grad_cos = None  # accum phase due to gradients (T,NRep,NVox)
        self.B0_grad_sin = None
        self.B0_grad_adj_cos = None  # adjoint grad phase accum (T,NRep,NVox)
        self.B0_grad_adj_sin = None
        
        self.B1 = None          # coil sensitivity profiles (NCoils,NVox,2,2)

        self.signal = None                # measured signal (NCoils,T,NRep,3)
        self.reco =  None                       # reconstructed image (NVox,) 
        
        self.ROI_signal = None            # measured signal (NCoils,T,NRep,3)
        self.ROI_def = 1
        self.lastM = None
        self.AF = None
        self.use_gpu =  use_gpu
        
        
        # experimental (fast grad op)
        self.collect_presignal = False
        
        # phase cycling
        self.phase_cycler = np.array([0,117,351,342,90,315,297,36,252,225,315,162,126,207,45,0,72,261,207,270,90,27,81,252,180,225,27,306,342,135,45,72,216,117,135,270,162,171,297,180,180,297,171,162,270,135,117,216,72,45,135,
                                      342,306,27,225,180,252,81,27,90,270,207,261,72,0,45,207,126,162,315,225,252,36,297,315,90,342,351,117,0,0,117,351,342,90,315,297,36,252,225,315,162,126,207,45,0,72,261,207,270,90,27,81,252,
                                      180,225,27,306,342,135,45,72,216,117,135,270,162,171,297,180,180,297,171,162,270,135,117,216])
        
        # preinit tensors
        # Spatial inhomogeneity component
        S = torch.zeros((1,1,self.NVox,3,3), dtype=torch.float32)
        self.SB0 = self.setdevice(S)
        
        # set linear gradient ramps
        self.init_ramps()
        self.init_intravoxel_dephasing_ramps()
        self.init_coil_sensitivities()
        
    # device setter
    def setdevice(self,x):
        if self.use_gpu > 0:
            x = x.cuda(self.use_gpu-1)
        return x        
        
    def set_adc_mask(self):
        adc_mask = torch.from_numpy(np.ones((self.T,1))).float()
        self.adc_mask = self.setdevice(adc_mask)

    def init_ramps(self):
        use_nonlinear_grads = False                       # very experimental
        
        baserampX = np.linspace(-1,1,self.sz[1] + 1)
        baserampY = np.linspace(-1,1,self.sz[0] + 1)
        
        if use_nonlinear_grads:
            baserampX = np.abs(baserampX)**1.2 * np.sign(baserampX)
            baserampY = np.abs(baserampY)**1.2 * np.sign(baserampY)
            

        # set gradient spatial forms
        rampX = np.pi*baserampX
        rampX = np.expand_dims(rampX[:-1],0)
        rampX = np.tile(rampX, (self.sz[0], 1))
        
        rampX = torch.from_numpy(rampX).float()
        rampX = rampX.view([1,1,self.NVox])                
        
        rampY = np.pi*baserampY
        rampY = np.expand_dims(rampY[:-1],1)
        rampY = np.tile(rampY, (1, self.sz[1]))
        
        rampY = torch.from_numpy(rampY).float()
        rampY = rampY.view([1,1,self.NVox])    
        
        # 1D case
        if self.sz[1] == 1:
            rampX[:,:,:] = 0
        
        self.rampX = self.setdevice(rampX)
        self.rampY = self.setdevice(rampY)
        
    def init_intravoxel_dephasing_ramps(self):
        dim = self.setdevice(torch.sqrt(torch.tensor(self.NSpins).float()))
        
        off = 1 / dim
        if dim == torch.floor(dim):
            xv, yv = torch.meshgrid([torch.linspace(-1+off,1-off,dim.int()), torch.linspace(-1+off,1-off,dim.int())])
            intravoxel_dephasing_ramp = np.pi*torch.stack((xv.flatten(),yv.flatten()),1)
        else:
            class ExecutionControl(Exception): pass
            raise ExecutionControl('init_intravoxel_dephasing_ramps: sqrt(NSpins) should be integer!')
            
            intravoxel_dephasing_ramp = np.pi*2*(torch.rand(self.NSpins,2) - 0.5)
            
        # remove coupling w.r.t. R2
        permvec = np.random.choice(self.NSpins,self.NSpins,replace=False)
        intravoxel_dephasing_ramp = intravoxel_dephasing_ramp[permvec,:]
        #intravoxel_dephasing_ramp = np.pi*2*(torch.rand(self.NSpins,2) - 0.5)
            
        #intravoxel_dephasing_ramp = np.pi*2*(torch.rand(self.NSpins,2) - 0.5)
        intravoxel_dephasing_ramp /= torch.from_numpy(self.sz-1).float().unsqueeze(0)
            
        self.intravoxel_dephasing_ramp = self.setdevice(intravoxel_dephasing_ramp)    
        
    # function is obsolete: deprecate in future, this dummy is for backward compat
    def get_ramps(self):
        pass
        
    def init_coil_sensitivities(self):
        # handle complex mul as matrix mul
        B1 = torch.zeros((self.NCoils,1,self.NVox,2,2), dtype=torch.float32)
        B1[:,0,:,0,0] = 1
        B1[:,0,:,1,1] = 1
        
        self.B1 = self.setdevice(B1)
        
    def init_flip_tensor_holder(self):
        F = torch.zeros((self.T,self.NRep,1,3,3), dtype=torch.float32)
        
        F[:,:,0,1,1] = 1
         
        self.F = self.setdevice(F)
         
    def set_flip_tensor(self,flips):
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        self.F[:,:,0,0,0] = flips_cos
        self.F[:,:,0,0,2] = flips_sin
        self.F[:,:,0,2,0] = -flips_sin
        self.F[:,:,0,2,2] = flips_cos 
        
    def set_flipXY_tensor(self,input_flips):
        
        vx = torch.cos(input_flips[:,:,1])
        vy = torch.sin(input_flips[:,:,1])
        
        theta = input_flips[:,:,0]
            
        theta = theta.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        
        self.F[:,:,0,0,0] = 0
        self.F[:,:,0,0,1] = 0
        self.F[:,:,0,0,2] = vy
        self.F[:,:,0,1,0] = 0
        self.F[:,:,0,1,1] = 0
        self.F[:,:,0,1,2] = -vx
        self.F[:,:,0,2,0] = -vy
        self.F[:,:,0,2,1] = vx
        self.F[:,:,0,2,2] = 0

        # matrix square
        F2 = torch.matmul(self.F,self.F)
        self.F = torch.sin(theta) * self.F + (1 - torch.cos(theta))*F2

        self.F[:,:,0,0,0] += 1
        self.F[:,:,0,1,1] += 1
        self.F[:,:,0,2,2] += 1
        
    # flip operator with B1plus inhomogeneity
    def set_flip_tensor_withB1plus(self,input_flips):
        if not hasattr(self,'B1plus'):
            class ExecutionControl(Exception): pass
            raise ExecutionControl('set_flip_tensor_withB1plus: set B1plus before use')
    
        Fglob = torch.zeros((self.T,self.NRep,1,3,3), dtype=torch.float32)
        
        Fglob[:,:,:,1,1] = 1
         
        Fglob = self.setdevice(Fglob)
        
        vx = torch.cos(input_flips[:,:,1])
        vy = torch.sin(input_flips[:,:,1])
        
        Fglob[:,:,0,0,0] = 0
        Fglob[:,:,0,0,1] = 0
        Fglob[:,:,0,0,2] = vy
        Fglob[:,:,0,1,0] = 0
        Fglob[:,:,0,1,1] = 0
        Fglob[:,:,0,1,2] = -vx
        Fglob[:,:,0,2,0] = -vy
        Fglob[:,:,0,2,1] = vx
        Fglob[:,:,0,2,2] = 0
    
        # matrix square
        F2 = torch.matmul(Fglob,Fglob)
        
        theta = input_flips[:,:,0]
        theta = theta.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        
        theta = theta * self.B1plus.view([1,1,self.NVox,1,1])
        
        F = torch.sin(theta)*Fglob + (1 - torch.cos(theta))*F2
        self.F = self.setdevice(F)       
    
        self.F[:,:,:,0,0] += 1
        self.F[:,:,:,1,1] += 1
        self.F[:,:,:,2,2] += 1
        
    # use Rodriguez' rotation formula to compute rotation around arbitrary axis
    # flips are now (T,NRep,3) -- axis angle representation
    # angle = norm of the rotation vector    
    def set_flipAxisAngle_tensor(self,flips):
        
        # ... greatly simplifies if assume rotations in XY plane ...
        theta = torch.norm(flips,dim=2).unsqueeze(2)
        v = flips / theta
        theta = theta.unsqueeze(2).unsqueeze(2)
        
        self.F[:,:,0,0,0] = 0
        self.F[:,:,0,0,1] = -v[:,:,2]
        self.F[:,:,0,0,2] = v[:,:,1]
        self.F[:,:,0,1,0] = v[:,:,2]
        self.F[:,:,0,1,1] = 0
        self.F[:,:,0,1,2] = -v[:,:,0]
        self.F[:,:,0,2,0] = -v[:,:,1]
        self.F[:,:,0,2,1] = v[:,:,0]
        self.F[:,:,0,2,2] = 0
        
        # matrix square
        F2 = torch.matmul(self.F,self.F)
        self.F = torch.sin(theta) * self.F + (1 - torch.cos(theta))*F2
        
        self.F[:,:,0,0,0] += 1
        self.F[:,:,0,1,1] += 1
        self.F[:,:,0,2,2] += 1  
        
    # rotate ADC phase to conform phase of the excitation
    def set_ADC_rot_tensor(self,input_phase):
        AF = torch.zeros((self.NRep,3,3))
        
        AF[:,0,0] = torch.cos(input_phase)
        AF[:,0,1] = -torch.sin(input_phase)
        AF[:,1,0] = torch.sin(input_phase)
        AF[:,1,1] = torch.cos(input_phase)
        
        self.AF = self.setdevice(AF)
        
    def set_relaxation_tensor(self,spins,dt):
        R = torch.zeros((self.NVox,3,3), dtype=torch.float32) 
        
        R = self.setdevice(R)
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,0,0] = T2_r
        R[:,1,1] = T2_r
        R[:,2,2] = T1_r
         
        R = R.view([1,self.NVox,3,3])
        
        self.R = R
        
    def set_freeprecession_tensor(self,spins,dt):
        P = torch.zeros((self.NSpins,1,1,3,3), dtype=torch.float32)
        P = self.setdevice(P)
        
        B0_nspins = spins.omega[:,0].view([self.NSpins])
        
        B0_nspins_cos = torch.cos(B0_nspins*dt)
        B0_nspins_sin = torch.sin(B0_nspins*dt)
         
        P[:,0,0,0,0] = B0_nspins_cos
        P[:,0,0,0,1] = -B0_nspins_sin
        P[:,0,0,1,0] = B0_nspins_sin
        P[:,0,0,1,1] = B0_nspins_cos
         
        P[:,0,0,2,2] = 1
         
        self.P = P
        
    def set_B0inhomogeneity_tensor(self,spins,delay):
        S = torch.zeros((1,1,self.NVox,3,3), dtype=torch.float32)
        S = self.setdevice(S)
        
        B0_inhomo = spins.B0inhomo.view([self.NVox]) * 2*np.pi
        
        B0_nspins_cos = torch.cos(B0_inhomo*delay)
        B0_nspins_sin = torch.sin(B0_inhomo*delay)
         
        S[0,0,:,0,0] = B0_nspins_cos
        S[0,0,:,0,1] = -B0_nspins_sin
        S[0,0,:,1,0] = B0_nspins_sin
        S[0,0,:,1,1] = B0_nspins_cos
         
        S[0,0,:,2,2] = 1
         
        self.SB0 = S
        
    
    def init_gradient_tensor_holder(self):
        G = torch.zeros((self.T,self.NVox,3,3), dtype=torch.float32)
        G[:,:,2,2] = 1
         
        G_adj = torch.zeros((self.T,self.NVox,3,3), dtype=torch.float32)
        G_adj[:,:,2,2] = 1
         
        self.G = self.setdevice(G)
        self.G_adj = self.setdevice(G_adj)
        
        # intravoxel precession
        IVP = torch.zeros((self.NSpins,1,1,3,3), dtype=torch.float32)
        IVP = self.setdevice(IVP)
        
        IVP[:,0,0,2,2] = 1
        
        self.IVP = IVP        
        
    def set_grad_op(self,r):
        
        self.G[:,:,0,0] = self.B0_grad_cos[:,r,:]
        self.G[:,:,0,1] = -self.B0_grad_sin[:,r,:]
        self.G[:,:,1,0] = self.B0_grad_sin[:,r,:]
        self.G[:,:,1,1] = self.B0_grad_cos[:,r,:]
        
    def set_grad_adj_op(self,r):
        
        self.G_adj[:,:,0,0] = self.B0_grad_adj_cos[:,r,:]
        self.G_adj[:,:,0,1] = self.B0_grad_adj_sin[:,r,:]
        self.G_adj[:,:,1,0] = -self.B0_grad_adj_sin[:,r,:]
        self.G_adj[:,:,1,1] = self.B0_grad_adj_cos[:,r,:]
        
    def set_gradient_precession_tensor(self,grad_moms,refocusing=False,wrap_k=False):
        grads=grad_moms
        
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
        temp = temp[:-1,:,:]
        k=torch.cumsum(temp,0)
        
        B0X = torch.unsqueeze(grads[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grads[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_cos = torch.cos(B0_grad)
        self.B0_grad_sin = torch.sin(B0_grad)
        
        if wrap_k:
            hx = self.sz[0]/2
            k[:,:,0] = torch.fmod(k[:,:,0]+hx,hx*2)-hx        
        
        # for backward pass
        if refocusing:
            B0X = torch.unsqueeze(k[:,:,0]-self.sz[0]/2,2) * self.rampX
        else:
            B0X = torch.unsqueeze(k[:,:,0],2) * self.rampX
            
        B0Y = torch.unsqueeze(k[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_adj_cos = torch.cos(B0_grad)
        self.B0_grad_adj_sin = torch.sin(B0_grad)
        
        # save grad_moms for intravoxel precession op
        self.grad_moms_for_intravoxel_precession = grad_moms    
        
    def flip(self,t,r,spins):
        spins.M = torch.matmul(self.F[t,r,:,:,:],spins.M)
        
    # apply flip at all repetition simultanetously (non TR transfer case)
    def flip_allRep(self,t,spins):
        spins.M = torch.matmul(self.F[t,:,:,:,:],spins.M)
        
    def relax(self,spins):
        spins.M = torch.matmul(self.R,spins.M)
        spins.M[:,:,:,2,0] += (1 - self.R[:,:,2,2]).view([1,1,self.NVox]) * spins.MZ0
        
    def relax_and_dephase(self,spins):
        
        spins.M = torch.matmul(self.R,spins.M)
        spins.M[:,:,:,2,0] += (1 - self.R[:,:,2,2]).view([1,1,self.NVox]) * spins.MZ0
        
        spins.M = torch.matmul(self.SB0,spins.M)
        spins.M = torch.matmul(self.P,spins.M)
        
    def grad_precess(self,t,spins):
        spins.M = torch.matmul(self.G[t,:,:,:],spins.M)
        
    def set_grad_intravoxel_precess_tensor(self,t,r):
        intra_b0 = self.grad_moms_for_intravoxel_precession[t,r,:].unsqueeze(0) * self.intravoxel_dephasing_ramp
        intra_b0 = torch.sum(intra_b0,1)
        
        IVP_nspins_cos = torch.cos(intra_b0)
        IVP_nspins_sin = torch.sin(intra_b0)
         
        self.IVP[:,0,0,0,0] = IVP_nspins_cos
        self.IVP[:,0,0,0,1] = -IVP_nspins_sin
        self.IVP[:,0,0,1,0] = IVP_nspins_sin
        self.IVP[:,0,0,1,1] = IVP_nspins_cos
   
        
    # intravoxel gradient-driven precession
    def grad_intravoxel_precess(self,t,r,spins):
        self.set_grad_intravoxel_precess_tensor(t,r)
        spins.M = torch.matmul(self.IVP,spins.M)
        
        
    def init_signal(self):
        signal = torch.zeros((self.NCoils,self.T,self.NRep,3,1), dtype=torch.float32) 
        #signal[:,:,:,2:,0] = 1                                 # aux dim zero ()
        self.signal = self.setdevice(signal)
        
        if self.collect_presignal:
            presignal = torch.zeros((self.NRep,1,self.NVox,3,1), dtype=torch.float32) 
            self.presignal = self.setdevice(presignal)
        
        self.ROI_signal = torch.zeros((self.T,self.NRep,5), dtype=torch.float32) # for trans magnetization
        self.ROI_signal = self.setdevice(self.ROI_signal)
        self.ROI_def= int((self.sz[0]/2)*self.sz[1]+ self.sz[1]/2)
        
    def init_reco(self):
        reco = torch.zeros((self.NVox,2), dtype = torch.float32)
        
        self.reco = self.setdevice(reco)
        
    def do_ifft_reco(self):
        spectrum = self.signal
        spectrum = spectrum[0,self.adc_mask.flatten()!=0,:,:2,0]
        space = torch.ifft(spectrum,2)
        output = space.clone()
        output[:self.sz[0]//2,:self.sz[1]//2,:] = space[-self.sz[0]//2:,-self.sz[1]//2:,:]
        output[:self.sz[0]//2,-self.sz[1]//2:,:] = space[-self.sz[0]//2:,:self.sz[1]//2,:]
        output[-self.sz[0]//2:,:self.sz[1]//2,:] = space[:self.sz[0]//2,-self.sz[1]//2:,:]
        output[-self.sz[0]//2:,-self.sz[1]//2:,:] = space[:self.sz[0]//2,:self.sz[1]//2,:] 
        
        return output
        
    def read_signal(self,t,r,spins):
        if self.adc_mask[t] != 0:
            
            # parallel imaging disabled for now
            #sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(0).unsqueeze(4))
            
            if self.noise_std > 0:
                sig = torch.sum(spins.M[:,:,:,:2],[0])
                noise = self.noise_std*torch.randn(sig.shape).float()
                noise = self.setdevice(noise)
                sig += noise  
            
                self.signal[0,t,r,:2] = ((torch.sum(sig,[0,1]) * self.adc_mask[t])) / self.NSpins
            else:
                self.signal[0,t,r,:2] = ((torch.sum(spins.M[:,:,:,:2],[0,1,2]) * self.adc_mask[t])) / self.NSpins  
                
    # run throw all repetition/actions and yield signal
    def forward(self,spins,event_time,do_dummy_scans=False,compact_grad_tensor=True):
        self.init_signal()
        if do_dummy_scans == False:
            spins.set_initial_magnetization()
    
        # scanner forward process loop
        for r in range(self.NRep):                                   # for all repetitions
            if compact_grad_tensor:
                self.set_grad_op(r)
            
            for t in range(self.T):                                      # for all actions
                delay = torch.abs(event_time[t,r]) + 1e-6
                self.read_signal(t,r,spins)    
                
                self.ROI_signal[t,r,0] =   delay
                self.ROI_signal[t,r,1:4] =  torch.sum(spins.M[:,0,self.ROI_def,:],[0]).flatten().detach().cpu()  # hard coded center pixel
                self.ROI_signal[t,r,4] =  torch.sum(torch.abs(spins.M[:,0,self.ROI_def,2]),[0]).flatten().detach().cpu()  # hard coded center pixel                 
                
                self.flip(t,r,spins)
                
                self.set_relaxation_tensor(spins,delay)
                self.set_freeprecession_tensor(spins,delay)
                self.set_B0inhomogeneity_tensor(spins,delay)
                self.relax_and_dephase(spins)
                    
                if compact_grad_tensor:
                    self.grad_precess(t,spins)
                else:
                    self.grad_precess(t,r,spins)
                    
                self.grad_intravoxel_precess(t,r,spins)
               
                
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)    
            
    def forward_mem(self,spins,event_time,do_dummy_scans=False,compact_grad_tensor=True):
        self.init_signal()
        if do_dummy_scans == False:
            spins.set_initial_magnetization()
    
        # scanner forward process loop
        for r in range(self.NRep):                                   # for all repetitions
            if compact_grad_tensor:
                self.set_grad_op(r)
                
            for t in range(self.T):                                      # for all actions
                delay = torch.abs(event_time[t,r]) + 1e-6
                self.read_signal(t,r,spins)
                
                self.ROI_signal[t,r,0] =   delay
                self.ROI_signal[t,r,1:4] =  torch.sum(spins.M[:,0,self.ROI_def,:],[0]).flatten().detach().cpu()  # hard coded center pixel
                self.ROI_signal[t,r,4] =  torch.sum(abs(spins.M[:,0,self.ROI_def,2]),[0]).flatten().detach().cpu()  # hard coded center pixel    
                
                spins.M = FlipClass.apply(self.F[t,r,:,:,:],spins.M,self)
                
                self.set_relaxation_tensor(spins,delay)
                self.set_freeprecession_tensor(spins,delay)
                self.set_B0inhomogeneity_tensor(spins,delay)
                
                spins.M = RelaxClass.apply(self.R,spins.M,delay,t,self,spins)
                spins.M = DephaseClass.apply(self.P,spins.M,self)
                spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                
                if compact_grad_tensor:
                    spins.M = GradPrecessClass.apply(self.G[t,:,:,:],spins.M,self)
                else:
                    spins.M = GradPrecessClass.apply(self.G[t,r,:,:,:],spins.M,self)

                self.set_grad_intravoxel_precess_tensor(t,r)
                spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                
                
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox)).byte()
        self.tmask = self.setdevice(self.tmask)
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.lastM = spins.M.clone()
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)
            
    def forward_sparse(self,spins,event_time,do_dummy_scans=False,compact_grad_tensor=True):
        self.init_signal()
        if do_dummy_scans == False:
            spins.set_initial_magnetization()
        
        PD0_mask = spins.PD0_mask.flatten()
        spins_cut = spins.M[:,:,PD0_mask,:,:].clone()
        
        if not compact_grad_tensor:
            G_cut = self.G[:,:,PD0_mask,:,:]
        
        # scanner forward process loop
        for r in range(self.NRep):                                   # for all repetitions
            if compact_grad_tensor:
                self.set_grad_op(r)
                G_cut = self.G[:,PD0_mask,:,:]
                
            for t in range(self.T):                                      # for all actions
                delay = torch.abs(event_time[t,r]) + 1e-6
                
                if self.adc_mask[t] != 0:
                    self.signal[0,t,r,:2] = ((torch.sum(spins_cut[:,:,:,:2],[0,1,2]) * self.adc_mask[t])) / self.NSpins 
                    
                self.ROI_signal[t,r,0] =   delay
                self.ROI_signal[t,r,1:4] =  torch.sum(spins_cut[:,0,0,:],[0]).flatten().detach().cpu()  # hard coded pixel id 0
                self.ROI_signal[t,r,4] =  torch.sum(abs(spins_cut[:,0,0,2]),[0]).flatten().detach().cpu()  # hard coded pixel id 0         
        
                spins_cut = FlipClass.apply(self.F[t,r,:,:,:],spins_cut,self)
                
                self.set_relaxation_tensor(spins,delay)
                self.set_freeprecession_tensor(spins,delay)
                self.set_B0inhomogeneity_tensor(spins,delay)
                
                spins_cut = RelaxSparseClass.apply(self.R[:,PD0_mask,:,:], spins_cut,delay,t,self,spins,PD0_mask)
                spins_cut = DephaseClass.apply(self.P,spins_cut,self)
                spins_cut = B0InhomoClass.apply(self.SB0[:,:,PD0_mask,:,:],spins_cut,self)
                
                if compact_grad_tensor:
                    spins_cut = GradPrecessClass.apply(G_cut[t,:,:,:],spins_cut,self)
                else:
                    spins_cut = GradPrecessClass.apply(G_cut[t,r,:,:,:],spins_cut,self)
                
                self.set_grad_intravoxel_precess_tensor(t,r)
                spins_cut = GradIntravoxelPrecessClass.apply(self.IVP,spins_cut,self)
                
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox)).byte()
        self.tmask = self.setdevice(self.tmask)
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.tmask = self.tmask[PD0_mask]
        
        self.lastM = spins_cut.clone()
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)   

    # run throw all repetition/actions and yield signal
    def forward_fast(self,spins,event_time,do_dummy_scans=False,compact_grad_tensor=True):
        self.init_signal()
        if do_dummy_scans == False:
            spins.set_initial_magnetization()
        self.reco = 0
        
        half_read = np.int(torch.sum(self.adc_mask != 0) / 2)
        
        # scanner forward process loop
        for r in range(self.NRep):                         # for all repetitions
            total_delay = 0
            start_t = 0
            
            if compact_grad_tensor:
                self.set_grad_op(r)
                self.set_grad_adj_op(r)
                
            for t in range(self.T):                            # for all actions
                delay = torch.abs(event_time[t,r]) + 1e-6
                
                if self.adc_mask[t] == 0:                         # regular pass
                    self.read_signal(t,r,spins)
                    
                    self.ROI_signal[t,r,0] =   delay
                    self.ROI_signal[t,r,1:4] =  torch.sum(spins.M[:,0,self.ROI_def,:],[0]).flatten().detach().cpu()  # hard coded center pixel
                    self.ROI_signal[t,r,4] =  torch.sum(abs(spins.M[:,0,self.ROI_def,2]),[0]).flatten().detach().cpu()  # hard coded center pixel                     
                    
                    spins.M = FlipClass.apply(self.F[t,r,:,:,:],spins.M,self)
                    
                    self.set_relaxation_tensor(spins,delay)
                    self.set_freeprecession_tensor(spins,delay)
                    self.set_B0inhomogeneity_tensor(spins,delay)
                    
                    spins.M = RelaxClass.apply(self.R,spins.M,delay,t,self,spins)
                    spins.M = DephaseClass.apply(self.P,spins.M,self)
                    spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                    
                    if compact_grad_tensor:
                        spins.M = GradPrecessClass.apply(self.G[t,:,:,:],spins.M,self)
                    else:
                        spins.M = GradPrecessClass.apply(self.G[t,r,:,:,:],spins.M,self)                

                    self.set_grad_intravoxel_precess_tensor(t,r)
                    spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                else:
                    if self.adc_mask[t-1] == 0:        # first sample in readout
                        total_delay = delay
                        start_t = t
                        
                    elif t == (self.T - half_read*2)//2 + half_read or self.adc_mask[t+1] == 0:
                        self.ROI_signal[start_t:t+1,r,0] = delay
                        self.ROI_signal[start_t:t+1,r,1:4] = torch.sum(spins.M[:,0,self.ROI_def,:],[0]).flatten().detach().cpu().unsqueeze(0)
                        self.ROI_signal[start_t:t+1,r,4] = torch.sum(abs(spins.M[:,0,self.ROI_def,2]),[0]).flatten().detach().cpu()
                        
                        self.set_relaxation_tensor(spins,total_delay)
                        self.set_freeprecession_tensor(spins,total_delay)
                        self.set_B0inhomogeneity_tensor(spins,total_delay)
                        
                        spins.M = RelaxClass.apply(self.R,spins.M,total_delay,t,self,spins)
                        spins.M = DephaseClass.apply(self.P,spins.M,self)
                        spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                        
                        if compact_grad_tensor:
                            REW = self.G_adj[start_t,:,:,:]
                        else:
                            REW = self.G_adj[start_t,r,:,:,:]
                            
                        # set grad intravoxel precession tensor
                        kum_grad_intravoxel = torch.sum(self.grad_moms_for_intravoxel_precession[start_t:t+1,r,:],0)
                        intra_b0 = kum_grad_intravoxel.unsqueeze(0) * self.intravoxel_dephasing_ramp
                        intra_b0 = torch.sum(intra_b0,1)
                        
                        IVP_nspins_cos = torch.cos(intra_b0)
                        IVP_nspins_sin = torch.sin(intra_b0)
                         
                        self.IVP[:,0,0,0,0] = IVP_nspins_cos
                        self.IVP[:,0,0,0,1] = -IVP_nspins_sin
                        self.IVP[:,0,0,1,0] = IVP_nspins_sin
                        self.IVP[:,0,0,1,1] = IVP_nspins_cos                
                        
                        # do intravoxel precession
                        spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                        
                        # read signal
                        if t == (self.T - half_read*2)//2 + half_read:  # read signal
                            if compact_grad_tensor:
                                FWD = self.G_adj[start_t:start_t+half_read*2,:,:,:].permute([0,1,3,2])
                            else:
                                FWD = self.G_adj[start_t:start_t+half_read*2,r,:,:,:].permute([0,1,3,2])
                                
                            FWD = torch.matmul(REW, FWD)
                            
                            if self.collect_presignal:
                                presignal = torch.sum(spins.M,0,keepdim=True)
                                self.presignal[r,0,:,:,:] = presignal
                                signal = torch.matmul(FWD,presignal)
                            else:
                                signal = torch.matmul(FWD,torch.sum(spins.M,0,keepdim=True))
                                
                            signal = torch.sum(signal,[2])
                            signal *= self.adc_mask[start_t:start_t+half_read*2].view([1,signal.shape[1],1,1])
                            
                            self.signal[0,start_t:start_t+half_read*2,r,:,0] = signal.squeeze() / self.NSpins 
                            
                        # do gradient precession (use adjoint as free kumulator)
                        if compact_grad_tensor:
                            FWD = self.G_adj[t+1,:,:,:].permute([0,2,1])
                        else:
                            FWD = self.G_adj[t+1,r,:,:,:].permute([0,2,1])
                            
                        FWD = torch.matmul(REW, FWD)
                        spins.M = torch.matmul(FWD,spins.M)                       
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay

                    
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox)).byte()
        self.tmask = self.setdevice(self.tmask)
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.lastM = spins.M.clone() 
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)
            
    # run throw all repetition/actions and yield signal
    def forward_sparse_fast(self,spins,event_time,do_dummy_scans=False,compact_grad_tensor=True):
        self.init_signal()
        if do_dummy_scans == False:
            spins.set_initial_magnetization()
        self.reco = 0
        
        half_read = np.int(torch.sum(self.adc_mask != 0) / 2)
        
        PD0_mask = spins.PD0_mask.flatten()
        spins_cut = spins.M[:,:,PD0_mask,:,:].clone()
        
        # scanner forward process loop
        for r in range(self.NRep):                         # for all repetitions
            total_delay = 0
            start_t = 0
            
            if compact_grad_tensor:
                self.set_grad_op(r)
                self.set_grad_adj_op(r)
                
                G_cut = self.G[:,PD0_mask,:,:]        
                G_adj_cut = self.G_adj[:,PD0_mask,:,:]                
            else:        
                G_cut = self.G[:,:,PD0_mask,:,:]        
                G_adj_cut = self.G_adj[:,:,PD0_mask,:,:]
            
            for t in range(self.T):                            # for all actions
                delay = torch.abs(event_time[t,r]) + 1e-6
                
                if self.adc_mask[t] == 0:                         # regular pass
                    self.read_signal(t,r,spins)
                    
                    self.ROI_signal[t,r,0] =   delay
                    self.ROI_signal[t,r,1:4] =  torch.sum(spins_cut[:,0,0,:],[0]).flatten().detach().cpu()  # hard coded pixel id 0
                    self.ROI_signal[t,r,4] =  torch.sum(abs(spins_cut[:,0,0,2]),[0]).flatten().detach().cpu()  # hard coded pixel id 0                        
                    
                    spins_cut = FlipClass.apply(self.F[t,r,:,:,:],spins_cut,self)
                    
                    self.set_relaxation_tensor(spins,delay)
                    self.set_freeprecession_tensor(spins,delay)
                    self.set_B0inhomogeneity_tensor(spins,delay)
                    
                    spins_cut = RelaxSparseClass.apply(self.R[:,PD0_mask,:,:],spins_cut,delay,t,self,spins,PD0_mask)
                    spins_cut = DephaseClass.apply(self.P,spins_cut,self)
                    spins_cut = B0InhomoClass.apply(self.SB0[:,:,PD0_mask,:,:],spins_cut,self)
                    
                    if compact_grad_tensor:
                        spins_cut = GradPrecessClass.apply(G_cut[t,:,:,:],spins_cut,self)
                    else:
                        spins_cut = GradPrecessClass.apply(G_cut[t,r,:,:,:],spins_cut,self)
                    
                    self.set_grad_intravoxel_precess_tensor(t,r)
                    spins_cut = GradIntravoxelPrecessClass.apply(self.IVP,spins_cut,self)
                else:
                    if self.adc_mask[t-1] == 0:        # first sample in readout
                        total_delay = delay
                        start_t = t
                        
                    elif t == (self.T - half_read*2)//2 + half_read or self.adc_mask[t+1] == 0:
                        self.ROI_signal[start_t:t+1,r,0] = total_delay
                        self.ROI_signal[start_t:t+1,r,1:4] = torch.sum(spins_cut[:,0,0,:],[0]).flatten().detach().cpu().unsqueeze(0)
                        self.ROI_signal[start_t:t+1,r,4] = torch.sum(abs(spins_cut[:,0,0,2]),[0]).flatten().detach().cpu()
                        
                        self.set_relaxation_tensor(spins,total_delay)
                        self.set_freeprecession_tensor(spins,total_delay)
                        self.set_B0inhomogeneity_tensor(spins,total_delay)
                        
                        spins_cut = RelaxSparseClass.apply(self.R[:,PD0_mask,:,:], spins_cut,total_delay,t,self,spins,PD0_mask)
                        spins_cut = DephaseClass.apply(self.P,spins_cut,self)
                        spins_cut = B0InhomoClass.apply(self.SB0[:,:,PD0_mask,:,:],spins_cut,self)
                        
                        # do gradient precession (use adjoint as free kumulator)
                        if compact_grad_tensor:
                            REW = G_adj_cut[start_t,:,:,:]
                        else:
                            REW = G_adj_cut[start_t,r,:,:,:]
                        
                        # set grad intravoxel precession tensor
                        kum_grad_intravoxel = torch.sum(self.grad_moms_for_intravoxel_precession[start_t:t+1,r,:],0)
                        intra_b0 = kum_grad_intravoxel.unsqueeze(0) * self.intravoxel_dephasing_ramp
                        intra_b0 = torch.sum(intra_b0,1)
                        
                        IVP_nspins_cos = torch.cos(intra_b0)
                        IVP_nspins_sin = torch.sin(intra_b0)
                         
                        self.IVP[:,0,0,0,0] = IVP_nspins_cos
                        self.IVP[:,0,0,0,1] = -IVP_nspins_sin
                        self.IVP[:,0,0,1,0] = IVP_nspins_sin
                        self.IVP[:,0,0,1,1] = IVP_nspins_cos                
                        
                        # do intravoxel precession
                        spins_cut = GradIntravoxelPrecessClass.apply(self.IVP,spins_cut,self)
                        
                        # read signal
                        if t == (self.T - half_read*2)//2 + half_read:  # read signal
                            if compact_grad_tensor:
                                FWD = G_adj_cut[start_t:start_t+half_read*2,:,:,:].permute([0,1,3,2])
                            else:
                                FWD = G_adj_cut[start_t:start_t+half_read*2,r,:,:,:].permute([0,1,3,2])
                                
                            FWD = torch.matmul(REW, FWD)
                            
                            signal = torch.matmul(FWD,torch.sum(spins_cut,0,keepdim=True))
                            signal = torch.sum(signal,[2])
                            
                            self.signal[0,start_t:start_t+half_read*2,r,:,0] = signal.squeeze() / self.NSpins 
                            
                        # do gradient precession (use adjoint as free kumulator)
                        if compact_grad_tensor:
                            FWD = G_adj_cut[t+1,:,:,:].permute([0,2,1])
                        else:
                            FWD = G_adj_cut[t+1,r,:,:,:].permute([0,2,1])
                            
                        FWD = torch.matmul(REW, FWD)
                        spins_cut = torch.matmul(FWD,spins_cut)                            
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay
                    
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox)).byte()
        self.tmask = self.setdevice(self.tmask)
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.tmask = self.tmask[PD0_mask]
        self.lastM = spins_cut.clone()
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)
            
    def do_dummy_scans(self,spins,event_time,compact_grad_tensor=True,nrep=0):
        spins.set_initial_magnetization()
        
        for nr in range(nrep):
            print('doing dummy #{} ...'.format(nr))        
        
            half_read = np.int(torch.sum(self.adc_mask != 0) / 2)
            
            # scanner forward process loop
            for r in range(self.NRep):                         # for all repetitions
                total_delay = 0
                start_t = 0
                
                if compact_grad_tensor:
                    self.set_grad_op(r)
                    self.set_grad_adj_op(r)
                    
                for t in range(self.T):                            # for all actions
                    delay = torch.abs(event_time[t,r]) + 1e-6
                    
                    if self.adc_mask[t] == 0:                         # regular pass
                        spins.M = FlipClass.apply(self.F[t,r,:,:,:],spins.M,self)
                        
                        self.set_relaxation_tensor(spins,delay)
                        self.set_freeprecession_tensor(spins,delay)
                        self.set_B0inhomogeneity_tensor(spins,delay)
                        
                        spins.M = RelaxClass.apply(self.R,spins.M,delay,t,self,spins)
                        spins.M = DephaseClass.apply(self.P,spins.M,self)
                        spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                        
                        if compact_grad_tensor:
                            spins.M = GradPrecessClass.apply(self.G[t,:,:,:],spins.M,self)
                        else:
                            spins.M = GradPrecessClass.apply(self.G[t,r,:,:,:],spins.M,self)                
    
                        self.set_grad_intravoxel_precess_tensor(t,r)
                        spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                    else:
                        if self.adc_mask[t-1] == 0:        # first sample in readout
                            total_delay = delay
                            start_t = t
                            
                        elif t == (self.T - half_read*2)//2 + half_read or self.adc_mask[t+1] == 0:
                            self.set_relaxation_tensor(spins,total_delay)
                            self.set_freeprecession_tensor(spins,total_delay)
                            self.set_B0inhomogeneity_tensor(spins,total_delay)
                            
                            spins.M = RelaxClass.apply(self.R,spins.M,total_delay,t,self,spins)
                            spins.M = DephaseClass.apply(self.P,spins.M,self)
                            spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                            
                            # do gradient precession (use adjoint as free kumulator)
                            if compact_grad_tensor:
                                REW = self.G_adj[start_t,:,:,:]
                                FWD = self.G_adj[t+1,:,:,:].permute([0,2,1])
                            else:
                                REW = self.G_adj[start_t,r,:,:,:]
                                FWD = self.G_adj[t+1,r,:,:,:].permute([0,2,1])
                                
                            FWD = torch.matmul(REW, FWD)
                            spins.M = torch.matmul(FWD,spins.M)
                            
                            # set grad intravoxel precession tensor
                            kum_grad_intravoxel = torch.sum(self.grad_moms_for_intravoxel_precession[start_t:t+1,r,:],0)
                            intra_b0 = kum_grad_intravoxel.unsqueeze(0) * self.intravoxel_dephasing_ramp
                            intra_b0 = torch.sum(intra_b0,1)
                            
                            IVP_nspins_cos = torch.cos(intra_b0)
                            IVP_nspins_sin = torch.sin(intra_b0)
                             
                            self.IVP[:,0,0,0,0] = IVP_nspins_cos
                            self.IVP[:,0,0,0,1] = -IVP_nspins_sin
                            self.IVP[:,0,0,1,0] = IVP_nspins_sin
                            self.IVP[:,0,0,1,1] = IVP_nspins_cos                
                            
                            # do intravoxel precession
                            spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                            
                            # reset readout position tracking vars
                            start_t = t + 1
                            total_delay = delay
                        else:                                       # keep accumulating
                            total_delay += delay
                    
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox)).byte()
        self.tmask = self.setdevice(self.tmask)
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.lastM = spins.M.clone() 
            
    def do_dummy_scans_sparse(self,spins,event_time,compact_grad_tensor=True,nrep=0):
        spins.set_initial_magnetization()
        
        for nr in range(nrep):
            print('doing dummy #{} ...'.format(nr)) 
                  
            PD0_mask = spins.PD0_mask.flatten()
            spins_cut = spins.M[:,:,PD0_mask,:,:].clone()
            
            # scanner forward process loop
            half_read = np.int(torch.sum(self.adc_mask != 0) / 2)
            
            PD0_mask = spins.PD0_mask.flatten()
            spins_cut = spins.M[:,:,PD0_mask,:,:].clone()
            
            # scanner forward process loop
            for r in range(self.NRep):                         # for all repetitions
                total_delay = 0
                start_t = 0
                
                if compact_grad_tensor:
                    self.set_grad_op(r)
                    self.set_grad_adj_op(r)
                    
                    G_cut = self.G[:,PD0_mask,:,:]        
                    G_adj_cut = self.G_adj[:,PD0_mask,:,:]                
                else:        
                    G_cut = self.G[:,:,PD0_mask,:,:]        
                    G_adj_cut = self.G_adj[:,:,PD0_mask,:,:]
                
                for t in range(self.T):                            # for all actions
                    delay = torch.abs(event_time[t,r]) + 1e-6
                    
                    if self.adc_mask[t] == 0:                         # regular pass
                        spins_cut = FlipClass.apply(self.F[t,r,:,:,:],spins_cut,self)
                        
                        self.set_relaxation_tensor(spins,delay)
                        self.set_freeprecession_tensor(spins,delay)
                        self.set_B0inhomogeneity_tensor(spins,delay)
                        
                        spins_cut = RelaxSparseClass.apply(self.R[:,PD0_mask,:,:],spins_cut,delay,t,self,spins,PD0_mask)
                        spins_cut = DephaseClass.apply(self.P,spins_cut,self)
                        spins_cut = B0InhomoClass.apply(self.SB0[:,:,PD0_mask,:,:],spins_cut,self)
                        
                        if compact_grad_tensor:
                            spins_cut = GradPrecessClass.apply(G_cut[t,:,:,:],spins_cut,self)
                        else:
                            spins_cut = GradPrecessClass.apply(G_cut[t,r,:,:,:],spins_cut,self)
                        
                        self.set_grad_intravoxel_precess_tensor(t,r)
                        spins_cut = GradIntravoxelPrecessClass.apply(self.IVP,spins_cut,self)
                    else:
                        if self.adc_mask[t-1] == 0:        # first sample in readout
                            total_delay = delay
                            start_t = t
                            
                        elif t == (self.T - half_read*2)//2 + half_read or self.adc_mask[t+1] == 0:
                            self.set_relaxation_tensor(spins,total_delay)
                            self.set_freeprecession_tensor(spins,total_delay)
                            self.set_B0inhomogeneity_tensor(spins,total_delay)
                            
                            spins_cut = RelaxSparseClass.apply(self.R[:,PD0_mask,:,:],spins_cut,delay,t,self,spins,PD0_mask)
                            spins_cut = DephaseClass.apply(self.P,spins_cut,self)
                            spins_cut = B0InhomoClass.apply(self.SB0[:,:,PD0_mask,:,:],spins_cut,self)
                            
                            # do gradient precession (use adjoint as free kumulator)
                            if compact_grad_tensor:
                                REW = G_adj_cut[start_t,:,:,:]
                                FWD = G_adj_cut[t+1,:,:,:].permute([0,2,1])
                            else:
                                REW = G_adj_cut[start_t,r,:,:,:]
                                FWD = G_adj_cut[t+1,r,:,:,:].permute([0,2,1])
                                
                            FWD = torch.matmul(REW, FWD)
                            spins_cut = torch.matmul(FWD,spins_cut)
                            
                            # set grad intravoxel precession tensor
                            kum_grad_intravoxel = torch.sum(self.grad_moms_for_intravoxel_precession[start_t:t+1,r,:],0)
                            intra_b0 = kum_grad_intravoxel.unsqueeze(0) * self.intravoxel_dephasing_ramp
                            intra_b0 = torch.sum(intra_b0,1)
                            
                            IVP_nspins_cos = torch.cos(intra_b0)
                            IVP_nspins_sin = torch.sin(intra_b0)
                             
                            self.IVP[:,0,0,0,0] = IVP_nspins_cos
                            self.IVP[:,0,0,0,1] = -IVP_nspins_sin
                            self.IVP[:,0,0,1,0] = IVP_nspins_sin
                            self.IVP[:,0,0,1,1] = IVP_nspins_cos                
                            
                            # do intravoxel precession
                            spins_cut = GradIntravoxelPrecessClass.apply(self.IVP,spins_cut,self)
                            
                            # reset readout position tracking vars
                            start_t = t + 1
                            total_delay = delay
                        else:                                       # keep accumulating
                            total_delay += delay
                        
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox)).byte()
        self.tmask = self.setdevice(self.tmask)
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.tmask = self.tmask[PD0_mask]
        self.lastM = spins_cut.clone()
                


    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,r,spins):
        s = self.signal[:,:,r,:,:]
         
        # for now we ignore parallel imaging options here (do naive sum sig over coil)
        s = torch.sum(s, 0)                                                  
        r = torch.matmul(self.G_adj.permute([1,0,2,3]), s)
        self.reco = self.reco + torch.sum(r[:,:,:2,0],1)
        
    def adjoint(self, spins):
        self.init_reco()
        
        #adc_mask = self.adc_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #self.signal *= adc_mask        
        
        for r in range(self.NRep):
            self.set_grad_adj_op(r)
            self.do_grad_adj_reco(r,spins)
        
       

# Fast, but memory inefficient version (also for now does not support parallel imagigng)
class Scanner_fast(Scanner):
    
    def init_gradient_tensor_holder(self):
        G = torch.zeros((self.T,self.NRep,self.NVox,3,3), dtype=torch.float32)
        G[:,:,:,2,2] = 1
         
        G_adj = torch.zeros((self.T,self.NRep,self.NVox,3,3), dtype=torch.float32)
        G_adj[:,:,:,2,2] = 1
        
        self.G = self.setdevice(G)
        self.G_adj = self.setdevice(G_adj)
        
        # intravoxel precession
        IVP = torch.zeros((self.NSpins,1,1,3,3), dtype=torch.float32)
        IVP = self.setdevice(IVP)
        
        IVP[:,0,0,2,2] = 1
        
        self.IVP = IVP
        
    def set_gradient_precession_tensor(self,grad_moms,refocusing=False,wrap_k=False,epi=False):
        
        # we need to shift grad_moms to the right for adjoint pass, since at each repetition we have:
        # meas-signal, flip, relax,grad order  (signal comes before grads!)
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
        temp = temp[:-1,:,:]
        k=torch.cumsum(temp,0)
        
        B0X = torch.unsqueeze(grad_moms[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        B0_grad_cos = torch.cos(B0_grad)
        B0_grad_sin = torch.sin(B0_grad)
        
        if wrap_k:
            hx = self.sz[0]/2
            k[:,:,0] = torch.fmod(k[:,:,0]+hx,hx*2)-hx
        
        # for backward pass
        if refocusing:
            #B0X = torch.unsqueeze(k[:,:,0]-self.sz[0]/2,2) * self.rampX
            
            refocusing_pulse_action_idx = 1
            kloc = 0
            for r in range(self.NRep):
                for t in range(self.T):
                    if refocusing_pulse_action_idx+1 == t:     # +1 because of right gradmom shift (see above )
                        kloc = -kloc
                    kloc += temp[t,r,:]
                    k[t,r,:] = kloc
                
        if epi:
            kloc = 0
            for r in range(self.NRep):
                for t in range(self.T):
                    kloc += temp[t,r,:]
                    k[t,r,:] = kloc            
                
        B0X = torch.unsqueeze(k[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(k[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        B0_grad_adj_cos = torch.cos(B0_grad)
        B0_grad_adj_sin = torch.sin(B0_grad)        
        
        self.G[:,:,:,0,0] = B0_grad_cos
        self.G[:,:,:,0,1] = -B0_grad_sin
        self.G[:,:,:,1,0] = B0_grad_sin
        self.G[:,:,:,1,1] = B0_grad_cos    
        
        # adjoint
        self.G_adj[:,:,:,0,0] = B0_grad_adj_cos
        self.G_adj[:,:,:,0,1] = B0_grad_adj_sin
        self.G_adj[:,:,:,1,0] = -B0_grad_adj_sin
        self.G_adj[:,:,:,1,1] = B0_grad_adj_cos
        
        # save grad_moms for intravoxel precession op
        self.grad_moms_for_intravoxel_precession = grad_moms
        
    # do full flip/gradprecess adjoint integrating over all repetition grad/flip history
    def set_gradient_precession_tensor_adjhistory(self,grad_moms):
        
        B0X = torch.unsqueeze(grad_moms[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        B0_grad_cos = torch.cos(B0_grad)
        B0_grad_sin = torch.sin(B0_grad)
        
        self.G[:,:,:,0,0] = B0_grad_cos
        self.G[:,:,:,0,1] = -B0_grad_sin
        self.G[:,:,:,1,0] = B0_grad_sin
        self.G[:,:,:,1,1] = B0_grad_cos

        self.G_adj[:,:,:,2,2] = 1
        self.G_adj[0,0,:,0,0] = 1
        self.G_adj[0,0,:,1,1] = 1
        
        propagator = self.G_adj[0,0,:,:,:]
        
        for r in range(self.NRep):
            for t in range(self.T):
                f = self.F[t,r,:,:,:]                                    # flip
                    
                if t == 0 and r == 0:
                    pass
                else:
                    propagator = torch.matmul(f, propagator)

                propagator = torch.matmul(self.G[t,r,:,:,:],propagator) # grads
                
                self.G_adj[t,r,:,:,:] = propagator
                
        self.G_adj = self.G_adj.permute([0,1,2,4,3])
                
    def grad_precess(self,t,r,spins):
        spins.M = torch.matmul(self.G[t,r,:,:,:],spins.M)
        
    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,t,spins):
        #adc_mask = self.adc_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #s = self.signal * adc_mask
        s = torch.sum(self.signal,0)
        
        r = torch.matmul(self.G_adj.permute([2,3,0,1,4]).contiguous().view([self.NVox,3,self.T*self.NRep*3]), s.view([1,self.T*self.NRep*3,1]))
        self.reco = r[:,:2,0]
        
    # run throw all repetition/actions and yield signal
    def forward(self,spins,event_time,do_dummy_scans=False):
        super().forward(spins,event_time,do_dummy_scans,compact_grad_tensor=False)
        
    def forward_mem(self,spins,event_time,do_dummy_scans=False):
        super().forward_mem(spins,event_time,do_dummy_scans,compact_grad_tensor=False)
        
    def forward_sparse(self,spins,event_time,do_dummy_scans=False):
        super().forward_sparse(spins,event_time,do_dummy_scans,compact_grad_tensor=False)
        
    def forward_fast(self,spins,event_time,do_dummy_scans=False):
        super().forward_fast(spins,event_time,do_dummy_scans,compact_grad_tensor=False)
        
    def forward_sparse_fast(self,spins,event_time,do_dummy_scans=False):
        super().forward_sparse_fast(spins,event_time,do_dummy_scans,compact_grad_tensor=False)   
        
    def do_dummy_scans(self,spins,event_time,nrep=0):
        super().do_dummy_scans(spins,event_time,compact_grad_tensor=False,nrep=0)

    def do_dummy_scans_sparse(self,spins,event_time,nrep=0):
        super().do_dummy_scans_sparse(spins,event_time,compact_grad_tensor=False,nrep=0)
        
    # compute adjoint encoding op-based reco    <            
    def adjoint(self,spins):
        self.init_reco()
        
        #adc_mask = self.adc_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #s = self.signal * adc_mask
        s = torch.sum(self.signal,0)
        
        r = torch.matmul(self.G_adj.permute([2,3,0,1,4]).contiguous().view([self.NVox,3,self.T*self.NRep*3]), s.view([1,self.T*self.NRep*3,1]))
        self.reco = r[:,:2,0]        

# AUX classes


# memory-efficient backpropagation-util classes
class FlipClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, scanner):
        ctx.f = f.clone()
        ctx.scanner = scanner
        
        return torch.matmul(f,x)

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,2,1]),grad_output)
        
        ctx.scanner.lastM = torch.matmul(ctx.f.permute([0,2,1]),ctx.scanner.lastM)
        gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
        gf = torch.sum(gf,[0,2])
        
        return (gf, gx, None) 
  
class RelaxClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, delay, t, scanner, spins):
        ctx.f = f.clone()
        ctx.delay = delay
        ctx.scanner = scanner
        ctx.spins = spins
        ctx.t = t
        ctx.thresh = 1e-2
        
        if ctx.delay > ctx.thresh or np.mod(ctx.t,ctx.scanner.T) == 0:
            ctx.M = x.clone()
            
        out = torch.matmul(f,x)
        out[:,:,:,2,0] += (1 - f[:,:,2,2]).view([1,1,scanner.NVox]) * spins.MZ0
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,1,3,2]),grad_output)
      
        gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
        gf = torch.sum(gf,[0])
        
        if ctx.delay > ctx.thresh or np.mod(ctx.t,ctx.scanner.T) == 0:
            ctx.scanner.lastM = ctx.M
        else:
            d1 = ctx.f[0,:,0,0]
            id1 = 1/d1
            
            d3 = ctx.f[0,:,2,2]
            id3 = 1/d3
            id3 = id3.view([1,ctx.scanner.NVox])
            
            ctx.scanner.lastM[:,0,:,:2,0] *= id1.view([1,ctx.scanner.NVox,1])
            ctx.scanner.lastM[:,0,:,2,0] = ctx.scanner.lastM[:,0,:,2,0]*id3 + (1-id3)*ctx.spins.MZ0[:,0,:]
            
            ctx.scanner.lastM[:,:,ctx.scanner.tmask,:] = 0
            
        return (gf, gx, None, None, None, None)  
    
class RelaxSparseClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, delay, t, scanner, spins, mask):
        ctx.f = f.clone()
        ctx.delay = delay
        ctx.scanner = scanner
        ctx.spins = spins
        ctx.t = t
        ctx.thresh = 1e-2
        ctx.mask = mask
        
        if ctx.delay > ctx.thresh or np.mod(ctx.t,ctx.scanner.T) == 0:
            ctx.M = x.clone()
            
        out = torch.matmul(f,x)
        out[:,:,:,2,0] += (1 - f[:,:,2,2]).view([1,1,x.shape[2]]) * spins.MZ0[:,:,mask]
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,1,3,2]),grad_output)
      
        gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
        gf = torch.sum(gf,[0])
        
        if ctx.delay > ctx.thresh or np.mod(ctx.t,ctx.scanner.T) == 0:
            ctx.scanner.lastM = ctx.M
        else:
            d1 = ctx.f[0,:,0,0]
            id1 = 1/d1
            
            d3 = ctx.f[0,:,2,2]
            id3 = 1/d3
            id3 = id3.view([1,grad_output.shape[2]])
            
            ctx.scanner.lastM[:,0,:,:2,0] *= id1.view([1,grad_output.shape[2],1])
            ctx.scanner.lastM[:,0,:,2,0] = ctx.scanner.lastM[:,0,:,2,0]*id3 + (1-id3)*ctx.spins.MZ0[:,0,ctx.mask]
            
            ctx.scanner.lastM[:,:,ctx.scanner.tmask,:] = 0
            
        return (gf, gx, None, None, None, None, None)
  
class DephaseClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, scanner):
        ctx.f = f.clone()
        ctx.scanner = scanner
        
        return torch.matmul(f,x)

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,1,2,4,3]),grad_output)
        
        ctx.scanner.lastM = torch.matmul(ctx.f.permute([0,1,2,4,3]),ctx.scanner.lastM)
        
        gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
        gf = torch.sum(gf,[2],keepdim=True)
        
        return (gf, gx, None) 
  
class GradPrecessClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, scanner):
        ctx.f = f.clone()
        ctx.scanner = scanner
        
        return torch.matmul(f,x)

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,2,1]),grad_output)
        
        ctx.scanner.lastM = torch.matmul(ctx.f.permute([0,2,1]),ctx.scanner.lastM)
        gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
        gf = torch.sum(gf,[0,1])
        
        return (gf, gx, None) 
  
class GradIntravoxelPrecessClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, scanner):
        ctx.f = f.clone()
        ctx.scanner = scanner
        
        return torch.matmul(f,x)

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,1,2,4,3]),grad_output)
        
        ctx.scanner.lastM = torch.matmul(ctx.f.permute([0,1,2,4,3]),ctx.scanner.lastM)
        gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
        gf = torch.sum(gf,[1,2],keepdim=True)
        
        return (gf, gx, None)
    
class B0InhomoClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, scanner):
        ctx.f = f.clone()
        ctx.scanner = scanner
        
        return torch.matmul(f,x)

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,1,2,4,3]),grad_output)
        
        ctx.scanner.lastM = torch.matmul(ctx.f.permute([0,1,2,4,3]),ctx.scanner.lastM)
        
        gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
        gf = torch.sum(gf,[0],keepdim=True)
        
        return (gf, gx, None)     