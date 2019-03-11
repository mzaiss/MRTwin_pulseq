import numpy as np
import torch


torch.cuda.get_device_properties(0)

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
        self.F = None                              # flip tensor (T,NRep,4,4)
        self.R = None                          # relaxation tensor (NVox,4,4)
        self.P = None                   # free precession tensor (NSpins,4,4)
        self.G = None            # gradient precession tensor (NRep,NVox,4,4)
        self.G_adj = None         # adjoint gradient operator (NRep,NVox,4,4)

        self.B0_grad_cos = None  # accum phase due to gradients (T,NRep,NVox)
        self.B0_grad_sin = None
        self.B0_grad_adj_cos = None  # adjoint grad phase accum (T,NRep,NVox)
        self.B0_grad_adj_sin = None
        
        self.B1 = None          # coil sensitivity profiles (NCoils,NVox,2,2)

        self.signal = None                # measured signal (NCoils,T,NRep,4)
        self.reco =  None                       # reconstructed image (NVox,) 
        
        self.ROI_signal = None                # measured signal (NCoils,T,NRep,4)
        self.ROI_def = 1
        
        self.use_gpu =  use_gpu
        
    # device setter
    def setdevice(self,x):
        if self.use_gpu:
            x = x.cuda(0)
            
        return x        
        
    def set_adc_mask(self):
        adc_mask = torch.from_numpy(np.ones((self.T,1))).float()
        #adc_mask[:self.T-self.sz[0]] = 0
        
        self.adc_mask = self.setdevice(adc_mask)

    def get_ramps(self):
        
        use_nonlinear_grads = False                       # very experimental
        
        baserampX = np.linspace(-1,1,self.sz[0] + 1)
        baserampY = np.linspace(-1,1,self.sz[1] + 1)
        
        if use_nonlinear_grads:
            baserampX = np.abs(baserampX)**1.2 * np.sign(baserampX)
            baserampY = np.abs(baserampY)**1.2 * np.sign(baserampY)
        
        rampX = np.pi*baserampX
        rampX = -np.expand_dims(rampX[:-1],1)
        rampX = np.tile(rampX, (1, self.sz[1]))
        
        rampX = torch.from_numpy(rampX).float()
        rampX = rampX.view([1,1,self.NVox])    
        
        # set gradient spatial forms
        rampY = np.pi*baserampY
        rampY = -np.expand_dims(rampY[:-1],0)
        rampY = np.tile(rampY, (self.sz[0], 1))
        
        rampY = torch.from_numpy(rampY).float()
        rampY = rampY.view([1,1,self.NVox])    
        
        # 1D case
        if self.sz[1] == 1:
            rampY[:,:,:] = 0
        
        self.rampX = self.setdevice(rampX)
        self.rampY = self.setdevice(rampY)
        
    def init_coil_sensitivities(self):
        # handle complex mul as matrix mul
        B1 = torch.zeros((self.NCoils,1,self.NVox,2,2), dtype=torch.float32)
        B1[:,0,:,0,0] = 1
        B1[:,0,:,1,1] = 1
        
        self.B1 = self.setdevice(B1)
        
    def init_flip_tensor_holder(self):
        F = torch.zeros((self.T,self.NRep,1,4,4), dtype=torch.float32)
        
        F[:,:,0,3,3] = 1
        F[:,:,0,1,1] = 1
         
        self.F = self.setdevice(F)
         
    def set_flip_tensor(self,flips):
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        self.F[:,:,0,0,0] = flips_cos
        self.F[:,:,0,0,2] = flips_sin
        self.F[:,:,0,2,0] = -flips_sin
        self.F[:,:,0,2,2] = flips_cos 
        
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
        self.F[:,:,0,3,3] = 1
        
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
        self.F[:,:,0,3,3] = 1
        

    def set_relaxation_tensor(self,spins,dt):
        R = torch.zeros((self.NVox,4,4), dtype=torch.float32) 
        
        R = self.setdevice(R)
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,3,3] = 1
        
        R[:,0,0] = T2_r
        R[:,1,1] = T2_r
        R[:,2,2] = T1_r
        R[:,2,3] = 1 - T1_r
         
        R = R.view([1,self.NVox,4,4])
        
        self.R = R
        
    def set_freeprecession_tensor(self,spins,dt):
        P = torch.zeros((self.NSpins,1,1,4,4), dtype=torch.float32)
        
        P = self.setdevice(P)
        
        B0_nspins = spins.omega[:,0].view([self.NSpins])
        
        B0_nspins_cos = torch.cos(B0_nspins*dt)
        B0_nspins_sin = torch.sin(B0_nspins*dt)
         
        P[:,0,0,0,0] = B0_nspins_cos
        P[:,0,0,0,1] = -B0_nspins_sin
        P[:,0,0,1,0] = B0_nspins_sin
        P[:,0,0,1,1] = B0_nspins_cos
         
        P[:,0,0,2,2] = 1
        P[:,0,0,3,3] = 1         
         
        self.P = P
         
    
    def init_gradient_tensor_holder(self):
        G = torch.zeros((self.NRep,self.NVox,4,4), dtype=torch.float32)
        G[:,:,2,2] = 1
        G[:,:,3,3] = 1
         
        G_adj = torch.zeros((self.NRep,self.NVox,4,4), dtype=torch.float32)
        G_adj[:,:,2,2] = 1
        G_adj[:,:,3,3] = 1
         
        self.G = self.setdevice(G)
        self.G_adj = self.setdevice(G_adj)
        
    def set_grad_op(self,t):
        
        self.G[:,:,0,0] = self.B0_grad_cos[t,:,:]
        self.G[:,:,0,1] = -self.B0_grad_sin[t,:,:]
        self.G[:,:,1,0] = self.B0_grad_sin[t,:,:]
        self.G[:,:,1,1] = self.B0_grad_cos[t,:,:]
        
    def set_grad_adj_op(self,t):
        
        self.G_adj[:,:,0,0] = self.B0_grad_adj_cos[t,:,:]
        self.G_adj[:,:,0,1] = self.B0_grad_adj_sin[t,:,:]
        self.G_adj[:,:,1,0] = -self.B0_grad_adj_sin[t,:,:]
        self.G_adj[:,:,1,1] = self.B0_grad_adj_cos[t,:,:]        
        
    def set_gradient_precession_tensor(self,grad_moms):
        
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
        #grads = temp[1:,:,:] - temp[:-1,:,:]      
        grads=grad_moms
        grad_moms=torch.cumsum(grads,0)
        
        B0X = torch.unsqueeze(grads[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grads[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_cos = torch.cos(B0_grad)
        self.B0_grad_sin = torch.sin(B0_grad)
        
        # for backward pass
        B0X = torch.unsqueeze(grad_moms[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_adj_cos = torch.cos(B0_grad)
        self.B0_grad_adj_sin = torch.sin(B0_grad)
        
    def flip(self,t,r,spins):
        spins.M = torch.matmul(self.F[t,r,:,:,:],spins.M)
        
    # apply flip at all repetition simultanetously (non TR transfer case)
    def flip_allRep(self,t,spins):
        spins.M = torch.matmul(self.F[t,:,:,:,:],spins.M)
        
    def relax(self,spins):
        spins.M = torch.matmul(self.R,spins.M)
        
    def relax_and_dephase(self,spins):
        
        spins.M = torch.matmul(self.R,spins.M)
        spins.M = torch.matmul(self.P,spins.M)
        
    def grad_precess(self,r,spins):
        spins.M = torch.matmul(self.G[r,:,:,:],spins.M)        
        
    def grad_precess_allRep(self,spins):
        spins.M = torch.matmul(self.G,spins.M)
        
    def init_signal(self):
        signal = torch.zeros((self.NCoils,self.T,self.NRep,4,1), dtype=torch.float32) 
        #signal[:,:,:,2:,0] = 1                                 # aux dim zero ()
              
        self.signal = self.setdevice(signal)
        
        self.ROI_signal = torch.zeros((self.T+1,self.NRep,6), dtype=torch.float32) # for trans magnetization
        self.ROI_signal = self.setdevice(self.ROI_signal)
        self.ROI_def= int((self.sz[0]/2)*self.sz[1]+ self.sz[1]/2)
        
    def init_reco(self):
        reco = torch.zeros((self.NVox,2), dtype = torch.float32)
        
        self.reco = self.setdevice(reco)
        
    def read_signal(self,t,r,spins):
        if self.adc_mask[t] > 0:
            sig = torch.sum(spins.M[:,0,:,:2,0],[0])
            sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(0).unsqueeze(4))
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(sig.shape).float()
                #noise[:,2:] = 0
                noise = self.setdevice(noise)
                sig += noise  
            
            self.signal[:,t,r,:2] = (torch.sum(sig,[2]) * self.adc_mask[t])
      
        
    def read_signal_allRep(self,t,spins):
        
        if self.adc_mask[t] > 0:
            #import pdb; pdb.set_trace()
            sig = torch.sum(spins.M[:,:,:,:2,0],[0])
            sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(4))
            
            self.signal[:,t,:,:2] = torch.sum(sig,[2]) * self.adc_mask[t]
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.signal[:,t,:,:,0].shape).float()
                noise[:,:,2:] = 0
                noise = self.setdevice(noise)
                self.signal[:,t,:,:,0] = self.signal[:,t,:,:,0] + noise

    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,t,spins):
        
        # redundant
        s = self.signal[:,t,:,:,:] * self.adc_mask[t]
        #s = self.signal[:,t,:,:,:]
         
        # for now we ignore parallel imaging options here (do naive sum sig over coil)
        s = torch.sum(s, 0)                                                  
        r = torch.matmul(self.G_adj.permute([1,0,2,3]), s)
        self.reco = self.reco + torch.sum(r[:,:,:2,0],1)
        
    ## extra func land        
    # aux flexible operators for sandboxing things
    def custom_flip(self,t,spins,flips):
        
        F = torch.zeros((self.T,self.NRep,1,4,4), dtype=torch.float32)
        
        F[:,:,0,3,3] = 1
        F[:,:,0,1,1] = 1
         
        F = self.setdevice(F)
        
        flips = self.setdevice(flips)
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        F[:,:,0,0,0] = flips_cos
        F[:,:,0,0,2] = flips_sin
        F[:,:,0,2,0] = -flips_sin
        F[:,:,0,2,2] = flips_cos         
        
        spins.M = torch.matmul(F[t,:,:,:],spins.M)
        
    def custom_relax(self,spins,dt=None):
        
        R = torch.zeros((self.NVox,4,4), dtype=torch.float32) 
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,3,3] = 1
        
        R[:,0,0] = T2_r
        R[:,1,1] = T2_r
        R[:,2,2] = T1_r
        R[:,2,3] = 1 - T1_r
         
        R = R.view([1,self.NVox,4,4])
        
        R = self.setdevice(R)
        
        spins.M = torch.matmul(R,spins.M)  
        
        
# variation for supervised learning
# TODO fix relax tensor -- is batch dependent
# TODO implement as child class of Scanner class, override methods
class Scanner_batched():
    
    def __init__(self,sz,NVox,NSpins,NRep,T,NCoils,noise_std,batch_size,use_gpu):
        
        self.sz = sz                                             # image size
        self.NVox = sz[0]*sz[1]                                 # voxel count
        self.NSpins = NSpins              # number of spin sims in each voxel
        self.NRep = NRep                              # number of repetitions
        self.T = T                       # number of "actions" with a readout
        self.NCoils = NCoils                # number of receive coil elements
        self.noise_std = noise_std              # additive Gaussian noise std
        
        self.adc_mask = None         # ADC signal acquisition event mask (T,)
        self.rampX = None        # spatial encoding linear gradient ramp (sz)
        self.rampY = None
        self.F = None                              # flip tensor (T,NRep,4,4)
        self.R = None                          # relaxation tensor (NVox,4,4)
        self.P = None                   # free precession tensor (NSpins,4,4)
        self.G = None            # gradient precession tensor (NRep,NVox,4,4)
        self.G_adj = None         # adjoint gradient operator (NRep,NVox,4,4)

        self.B0_grad_cos = None  # accum phase due to gradients (T,NRep,NVox)
        self.B0_grad_sin = None
        self.B0_grad_adj_cos = None  # adjoint grad phase accum (T,NRep,NVox)
        self.B0_grad_adj_sin = None
        
        self.B1 = None          # coil sensitivity profiles (NCoils,NVox,2,2)

        self.signal = None                # measured signal (NCoils,T,NRep,4)
        self.reco =  None                       # reconstructed image (NVox,) 
        
        self.batch_size = batch_size
        self.use_gpu =  use_gpu
        
    # device setter
    def setdevice(self,x):
        if self.use_gpu:
            x = x.cuda(0)
            
        return x        
        
    def set_adc_mask(self):
        adc_mask = torch.from_numpy(np.ones((self.T,1))).float()
        #adc_mask[:self.T-self.sz[0]] = 0
        
        self.adc_mask = self.setdevice(adc_mask)

    def get_ramps(self):
        
        use_nonlinear_grads = False                       # very experimental
        
        baserampX = np.linspace(-1,1,self.sz[0] + 1)
        baserampY = np.linspace(-1,1,self.sz[1] + 1)
        
        if use_nonlinear_grads:
            baserampX = np.abs(baserampX)**1.2 * np.sign(baserampX)
            baserampY = np.abs(baserampY)**1.2 * np.sign(baserampY)
        
        rampX = np.pi*baserampX
        rampX = -np.expand_dims(rampX[:-1],1)
        rampX = np.tile(rampX, (1, self.sz[1]))
        
        rampX = torch.from_numpy(rampX).float()
        rampX = rampX.view([1,1,self.NVox])    
        
        # set gradient spatial forms
        rampY = np.pi*baserampY
        rampY = -np.expand_dims(rampY[:-1],0)
        rampY = np.tile(rampY, (self.sz[0], 1))
        
        rampY = torch.from_numpy(rampY).float()
        rampY = rampY.view([1,1,self.NVox])    
        
        self.rampX = self.setdevice(rampX)
        self.rampY = self.setdevice(rampY)
        
    def init_coil_sensitivities(self):
        # handle complex mul as matrix mul
        B1 = torch.zeros((self.NCoils,1,self.NVox,2,2), dtype=torch.float32)
        B1[:,0,:,0,0] = 1
        B1[:,0,:,1,1] = 1
        
        self.B1 = self.setdevice(B1)
        
    def init_flip_tensor_holder(self):
        F = torch.zeros((self.T,self.NRep,1,4,4), dtype=torch.float32)
        
        F[:,:,0,3,3] = 1
        F[:,:,0,1,1] = 1
         
        F = F.unsqueeze(0)  # XXXX
         
        self.F = self.setdevice(F)
         
    def set_flip_tensor(self,flips):
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        # XXXX
        self.F[0,:,:,0,0,0] = flips_cos
        self.F[0,:,:,0,0,2] = flips_sin
        self.F[0,:,:,0,2,0] = -flips_sin
        self.F[0,:,:,0,2,2] = flips_cos 
        
    def set_flipAxisAngle_tensor(self,flips):
        
        # ... greatly simplifies if assume rotations in XY plane ...
        theta = torch.norm(flips,dim=2).unsqueeze(2)
        v = flips / theta
        theta = theta.unsqueeze(2).unsqueeze(2)
        
        self.F[0,:,:,0,0,0] = 0
        self.F[0,:,:,0,0,1] = -v[:,:,2]
        self.F[0,:,:,0,0,2] = v[:,:,1]
        self.F[0,:,:,0,1,0] = v[:,:,2]
        self.F[0,:,:,0,1,1] = 0
        self.F[0,:,:,0,1,2] = -v[:,:,0]
        self.F[0,:,:,0,2,0] = -v[:,:,1]
        self.F[0,:,:,0,2,1] = v[:,:,0]
        self.F[0,:,:,0,2,2] = 0
        
        # matrix square
        F2 = torch.matmul(self.F,self.F)
        self.F = torch.sin(theta) * self.F + (1 - torch.cos(theta))*F2
        
        self.F[0,:,:,0,0,0] += 1
        self.F[0,:,:,0,1,1] += 1
        self.F[0,:,:,0,2,2] += 1  
        self.F[0,:,:,0,3,3] = 1
        
    def set_flipXY_tensor(self,input_flips):
        
        vx = torch.cos(input_flips[:,:,1])
        vy = torch.sin(input_flips[:,:,1])
        
        theta = input_flips[:,:,0]
            
        theta = theta.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        
        self.F[0,:,:,0,0,0] = 0
        self.F[0,:,:,0,0,1] = 0
        self.F[0,:,:,0,0,2] = vy
        self.F[0,:,:,0,1,0] = 0
        self.F[0,:,:,0,1,1] = 0
        self.F[0,:,:,0,1,2] = -vx
        self.F[0,:,:,0,2,0] = -vy
        self.F[0,:,:,0,2,1] = vx
        self.F[0,:,:,0,2,2] = 0

        # matrix square
        F2 = torch.matmul(self.F,self.F)
        self.F = torch.sin(theta) * self.F + (1 - torch.cos(theta))*F2

        self.F[0,:,:,0,0,0] += 1
        self.F[0,:,:,0,1,1] += 1
        self.F[0,:,:,0,2,2] += 1
        self.F[0,:,:,0,3,3] = 1     
         
    def set_relaxation_tensor(self,spins,dt):
        R = torch.zeros((self.batch_size,self.NVox,4,4), dtype=torch.float32) 
        
        R = self.setdevice(R)
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,:,3,3] = 1
        
        R[:,:,0,0] = T2_r
        R[:,:,1,1] = T2_r
        R[:,:,2,2] = T1_r
        R[:,:,2,3] = 1 - T1_r
         
        R = R.view([self.batch_size,1,1,self.NVox,4,4])  # XXXX
        
        self.R = R
        
    def set_freeprecession_tensor(self,spins,dt):
        P = torch.zeros((self.NSpins,1,1,4,4), dtype=torch.float32)
        
        P = self.setdevice(P)
        
        B0_nspins = spins.omega.view([self.NSpins])
        
        B0_nspins_cos = torch.cos(B0_nspins*dt)
        B0_nspins_sin = torch.sin(B0_nspins*dt)
         
        P[:,0,0,0,0] = B0_nspins_cos
        P[:,0,0,0,1] = -B0_nspins_sin
        P[:,0,0,1,0] = B0_nspins_sin
        P[:,0,0,1,1] = B0_nspins_cos
         
        P[:,0,0,2,2] = 1
        P[:,0,0,3,3] = 1           
         
        P = P.unsqueeze(0)  # XXXX
         
        self.P = P
         
    
    def init_gradient_tensor_holder(self):
        G = torch.zeros((self.NRep,self.NVox,4,4), dtype=torch.float32)
        G[:,:,2,2] = 1
        G[:,:,3,3] = 1
         
        G_adj = torch.zeros((self.NRep,self.NVox,4,4), dtype=torch.float32)
        G_adj[:,:,2,2] = 1
        G_adj[:,:,3,3] = 1
             
        G = G.unsqueeze(0)  # XXXX
        G_adj = G_adj.unsqueeze(0)  # XXXX
         
        self.G = self.setdevice(G)
        self.G_adj = self.setdevice(G_adj)
        
    def set_grad_op(self,t):
        
        # XXXX
        self.G[0,:,:,0,0] = self.B0_grad_cos[t,:,:]
        self.G[0,:,:,0,1] = -self.B0_grad_sin[t,:,:]
        self.G[0,:,:,1,0] = self.B0_grad_sin[t,:,:]
        self.G[0,:,:,1,1] = self.B0_grad_cos[t,:,:]
        
    def set_grad_adj_op(self,t):
        
        # XXXX
        self.G_adj[0,:,:,0,0] = self.B0_grad_adj_cos[t,:,:]
        self.G_adj[0,:,:,0,1] = self.B0_grad_adj_sin[t,:,:]
        self.G_adj[0,:,:,1,0] = -self.B0_grad_adj_sin[t,:,:]
        self.G_adj[0,:,:,1,1] = self.B0_grad_adj_cos[t,:,:]        
        
    def set_gradient_precession_tensor(self,grad_moms):
        
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
        grads = temp[1:,:,:] - temp[:-1,:,:]        
        
        B0X = torch.unsqueeze(grads[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grads[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_cos = torch.cos(B0_grad)
        self.B0_grad_sin = torch.sin(B0_grad)
        
        # for backward pass
        B0X = torch.unsqueeze(grad_moms[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_adj_cos = torch.cos(B0_grad)
        self.B0_grad_adj_sin = torch.sin(B0_grad)
        
    def flip(self,t,r,spins):
        spins.M = torch.matmul(self.F[0,t,r,:,:,:],spins.M)
        
    # apply flip at all repetition simultanetously (non TR transfer case)
    def flip_allRep(self,t,spins):
        spins.M = torch.matmul(self.F[0,t,:,:,:,:],spins.M)
        
    def relax(self,spins):
        spins.M = torch.matmul(self.R,spins.M)
        
    def relax_and_dephase(self,spins):
        
        spins.M = torch.matmul(self.R,spins.M)
        spins.M = torch.matmul(self.P,spins.M)
        
    def grad_precess(self,r,spins):
        spins.M = torch.matmul(self.G[0,r,:,:,:],spins.M)        
        
    def grad_precess_allRep(self,spins):
        spins.M = torch.matmul(self.G,spins.M)
        
    def init_signal(self):
        signal = torch.zeros((self.batch_size,self.NCoils,self.T,self.NRep,4,1), dtype=torch.float32) 
        #signal[:,:,:,:,2:,0] = 1                                 # aux dim zero
              
        self.signal = self.setdevice(signal)
        
    def init_reco(self):
        reco = torch.zeros((self.batch_size,self.NVox,2), dtype = torch.float32)
        
        self.reco = self.setdevice(reco)
        
    # XXX
    # TODO: noise treatment incosistent with Scanner class protocol
    def read_signal(self,t,r,spins):
        if self.adc_mask[t] > 0:
            
            sig = torch.sum(spins.M[:,:,0,:,:2,0],[1])
            sig = torch.matmul(self.B1.unsqueeze(0),sig.unsqueeze(1).unsqueeze(1).unsqueeze(5))
            
            self.signal[:,:,t,r,:2] = (torch.sum(sig,[3]) * self.adc_mask[t])
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.signal[:,:,t,r,:,0].shape).float()
                noise[:,:,2:] = 0
                noise = self.setdevice(noise)
                self.signal[:,:,t,r,:,0] = self.signal[:,:,t,r,:,0] + noise        
        
    # XXX  -- slow
    def read_signal_allRep(self,t,spins):
        
        if self.adc_mask[t] > 0:
            
            sig = torch.matmul(self.B1.unsqueeze(0),spins.M[:,:,:,:,:2,:])
            self.signal[:,0,t,:,:2] = torch.sum(sig,[1,3]) * self.adc_mask[t]
            
            # slow
#            sig = torch.sum(spins.M[:,:,:,:,:2,0],[1])
#            sig = torch.matmul(self.B1.unsqueeze(0),sig.unsqueeze(1).unsqueeze(5))
#            self.signal[:,:,t,:,:2] = torch.sum(sig,[3]) * self.adc_mask[t]            
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.signal[:,:,t,:,:,0].shape).float()
                noise[:,:,:,2:] = 0
                noise = self.setdevice(noise)
                self.signal[:,:,t,:,:,0] = self.signal[:,:,t,:,:,0] + noise

    # XXX
    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,t,spins):
        s = self.signal[:,:,t,:,:,:] * self.adc_mask[t]
        # for now we ignore parallel imaging options here (do naive sum sig over coil)
        s = torch.sum(s, 1).unsqueeze(1)

        r = torch.matmul(self.G_adj.permute([0,2,1,3,4]), s)
        self.reco = self.reco + torch.sum(r[:,:,:,:2,0],2)
        
      
      
      
      
# Fast, but memory inefficient version (also for now does not support parallel imagigng)
class Scanner_fast(Scanner):
    
    def init_gradient_tensor_holder(self):
        G = torch.zeros((self.T,self.NRep,self.NVox,4,4), dtype=torch.float32)
        G[:,:,:,2,2] = 1
        G[:,:,:,3,3] = 1
         
        G_adj = torch.zeros((self.T,self.NRep,self.NVox,4,4), dtype=torch.float32)
        G_adj[:,:,:,2,2] = 1
        G_adj[:,:,:,3,3] = 1
         
        self.G = self.setdevice(G)
        self.G_adj = self.setdevice(G_adj)
        
    def set_gradient_precession_tensor(self,grad_moms):
        
        k=torch.cumsum(grad_moms,0)
        
        B0X = torch.unsqueeze(grad_moms[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        B0_grad_cos = torch.cos(B0_grad)
        B0_grad_sin = torch.sin(B0_grad)
        
        # for backward pass
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
        
    def grad_precess(self,t,r,spins):
        spins.M = torch.matmul(self.G[t,r,:,:,:],spins.M)        
        
    # TODO: fix
    def grad_precess_allRep(self,spins):
        
        class ExecutionControl(Exception): pass
        raise ExecutionControl('grad_precess_allRep: WIP not implemented')
        
        spins.M = torch.matmul(self.G,spins.M)
        
    def read_signal(self,t,r,spins):
        if self.adc_mask[t] > 0:
            
            # parallel imaging disabled for now
            #sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(0).unsqueeze(4))
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(spins.M[:,:,:,:2].shape).float()
                noise = self.setdevice(noise)
                sig = spins.M[:,:,:,:2]
                sig += noise  
            
                self.signal[0,t,r,:2] = ((torch.sum(sig,[0,1,2]) * self.adc_mask[t]))
            else:
                self.signal[0,t,r,:2] = ((torch.sum(spins.M[:,:,:,:2],[0,1,2]) * self.adc_mask[t]))
     
        
    # TODO: fix
    def read_signal_allRep(self,t,spins):
        
        class ExecutionControl(Exception): pass
        raise ExecutionControl('read_signal_allRep: WIP not implemented')
        
        if self.adc_mask[t] > 0:
            #import pdb; pdb.set_trace()
            sig = torch.sum(spins.M[:,:,:,:2,0],[0])
            sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(4))
            
            self.signal[:,t,:,:2] = torch.sum(sig,[2]) * self.adc_mask[t]
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.signal[:,t,:,:,0].shape).float()
                noise[:,:,2:] = 0
                noise = self.setdevice(noise)
                self.signal[:,t,:,:,0] = self.signal[:,t,:,:,0] + noise

    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,t,spins):
        
        adc_mask = self.adc_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        s = self.signal * adc_mask
        s = torch.sum(s,0)
        
        r = torch.matmul(self.G_adj.permute([2,3,0,1,4]).contiguous().view([self.NVox,4,self.T*self.NRep*4]), s.view([1,self.T*self.NRep*4,1]))
        self.reco = r[:,:2,0]
        
    # run throw all repetition/actions and yield signal
    def forward(self,spins,event_time):
        self.init_signal()
        spins.set_initial_magnetization()
    
        
        
                         
        # scanner forward process loop
        for r in range(self.NRep):                                   # for all repetitions
            
            self.ROI_signal[0,r,0] =   0
            self.ROI_signal[0,r,1:5] =  torch.sum(spins.M[:,0,self.ROI_def,:],[0]).flatten().detach().cpu()  # hard coded 16
            
            for t in range(self.T):                                      # for all actions
                self.flip(t,r,spins)
                
                #delay = event_time[t,r]**2
                delay = torch.abs(event_time[t,r] + 1e-6) * 1
                #delay = (torch.abs(event_time[t,r]) + 1e-6)
                #delay = torch.sqrt((event_time[t,r]) ** 2 + 1e-6)
                #delay = torch.sqrt((event_time[t,r]) ** 2)
                self.set_relaxation_tensor(spins,delay)
                self.set_freeprecession_tensor(spins,delay)
                self.relax_and_dephase(spins)
                    
                self.grad_precess(t,r,spins)
                self.read_signal(t,r,spins)    
                
                self.ROI_signal[t+1,r,0] =   delay
                self.ROI_signal[t+1,r,1:5] =  torch.sum(spins.M[:,0,self.ROI_def,:],[0]).flatten().detach().cpu()  # hard coded center pixel
                
                self.ROI_signal[t+1,r,5] =  torch.sum(abs(spins.M[:,0,self.ROI_def,2]),[0]).flatten().detach().cpu()  # hard coded center pixel
                
             
    # compute adjoint encoding op-based reco                
    def adjoint(self,spins):
        self.init_reco()

        for t in range(self.T-1,-1,-1):
            if self.adc_mask[t] > 0:
                self.do_grad_adj_reco(t,spins)        
        
        
# Fast, but memory inefficient version (also for now does not support parallel imagigng)
class Scanner_batched_fast(Scanner_batched):
    
    def init_gradient_tensor_holder(self):
        G = torch.zeros((self.T,self.NRep,self.NVox,4,4), dtype=torch.float32)
        G[:,:,:,2,2] = 1
        G[:,:,:,3,3] = 1
         
        G_adj = torch.zeros((self.T,self.NRep,self.NVox,4,4), dtype=torch.float32)
        G_adj[:,:,:,2,2] = 1
        G_adj[:,:,:,3,3] = 1
         
        self.G = self.setdevice(G)
        self.G_adj = self.setdevice(G_adj)
        
    def set_gradient_precession_tensor(self,grad_moms):
        
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
        grads = temp[1:,:,:] - temp[:-1,:,:]        
        
        B0X = torch.unsqueeze(grads[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grads[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        B0_grad_cos = torch.cos(B0_grad)
        B0_grad_sin = torch.sin(B0_grad)
        
        # for backward pass
        B0X = torch.unsqueeze(grad_moms[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * self.rampY
        
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
        
    def grad_precess(self,t,r,spins):
        spins.M = torch.matmul(self.G[t,r,:,:,:],spins.M)        
        
    # TODO: fix
    def grad_precess_allRep(self,spins):
        
        class ExecutionControl(Exception): pass
        raise ExecutionControl('grad_precess_allRep: WIP not implemented')
        
        spins.M = torch.matmul(self.G,spins.M)
        
    def read_signal(self,t,r,spins):
        
        if self.adc_mask[t] > 0:
            
            # parallel imaging disabled for now
            #sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(0).unsqueeze(4))
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(spins.M[:,:,:,:2].shape).float()
                noise = self.setdevice(noise)
                sig = spins.M[:,:,:,:2]
                sig += noise  
            
                self.signal[0,t,r,:2] = ((torch.sum(sig,[0,1,2]) * self.adc_mask[t]))
            else:
                self.signal[:,0,t,r,:2] = ((torch.sum(spins.M[:,:,:,:,:2],[1,2,3]) * self.adc_mask[t]))
            
                
                
    # XXX
    # TODO: noise treatment incosistent with Scanner class protocol
#    def read_signal(self,t,r,spins):
#        if self.adc_mask[t] > 0:
#            
#            sig = torch.sum(spins.M[:,:,0,:,:2,0],[1])
#            sig = torch.matmul(self.B1.unsqueeze(0),sig.unsqueeze(1).unsqueeze(1).unsqueeze(5))
#            
#            self.signal[:,:,t,r,:2] = (torch.sum(sig,[3]) * self.adc_mask[t])
#            
#            if self.noise_std > 0:
#                noise = self.noise_std*torch.randn(self.signal[:,:,t,r,:,0].shape).float()
#                noise[:,:,2:] = 0
#                noise = self.setdevice(noise)
#                self.signal[:,:,t,r,:,0] = self.signal[:,:,t,r,:,0] + noise                    
            
     
        
    # TODO: fix
    def read_signal_allRep(self,t,spins):
        
        class ExecutionControl(Exception): pass
        raise ExecutionControl('read_signal_allRep: WIP not implemented')
        
        if self.adc_mask[t] > 0:
            #import pdb; pdb.set_trace()
            sig = torch.sum(spins.M[:,:,:,:2,0],[0])
            sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(4))
            
            self.signal[:,t,:,:2] = torch.sum(sig,[2]) * self.adc_mask[t]
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.signal[:,t,:,:,0].shape).float()
                noise[:,:,2:] = 0
                noise = self.setdevice(noise)
                self.signal[:,t,:,:,0] = self.signal[:,t,:,:,0] + noise

    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,t,spins):
        
        adc_mask = self.adc_mask.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4)
        s = self.signal * adc_mask
        s = torch.sum(s,1)
        
        r = torch.matmul(self.G_adj.permute([2,3,0,1,4]).contiguous().view([1,self.NVox,4,self.T*self.NRep*4]), s.view([self.batch_size,1,self.T*self.NRep*4,1]))
        
        self.reco = r[:,:,:2,0]
        
        
    
#        
#
#
#    # XXX
#    # reconstruct image readout by readout            
#    def do_grad_adj_reco(self,t,spins):
#        s = self.signal[:,:,t,:,:,:] * self.adc_mask[t]
#        # for now we ignore parallel imaging options here (do naive sum sig over coil)
#        s = torch.sum(s, 1).unsqueeze(1)
#
#        r = torch.matmul(self.G_adj.permute([0,2,1,3,4]), s)
#        self.reco = self.reco + torch.sum(r[:,:,:,:2,0],2)
        



