import numpy as np
import torch

import core.scanner

# variation for supervised learning
# TODO fix relax tensor -- is batch dependent
# TODO implement as child class of Scanner class, override methods
class Scanner_batched(core.scanner.Scanner):
    
    def __init__(self,sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu,batch_size):
        super(Scanner_batched, self).__init__(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu)
        self.batch_size = batch_size    
        
    def set_relaxation_tensor(self,spins,dt):
        R = torch.zeros((self.batch_size,self.NVox,3,3), dtype=torch.float32) 
        
        R = self.setdevice(R)
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,:,0,0] = T2_r
        R[:,:,1,1] = T2_r
        R[:,:,2,2] = T1_r
         
        R = R.view([self.batch_size,1,1,self.NVox,3,3])
        
        self.R = R
        
    def set_B0inhomogeneity_tensor(self,spins,delay):
        S = torch.zeros((self.batch_size,1,1,self.NVox,3,3), dtype=torch.float32)
        S = self.setdevice(S)
        
        B0_inhomo = spins.B0inhomo.view([self.batch_size,self.NVox]) * 2*np.pi
        
        B0_nspins_cos = torch.cos(B0_inhomo*delay)
        B0_nspins_sin = torch.sin(B0_inhomo*delay)
         
        S[:,0,0,:,0,0] = B0_nspins_cos
        S[:,0,0,:,0,1] = -B0_nspins_sin
        S[:,0,0,:,1,0] = B0_nspins_sin
        S[:,0,0,:,1,1] = B0_nspins_cos
         
        S[:,0,0,:,2,2] = 1
         
        self.SB0 = S
        
    def relax(self,spins):
        spins.M = torch.matmul(self.R,spins.M)
        spins.M[:,:,:,:,2,0] += (1 - self.R[:,:,:,:,2,2]).view([self.batch_size,1,1,self.NVox]) * spins.MZ0
        
    def relax_and_dephase(self,spins):
        spins.M = torch.matmul(self.R,spins.M)
        spins.M[:,:,:,:,2,0] += (1 - self.R[:,:,:,:,2,2]).view([self.batch_size,1,1,self.NVox]) * spins.MZ0
        
        spins.M = torch.matmul(self.SB0,spins.M)
        spins.M = torch.matmul(self.P,spins.M)
        

    def init_signal(self):
        signal = torch.zeros((self.batch_size,self.NCoils,self.T,self.NRep,3,1), dtype=torch.float32) 
        self.signal = self.setdevice(signal)
        
        
    def init_reco(self):
        reco = torch.zeros((self.batch_size,self.NVox,2), dtype = torch.float32)
        
        self.reco = self.setdevice(reco)
        
    def do_ifft_reco(self):
        class ExecutionControl(Exception): pass
        raise ExecutionControl('scanner_batched: do_ifft_reco: WIP not implemented')

        
    def read_signal(self,t,r,spins):
        if self.adc_mask[t] > 0:
            if self.noise_std > 0:
                sig = torch.sum(spins.M[:,:,:,:,:2],[0])
                noise = self.noise_std*torch.randn(sig.shape).float()
                noise = self.setdevice(noise)
                sig += noise  
            
                self.signal[:,0,t,r,:2] = ((torch.sum(sig,[1,2]) * self.adc_mask[t])) / self.NSpins
            else:
                self.signal[:,0,t,r,:2] = ((torch.sum(spins.M[:,:,:,:,:2],[1,2,3]) * self.adc_mask[t])) / self.NSpins  
                
    # run throw all repetition/actions and yield signal
    def forward(self,spins,event_time,do_dummy_scans=False):
        self.init_signal()
        if do_dummy_scans == False:
            spins.set_initial_magnetization()
    
        # scanner forward process loop
        for r in range(self.NRep):                                   # for all repetitions
            for t in range(self.T):                                      # for all actions
                delay = torch.abs(event_time[t,r]) + 1e-6
                self.read_signal(t,r,spins)    
                
                self.flip(t,r,spins)
                
                self.set_relaxation_tensor(spins,delay)
                self.set_freeprecession_tensor(spins,delay)
                self.set_B0inhomogeneity_tensor(spins,delay)
                self.relax_and_dephase(spins)
                    
                self.set_grad_op(t)
                self.grad_precess(r,spins)
                
                self.grad_intravoxel_precess(t,r,spins)
                
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)
                
    def forward_mem(self,spins,event_time):
        self.init_signal()
        spins.set_initial_magnetization()
    
        # scanner forward process loop
        for r in range(self.NRep):                                   # for all repetitions
            
            for t in range(self.T):                                      # for all actions
                delay = torch.abs(event_time[t,r]) + 1e-6
                self.read_signal(t,r,spins)    
                
                spins.M = FlipClass.apply(self.F[t,r,:,:,:],spins.M,self)
                
                
                self.set_relaxation_tensor(spins,delay)
                self.set_freeprecession_tensor(spins,delay)
                self.set_B0inhomogeneity_tensor(spins,delay)
                
                spins.M = RelaxClass.apply(self.R,spins.M,delay,t,self,spins)
                spins.M = DephaseClass.apply(self.P,spins.M,self)
                spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                
                self.set_grad_intravoxel_precess_tensor(t,r)
                
                self.set_grad_op(t)
                    
                spins.M = GradPrecessClass.apply(self.G[r,:,:,:],spins.M,self)
                spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                
                
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.batch_size,self.NVox)).byte()
        self.tmask = self.setdevice(self.tmask)
        
        for i in range(self.batch_size):
            self.tmask[i,spins.T1[i,:] < 1e-2] = 1
            self.tmask[i,spins.T2[i,:] < 1e-2] = 1
        
        self.lastM = spins.M.clone()
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)                  
      
    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,t,spins):
        s = self.signal[:,:,t,:,:,:] * self.adc_mask[t]
         
        # for now we ignore parallel imaging options here (do naive sum sig over coil)
        s = torch.sum(s, 1)                                                  
        r = torch.matmul(self.G_adj.permute([1,0,2,3]), s)
        self.reco = self.reco + torch.sum(r[:,:,:,:2,0],2)
        
    def adjoint(self, spins):
        self.init_reco()
        
        for t in range(self.T-1,-1,-1):
            if self.adc_mask[t] > 0:
                self.set_grad_adj_op(t)
                self.do_grad_adj_reco(t,spins)        
        
       

# Fast, but memory inefficient version (also for now does not support parallel imagigng)
class Scanner_fast_batched(core.scanner.Scanner_fast):
    
    def __init__(self,sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu,batch_size):
        super(Scanner_fast_batched, self).__init__(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu)
        self.batch_size = batch_size    
        
    def set_relaxation_tensor(self,spins,dt):
        R = torch.zeros((self.batch_size,self.NVox,3,3), dtype=torch.float32) 
        
        R = self.setdevice(R)
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,:,0,0] = T2_r
        R[:,:,1,1] = T2_r
        R[:,:,2,2] = T1_r
         
        R = R.view([self.batch_size,1,1,self.NVox,3,3])
        
        self.R = R
        
    def set_B0inhomogeneity_tensor(self,spins,delay):
        S = torch.zeros((self.batch_size,1,1,self.NVox,3,3), dtype=torch.float32)
        S = self.setdevice(S)
        
        B0_inhomo = spins.B0inhomo.view([self.batch_size,self.NVox]) * 2*np.pi
        
        B0_nspins_cos = torch.cos(B0_inhomo*delay)
        B0_nspins_sin = torch.sin(B0_inhomo*delay)
         
        S[:,0,0,:,0,0] = B0_nspins_cos
        S[:,0,0,:,0,1] = -B0_nspins_sin
        S[:,0,0,:,1,0] = B0_nspins_sin
        S[:,0,0,:,1,1] = B0_nspins_cos
         
        S[:,0,0,:,2,2] = 1
         
        self.SB0 = S
        
    def set_freeprecession_tensor(self,spins,dt):
        P = torch.zeros((self.batch_size,self.NSpins,1,1,3,3), dtype=torch.float32)
        P = self.setdevice(P)
        
        B0_nspins = spins.omega[:,:,0].view([self.batch_size,self.NSpins])
        
        B0_nspins_cos = torch.cos(B0_nspins*dt)
        B0_nspins_sin = torch.sin(B0_nspins*dt)
         
        P[:,:,0,0,0,0] = B0_nspins_cos
        P[:,:,0,0,0,1] = -B0_nspins_sin
        P[:,:,0,0,1,0] = B0_nspins_sin
        P[:,:,0,0,1,1] = B0_nspins_cos
         
        P[:,:,0,0,2,2] = 1
         
        self.P = P        
        
    def relax(self,spins):
        spins.M = torch.matmul(self.R,spins.M)
        spins.M[:,:,:,:,2,0] += (1 - self.R[:,:,:,:,2,2]).view([self.batch_size,1,1,self.NVox]) * spins.MZ0
        
    def relax_and_dephase(self,spins):
        spins.M = torch.matmul(self.R,spins.M)
        spins.M[:,:,:,:,2,0] += (1 - self.R[:,:,:,:,2,2]).view([self.batch_size,1,1,self.NVox]) * spins.MZ0
        
        spins.M = torch.matmul(self.SB0,spins.M)
        spins.M = torch.matmul(self.P,spins.M)    
        
    def init_signal(self):
        signal = torch.zeros((self.batch_size,self.NCoils,self.T,self.NRep,3,1), dtype=torch.float32) 
        self.signal = self.setdevice(signal)
        
        self.ROI_signal = torch.zeros((self.T+1,self.NRep,5), dtype=torch.float32) # for trans magnetization
        self.ROI_signal = self.setdevice(self.ROI_signal)
        self.ROI_def= int((self.sz[0]/2)*self.sz[1]+ self.sz[1]/2)        
        
        
    def init_reco(self):
        reco = torch.zeros((self.batch_size,self.NVox,2), dtype = torch.float32)
        
        self.reco = self.setdevice(reco)
        
    def do_ifft_reco(self):
        class ExecutionControl(Exception): pass
        raise ExecutionControl('scanner_batched: do_ifft_reco: WIP not implemented')        
    
    def read_signal(self,t,r,spins):
        if self.adc_mask[t] > 0:
            
            if self.noise_std > 0:
                sig = torch.sum(spins.M[:,:,:,:,:2],[0])
                noise = self.noise_std*torch.randn(sig.shape).float()
                noise = self.setdevice(noise)
                sig += noise  
            
                self.signal[:,0,t,r,:2] = ((torch.sum(sig,[1,2]) * self.adc_mask[t])) / self.NSpins
            else:
                self.signal[:,0,t,r,:2] = ((torch.sum(spins.M[:,:,:,:,:2],[1,2,3]) * self.adc_mask[t])) / self.NSpins
     
    # run throw all repetition/actions and yield signal
    def do_dummy_scans(self,spins,event_time,nrep=0):
        class ExecutionControl(Exception): pass
        raise ExecutionControl('scanner_batched_fast: do_dummy_scans: WIP not implemented')    

        
    # run throw all repetition/actions and yield signal
    def forward(self,spins,event_time,do_dummy_scans=False):
        self.init_signal()
        if do_dummy_scans == False:
            spins.set_initial_magnetization()
    
        # scanner forward process loop
        for r in range(self.NRep):                                   # for all repetitions
            for t in range(self.T):                                      # for all actions
                delay = torch.abs(event_time[t,r]) + 1e-6
                self.read_signal(t,r,spins)    
                self.flip(t,r,spins)
                
                self.set_relaxation_tensor(spins,delay)
                self.set_freeprecession_tensor(spins,delay)
                self.set_B0inhomogeneity_tensor(spins,delay)
                self.relax_and_dephase(spins)
                    
                self.grad_precess(t,r,spins)
                self.grad_intravoxel_precess(t,r,spins)
                
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)
                
    def forward_mem(self,spins,event_time):
        self.init_signal()
        spins.set_initial_magnetization()
    
        # scanner forward process loop
        for r in range(self.NRep):                                   # for all repetitions
            for t in range(self.T):                                      # for all actions
                delay = torch.abs(event_time[t,r]) + 1e-6
                self.read_signal(t,r,spins)    
                
                spins.M = FlipClass.apply(self.F[t,r,:,:,:],spins.M,self)
                
                
                self.set_relaxation_tensor(spins,delay)
                self.set_freeprecession_tensor(spins,delay)
                self.set_B0inhomogeneity_tensor(spins,delay)
                
                spins.M = RelaxClass.apply(self.R,spins.M,delay,t,self,spins)
                spins.M = DephaseClass.apply(self.P,spins.M,self)
                spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                
                self.set_grad_intravoxel_precess_tensor(t,r)
                    
                spins.M = GradPrecessClass.apply(self.G[t,r,:,:,:],spins.M,self)
                spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.batch_size,self.NVox)).byte()
        self.tmask = self.setdevice(self.tmask)
        
        for i in range(self.batch_size):
            self.tmask[i,spins.T1[i,:] < 1e-2] = 1
            self.tmask[i,spins.T2[i,:] < 1e-2] = 1
        
        self.lastM = spins.M.clone()
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)      
            
    # compute adjoint encoding op-based reco    <            
    def adjoint(self,spins):
        self.init_reco()
        
        adc_mask = self.adc_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(0)
        s = self.signal * adc_mask
        s = torch.sum(s,1)
        
        r = torch.matmul(self.G_adj.permute([2,3,0,1,4]).contiguous().view([self.NVox,3,self.T*self.NRep*3]), s.view([self.batch_size,1,self.T*self.NRep*3,1]))
        self.reco = r[:,:,:2,0]        

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
        gf = ctx.scanner.lastM.permute([0,1,2,3,5,4]) * grad_output
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
        out[:,:,:,:,2,0] += (1 - f[:,:,:,:,2,2]).view([ctx.scanner.batch_size,1,1,scanner.NVox]) * spins.MZ0
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,1,2,3,5,4]),grad_output)
      
        gf = ctx.scanner.lastM.permute([0,1,2,3,5,4]) * grad_output
        gf = torch.sum(gf,[2])
        
        if ctx.delay > ctx.thresh or np.mod(ctx.t,ctx.scanner.T) == 0:
            ctx.scanner.lastM = ctx.M
        else:
            d1 = ctx.f[:,0,0,:,0,0]
            id1 = 1/d1
            
            d3 = ctx.f[:,0,0,:,2,2]
            id3 = 1/d3
            id3 = id3.view([ctx.scanner.batch_size,1,ctx.scanner.NVox])
            
            ctx.scanner.lastM[:,:,0,:,:2,0] *= id1.view([ctx.scanner.batch_size,1,ctx.scanner.NVox,1])
            ctx.scanner.lastM[:,:,0,:,2,0] = ctx.scanner.lastM[:,:,0,:,2,0]*id3 + (1-id3)*ctx.spins.MZ0[:,:,0,:]
            
            for i in range(ctx.scanner.batch_size):
                ctx.scanner.lastM[i,:,:,ctx.scanner.tmask[i,:],:] = 0
            
        return (gf, gx, None, None, None, None)  
  
class DephaseClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, scanner):
        ctx.f = f.clone()
        ctx.scanner = scanner
        
        return torch.matmul(f,x)

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,1,2,3,5,4]),grad_output)
        
        ctx.scanner.lastM = torch.matmul(ctx.f.permute([0,1,2,3,5,4]),ctx.scanner.lastM)
        
        gf = ctx.scanner.lastM.permute([0,1,2,3,5,4]) * grad_output
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
        gf = ctx.scanner.lastM.permute([0,1,2,3,5,4]) * grad_output
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
        gf = ctx.scanner.lastM.permute([0,1,2,3,5,4]) * grad_output
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
        gx = torch.matmul(ctx.f.permute([0,1,2,3,5,4]),grad_output)
        
        ctx.scanner.lastM = torch.matmul(ctx.f.permute([0,1,2,3,5,4]),ctx.scanner.lastM)
        
        gf = ctx.scanner.lastM.permute([0,1,2,3,5,4]) * grad_output
        gf = torch.sum(gf,[2],keepdim=True)
        
        return (gf, gx, None)     
    
    

        
        