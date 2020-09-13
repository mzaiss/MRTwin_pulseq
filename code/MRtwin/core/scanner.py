import numpy as np
import torch
import time
import os
import sys
from sys import platform
import scipy
import shutil
import socket


#sys.path.append("../scannerloop_libs/twixreader")
#import twixreader as tr # twixreader install: conda install pyyaml; conda install -c certik antlr4-python3-runtime



# general func
def roll(x,n,dim):
    
    if dim == 0:
        return torch.cat((x[-n:], x[:-n]))
    elif dim == 1:
        return torch.cat((x[:, -n:], x[:, :-n]), dim=1)        
    else:
        class ExecutionControl(Exception): pass
        raise ExecutionControl('roll > 2 dim = FAIL!')
        return 0


# HOW we measure
class Scanner():
    def __init__(self,sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu,double_precision=False,do_voxel_rand_ramp_distr=False,do_voxel_rand_r2_distr=False):
        
        self.sz = sz                                             # image size
        self.NVox = sz[0]*sz[1]                                 # voxel count
        self.NSpins = NSpins              # number of spin sims in each voxel
        self.NRep = NRep                              # number of repetitions
        self.T = T                     # number of "actions" within a readout
        self.NCoils = NCoils                # number of receive coil elements
        self.NTransmitCoils = 1
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
        self.double_precision = double_precision
        self.do_voxel_rand_ramp_distr = do_voxel_rand_ramp_distr
        self.do_voxel_rand_r2_distr = do_voxel_rand_r2_distr
        self.NCol = None # number of samples in readout (computed at adc_mask set)
        
        
        # experimental (fast grad op)
        self.collect_presignal = False
        
        # phase cycling
        # array is obsolete: deprecate in future, this dummy is for backward compat
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
        if self.double_precision:
            x = x.double()
        else:
            x = x.float()
        if self.use_gpu > 0:
            x = x.cuda(self.use_gpu-1)
        return x        
    
    def tonumpy(self, x):
        return x.detach().cpu().numpy()    
    
    def do_SAR_test(self, rf_event, event_time):
        TACQ = torch.sum(event_time)*1000 + 2e3
        watchdog_norm = 100 / 0.45245
        SAR_watchdog = (torch.sum(rf_event[:,:,0]**2) / TACQ).cpu()
        print("SAR_watchdog = {}%".format(np.round(SAR_watchdog*watchdog_norm)))
        
        
    def set_adc_mask(self, adc_mask = None):
        if adc_mask is None:
            adc_mask = torch.from_numpy(np.ones((self.T,1))).float()
            self.adc_mask = self.setdevice(adc_mask)
        else:
            self.adc_mask = adc_mask
        
        adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
        self.NCol = adc_idx.size

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
        
        # same intravoxel dephasing ramp random jitter at each voxel
        if not self.do_voxel_rand_ramp_distr:
            if dim == torch.floor(dim):
                xv, yv = torch.meshgrid([torch.linspace(-1+off,1-off,dim.int()), torch.linspace(-1+off,1-off,dim.int())])
                # this generates an anti-symmetric distribution in x
                Rx1= torch.randn(torch.Size([torch.floor_divide(dim.int(),2),dim.int()]))*off
                Rx2=-torch.flip(Rx1, [0])
                Rx= torch.cat((Rx1, Rx2),0)
                # this generates an anti-symmetric distribution in y
                Ry1= torch.randn(torch.Size([dim.int(),torch.floor_divide(dim.int(),2)]))*off
                Ry2=-torch.flip(Ry1, [1])
                Ry= torch.cat((Ry1, Ry2),1)
                            
                xv = xv + Rx
                yv = yv + Ry
    
                intravoxel_dephasing_ramp = np.pi*torch.stack((xv.flatten(),yv.flatten()),1)
            else:
                class ExecutionControl(Exception): pass
                raise ExecutionControl('init_intravoxel_dephasing_ramps: sqrt(NSpins) should be integer!')
                
                intravoxel_dephasing_ramp = np.pi*2*(torch.rand(self.NSpins,2) - 0.5)
                
            # remove coupling w.r.t. R2
            permvec = np.random.choice(self.NSpins,self.NSpins,replace=False)
            intravoxel_dephasing_ramp = intravoxel_dephasing_ramp[permvec,:]
            #intravoxel_dephasing_ramp = np.pi*2*(torch.rand(self.NSpins,2) - 0.5)
            #intravoxel_dephasing_ramp /= torch.from_numpy(self.sz-1).float().unsqueeze(0)
            intravoxel_dephasing_ramp /= torch.from_numpy(self.sz).float().unsqueeze(0)
        else:  # different intravoxel dephasing ramp random jitter at each voxel
            intravoxel_dephasing_ramp = self.setdevice(torch.zeros((self.NSpins,self.NVox,2), dtype=torch.float32))
            xvb, yvb = torch.meshgrid([torch.linspace(-1+off,1-off,dim.int()), torch.linspace(-1+off,1-off,dim.int())])
            
            for i in range(self.NVox):
                # this generates an anti-symmetric distribution in x
                Rx1= torch.randn(torch.Size([dim.int()/2,dim.int()]))*off
                Rx2=-torch.flip(Rx1, [0])
                Rx= torch.cat((Rx1, Rx2),0)
                # this generates an anti-symmetric distribution in y
                Ry1= torch.randn(torch.Size([dim.int(),dim.int()/2]))*off
                Ry2=-torch.flip(Ry1, [1])
                Ry= torch.cat((Ry1, Ry2),1)
                            
                xv = xvb + Rx
                yv = yvb + Ry
            
                intravoxel_dephasing_ramp[:,i,:] = np.pi*torch.stack((xv.flatten(),yv.flatten()),1)
            
            # remove coupling w.r.t. R2
            permvec = np.random.choice(self.NSpins,self.NSpins,replace=False)
            intravoxel_dephasing_ramp = intravoxel_dephasing_ramp[permvec,:,:]
            
            intravoxel_dephasing_ramp /= self.setdevice(torch.from_numpy(self.sz).float().unsqueeze(0).unsqueeze(0))  
            self.intravoxel_dephasing_ramp = intravoxel_dephasing_ramp            
            
        self.intravoxel_dephasing_ramp = self.setdevice(intravoxel_dephasing_ramp)    
        
    # function is obsolete: deprecate in future, this dummy is for backward compat
    def get_ramps(self):
        pass
    
    
    def get_phase_cycler(self, n, dphi):
        out = np.cumsum(np.arange(n) * dphi)
        out = torch.from_numpy(np.mod(out, 360).astype(np.float32))
        
        return out    
        
    def init_coil_sensitivities(self, B1=None):
        # handle complex mul as matrix mul
        B1_init = torch.zeros((self.NCoils,1,self.NVox,2,2), dtype=torch.float32)
        
        if B1 is None:
            B1 = torch.ones((self.NCoils,self.sz[0],self.sz[1],2))     # last dimension real/imag for B1 minus
            B1[:,:,:,1] = 0
        
        B1 = torch.reshape(B1,(self.NCoils,self.NVox,2))
        
        B1_init[:,0,:,0,0] = B1[:,:,0]
        B1_init[:,0,:,0,1] = -B1[:,:,1]
        B1_init[:,0,:,1,0] = B1[:,:,1]
        B1_init[:,0,:,1,1] = B1[:,:,0]
            
        self.B1 = self.setdevice(B1_init)
        
    def init_flip_tensor_holder(self):
        F = torch.zeros((self.T,self.NRep,1,3,3), dtype=torch.float32)
        
        F[:,:,0,1,1] = 1
         
        self.F = self.setdevice(F)
         
    def set_flip_tensor(self,rf_event):
        
        rf_event_cos = torch.cos(rf_event)
        rf_event_sin = torch.sin(rf_event)
        
        self.F[:,:,0,0,0] = rf_event_cos
        self.F[:,:,0,0,2] = rf_event_sin
        self.F[:,:,0,2,0] = -rf_event_sin
        self.F[:,:,0,2,2] = rf_event_cos 
        
    def set_flipXY_tensor(self,input_rf_event):
        
        class ExecutionControl(Exception): pass
        raise ExecutionControl('set_flipXY_tensor method is deprecated! use set_flip_tensor_withB1plus instead')
        
        vx = torch.cos(input_rf_event[:,:,1])
        vy = torch.sin(input_rf_event[:,:,1])
        
        theta = input_rf_event[:,:,0]
            
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
    def set_flip_tensor_withB1plus(self,input_rf_event):
        if not hasattr(self,'B1plus'):
            class ExecutionControl(Exception): pass
            raise ExecutionControl('set_flip_tensor_withB1plus: set B1plus before use')
    
        Fglob = torch.zeros((self.T,self.NRep,1,3,3), dtype=torch.float32)
        
        Fglob[:,:,:,1,1] = 1
         
        Fglob = self.setdevice(Fglob)
        
        vx = torch.cos(input_rf_event[:,:,1])
        vy = torch.sin(input_rf_event[:,:,1])
        
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
        
        theta = input_rf_event[:,:,0]
        theta = theta.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        
        # WARNING TODO, not future proof
        if self.sz[0] > 64:
            theta = theta[:(self.T - self.NCol)//2,:,:,:,:]
            Fglob = Fglob[:(self.T - self.NCol)//2,:,:,:,:]
            F2 = F2[:(self.T - self.NCol)//2,:,:,:,:]        
        
        theta = theta * self.B1plus.view([1,1,self.NVox,1,1])
        
        F = torch.sin(theta)*Fglob + (1 - torch.cos(theta))*F2
        self.F = self.setdevice(F)       
    
        self.F[:,:,:,0,0] += 1
        self.F[:,:,:,1,1] += 1
        self.F[:,:,:,2,2] += 1
        

        
    # use Rodriguez' rotation formula to compute rotation around arbitrary axis
    # rf_event are now (T,NRep,3) -- axis angle representation
    # angle = norm of the rotation vector    
    def set_flipAxisAngle_tensor(self,rf_event):
        
        # ... greatly simplifies if assume rotations in XY plane ...
        theta = torch.norm(rf_event,dim=2).unsqueeze(2)
        v = rf_event / theta
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
        if not self.do_voxel_rand_r2_distr:
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
        else:
            P = torch.zeros((self.NSpins,1,self.NVox,3,3), dtype=torch.float32)
            P = self.setdevice(P)
            
            B0_nspins = spins.omega
            
            B0_nspins_cos = torch.cos(B0_nspins*dt)
            B0_nspins_sin = torch.sin(B0_nspins*dt)
             
            P[:,0,:,0,0] = B0_nspins_cos
            P[:,0,:,0,1] = -B0_nspins_sin
            P[:,0,:,1,0] = B0_nspins_sin
            P[:,0,:,1,1] = B0_nspins_cos
             
            P[:,0,:,2,2] = 1
             
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
        if not self.do_voxel_rand_ramp_distr:
            IVP = torch.zeros((self.NSpins,1,1,3,3), dtype=torch.float32)
            IVP = self.setdevice(IVP)
            IVP[:,0,0,2,2] = 1
            self.IVP = IVP    
        else:
            IVP = torch.zeros((self.NSpins,1,self.NVox,3,3), dtype=torch.float32)
            IVP = self.setdevice(IVP)
            IVP[:,0,:,2,2] = 1
            self.IVP = IVP          
        
    def set_grad_op(self,r):
          
        B0X = torch.unsqueeze(torch.unsqueeze(self.grads[:,r,0],1),1) * self.rampX
        B0Y = torch.unsqueeze(torch.unsqueeze(self.grads[:,r,1],1),1) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NVox])
        
        B0_grad_cos = torch.cos(B0_grad)
        B0_grad_sin = torch.sin(B0_grad)          
       
        self.G[:,:,0,0] = B0_grad_cos
        self.G[:,:,0,1] = -B0_grad_sin
        self.G[:,:,1,0] = B0_grad_sin
        self.G[:,:,1,1] = B0_grad_cos
        
    def set_grad_adj_op(self,r):
          
        B0X = torch.unsqueeze(torch.unsqueeze(self.kspace_loc[:,r,0],1),1) * self.rampX
        B0Y = torch.unsqueeze(torch.unsqueeze(self.kspace_loc[:,r,1],1),1) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NVox])
        
        B0_grad_adj_cos = torch.cos(B0_grad)
        B0_grad_adj_sin = torch.sin(B0_grad)       
        
        self.G_adj[:,:,0,0] = B0_grad_adj_cos
        self.G_adj[:,:,0,1] = B0_grad_adj_sin
        self.G_adj[:,:,1,0] = -B0_grad_adj_sin
        self.G_adj[:,:,1,1] = B0_grad_adj_cos
        
    def set_gradient_precession_tensor(self,gradm_event,sequence_class):
        grads=gradm_event
        
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,gradm_event),0)
        temp = temp[:-1,:,:]
        k=torch.cumsum(temp,0)
        
        # for backward pass
        if sequence_class.lower() == "gre" or sequence_class.lower() == "bssfp":
            pass
        
        if sequence_class.lower() == "gre_dream" :
            temp[0:4,:,:]=0
            k=torch.cumsum(temp,0)
        
        if sequence_class.lower() == "rare":
            refocusing_pulse_action_idx = 1
            kloc = 0
            for r in range(self.NRep):
                for t in range(self.T):
                    if refocusing_pulse_action_idx+1 == t:     # +1 because of right gradmom shift (see above )
                        kloc = -kloc
                    kloc += temp[t,r,:]
                    k[t,r,:] = kloc
                    
        if sequence_class.lower() == "se":
            refocusing_pulse_action_idx = 1
            for r in range(self.NRep):
                kloc = 0
                for t in range(self.T):
                    if refocusing_pulse_action_idx+1 == t:     # +1 because of right gradmom shift (see above )
                        kloc = -kloc
                    kloc += temp[t,r,:]
                    k[t,r,:] = kloc
                    
        if sequence_class.lower() == "epi":
            kloc = 0
            for r in range(self.NRep):
                for t in range(self.T):
                    kloc += temp[t,r,:]
                    k[t,r,:] = kloc              
            
        self.grads = grads
        
        # save gradm_event for intravoxel precession op
        self.gradm_event_for_intravoxel_precession = gradm_event
        
        self.kspace_loc = k
        
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
        
    #def set_grad_intravoxel_precess_tensor_allvox_samedistr(self,t,r):
    def set_grad_intravoxel_precess_tensor(self,t,r):
        # assume voxel-variable ramp distribution

        if not self.do_voxel_rand_ramp_distr:
            intra_b0 = self.gradm_event_for_intravoxel_precession[t,r,:].unsqueeze(0) * self.intravoxel_dephasing_ramp
            
            intra_b0 = torch.sum(intra_b0,1)
            
            IVP_nspins_cos = torch.cos(intra_b0)
            IVP_nspins_sin = torch.sin(intra_b0)
             
            self.IVP[:,0,0,0,0] = IVP_nspins_cos
            self.IVP[:,0,0,0,1] = -IVP_nspins_sin
            self.IVP[:,0,0,1,0] = IVP_nspins_sin
            self.IVP[:,0,0,1,1] = IVP_nspins_cos             
        else:
            self.set_grad_intravoxel_precess_tensor_var_voxramp(t,r)
           
        
    def set_grad_intravoxel_precess_tensor_var_voxramp(self,t,r):
        intra_b0 = self.gradm_event_for_intravoxel_precession[t,r,:].unsqueeze(0).unsqueeze(0) * self.intravoxel_dephasing_ramp
        intra_b0 = torch.sum(intra_b0,2)
        
        IVP_nspins_cos = torch.cos(intra_b0)
        IVP_nspins_sin = torch.sin(intra_b0)
        
        self.IVP[:,0,:,0,0] = IVP_nspins_cos
        self.IVP[:,0,:,0,1] = -IVP_nspins_sin
        self.IVP[:,0,:,1,0] = IVP_nspins_sin
        self.IVP[:,0,:,1,1] = IVP_nspins_cos
   
    # intravoxel gradient-driven precession
    def grad_intravoxel_precess(self,t,r,spins):
        self.set_grad_intravoxel_precess_tensor(t,r)
        if len(self.intravoxel_dephasing_ramp.shape) == 4:
            spins.M = torch.matmul(self.IVP[:,r,:,:,:].unsqueeze(1),spins.M)
        else:
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
        
        if self.NCoils == 1:                      # for backwards compat mostly
            spectrum = self.signal[0,self.adc_mask.flatten()!=0,:,:2,0].clone()
            
            # fftshift
            spectrum = roll(spectrum,self.NCol//2-1,0)
            spectrum = roll(spectrum,self.NRep//2-1,1)
            
            space = torch.ifft(spectrum,2)
            
            if self.NCol > self.sz[0]:
                print("do_ifft_reco: oversampled singal detected, doing crop around center in space...")
                hsz = (self.NCol - self.sz[0])//2
                space = space[hsz:hsz+self.sz[0]]
            
            # fftshift
            space = roll(space,self.sz[0]//2-1,0)
            space = roll(space,self.sz[1]//2-1,1)
           
            self.reco = space.reshape([self.NVox,2])
        else:
            class ExecutionControl(Exception): pass
            raise ExecutionControl('do_ifft_rec: parallel imaging not implemented!')
        
    def do_nufft_reco(self):
        sz = self.sz
        NCol = self.NCol
        NRep = self.NRep
        
        if self.NCoils == 1:                      # for backwards compat mostly
            adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
            spectrum = self.signal[0,self.adc_mask.flatten()!=0,:,:2,0].detach().cpu().numpy()
            
            X, Y = np.meshgrid(np.linspace(0,NCol-1,NCol) - NCol / 2, np.linspace(0,NRep-1,NRep) - NRep/2)
            
            grid = self.kspace_loc[adc_idx,:,:]
            grid = torch.flip(grid, [2]).detach().cpu().numpy()
            
            try:
                spectrum_resampled_x = scipy.interpolate.griddata((grid[:,:,0].ravel(), grid[:,:,1].ravel()), spectrum[:,:,0].ravel(), (X, Y), method='cubic')
                spectrum_resampled_y = scipy.interpolate.griddata((grid[:,:,0].ravel(), grid[:,:,1].ravel()), spectrum[:,:,1].ravel(), (X, Y), method='cubic')
                
                spectrum_resampled = np.stack((spectrum_resampled_x.reshape(sz),spectrum_resampled_y.reshape(sz))).transpose([1,2,0])
                spectrum_resampled[np.isnan(spectrum_resampled)] = 0
                spectrum_resampled = self.setdevice(torch.from_numpy(spectrum_resampled).float())
            except:
                print("do_nufft_reco: FATAL, gridding failed, returning zeros")
                spectrum_resampled = self.setdevice(torch.from_numpy(spectrum))
            
            # fftshift
            spectrum_resampled = roll(spectrum_resampled,NCol//2-1,0)
            spectrum_resampled = roll(spectrum_resampled,NRep//2-1,1)
            
            space = torch.ifft(spectrum_resampled,2)
            
            if NCol > sz[0]:
                print("do_nufft_reco: oversampled singal detected, doing crop around center in space...")
                hsz = (NCol - sz[0])//2
                space = space[hsz:hsz+sz[0]]        
            
            # fftshift
            space = roll(space,self.sz[0]//2-1,0)
            space = roll(space,self.sz[1]//2-1,1)
               
            self.reco = space.reshape([self.NVox,2])
        else:
            class ExecutionControl(Exception): pass
            raise ExecutionControl('do_ifft_rec: parallel imaging not implemented!')            
    
    def discard_out_of_kspace_sig(self):
        kspace_loc_cloned = self.kspace_loc.clone().view([self.T*self.NRep,2])
        out_of_kspace_x = np.where(np.abs(self.tonumpy(kspace_loc_cloned[:,0])) > self.sz[0]/2)[0]
        out_of_kspace_y = np.where(np.abs(self.tonumpy(kspace_loc_cloned[:,1])) > self.sz[1]/2)[0]
        
        signal = self.signal.view([self.NCoils, self.T*self.NRep,3,1])
        signal[:,out_of_kspace_x,:,:] = 0
        signal[:,out_of_kspace_y,:,:] = 0
        self.signal = signal.view([self.NCoils, self.T, self.NRep, 3, 1])        
        
    def read_signal(self,t,r,spins):
        if self.adc_mask[t] != 0:
            
            # parallel imaging disabled for now
            sig = spins.M[:,0,:,:2,0]
            sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(4))
            
            if self.noise_std > 0:
                sig = torch.sum(sig,[1])
                noise = self.noise_std*torch.randn(sig.shape).float()
                noise = self.setdevice(noise)
                sig += noise
            
                self.signal[:,t,r,:2] = ((torch.sum(sig,[1]) * self.adc_mask[t])) / self.NSpins
            else:
                self.signal[:,t,r,:2] = ((torch.sum(sig,[1,2]) * self.adc_mask[t])) / self.NSpins  
                
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
                delay = torch.abs(event_time[t,r]) 
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
        if self.do_voxel_rand_ramp_distr:
            class ExecutionControl(Exception): pass
            raise ExecutionControl('forward_mem:  do_voxel_rand_ramp_distr mode not supported')
        
        self.init_signal()
        if do_dummy_scans == False:
            spins.set_initial_magnetization()
    
        # scanner forward process loop
        for r in range(self.NRep):                                   # for all repetitions
            if compact_grad_tensor:
                self.set_grad_op(r)
                
            for t in range(self.T):                                      # for all actions
                delay = torch.abs(event_time[t,r]) 
                self.read_signal(t,r,spins)
                
                self.ROI_signal[t,r,0] =   delay
                self.ROI_signal[t,r,1:4] =  torch.sum(spins.M[:,0,self.ROI_def,:],[0]).flatten().detach().cpu()  # hard coded center pixel
                self.ROI_signal[t,r,4] =  torch.sum(abs(spins.M[:,0,self.ROI_def,2]),[0]).flatten().detach().cpu()  # hard coded center pixel    
                
                spins.M = FlipClass.apply(self.F[t,r,:,:,:],spins.M,self)
                
                self.set_relaxation_tensor(spins,delay)
                self.set_freeprecession_tensor(spins,delay)
                self.set_B0inhomogeneity_tensor(spins,delay)
                
                spins.M = RelaxRAMClass.apply(self.R,spins.M,delay,t,self,spins)
                spins.M = DephaseClass.apply(self.P,spins.M,self)
                spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                
                if compact_grad_tensor:
                    spins.M = GradPrecessClass.apply(self.G[t,:,:,:],spins.M,self)
                else:
                    spins.M = GradPrecessClass.apply(self.G[t,r,:,:,:],spins.M,self)

                self.set_grad_intravoxel_precess_tensor(t,r)
                spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                
            #if r == 0:
            #    spins.M[:,0,:,:2,0] = 0                
                
#            spins.M[:,0,:,:2,0] = 0
#            spins.set_initial_magnetization()
            
                
                
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.lastM = spins.M.clone()
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)
            
    def forward_sparse(self,spins,event_time,do_dummy_scans=False,compact_grad_tensor=True):
        if self.do_voxel_rand_ramp_distr:
            class ExecutionControl(Exception): pass
            raise ExecutionControl('forward_sparse:  do_voxel_rand_ramp_distr mode not supported')
        
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
                delay = torch.abs(event_time[t,r]) 
                
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
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.tmask = self.tmask[PD0_mask]
        
        self.lastM = spins_cut.clone()
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)   

    # run throw all repetition/actions and yield signal
    # broken broken ... why?
    def forward_fast_broken(self,spins,event_time,do_dummy_scans=False,compact_grad_tensor=True):
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
                
                G = self.G
                G_adj = self.G_adj
                
            for t in range(self.T):                            # for all actions
                delay = torch.abs(event_time[t,r]) 
                
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
                        spins.M = GradPrecessClass.apply(G[t,:,:,:],spins.M,self)
                    else:
                        spins.M = GradPrecessClass.apply(G[t,r,:,:,:],spins.M,self)                

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
                            REW = G_adj[start_t,:,:,:]
                        else:
                            REW = G_adj[start_t,r,:,:,:]
                            
                        # set grad intravoxel precession tensor
                        kum_grad_intravoxel = torch.sum(self.gradm_event_for_intravoxel_precession[start_t:t+1,r,:],0)
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
                                FWD = G_adj[start_t:start_t+half_read*2,:,:,:].permute([0,1,3,2])
                            else:
                                FWD = G_adj[start_t:start_t+half_read*2,r,:,:,:].permute([0,1,3,2])
                                
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
                            FWD = G_adj[t+1,:,:,:].permute([0,2,1])
                        else:
                            FWD = G_adj[t+1,r,:,:,:].permute([0,2,1])
                            
                        spins.M = torch.matmul(torch.matmul(REW, FWD),spins.M)
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay

                    
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.lastM = spins.M.clone() 
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)
            
    def forward_fast(self,spins,event_time,do_dummy_scans=False,compact_grad_tensor=True,kill_transverse=False):
        self.init_signal()
        if do_dummy_scans == False:
            spins.set_initial_magnetization()
        self.reco = 0
       
        half_read = np.int(torch.floor_divide(torch.sum(self.adc_mask != 0) ,2))
        
        PD0_mask = spins.PD0_mask.flatten()
        PD0_mask[:] = True
        PD0_mask = PD0_mask.type(torch.bool)
        
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
                delay = torch.abs(event_time[t,r]) 
                
                if self.adc_mask[t] == 0:                         # regular pass
                    self.read_signal(t,r,spins)
                    
                    self.ROI_signal[t,r,0] =   delay
                    self.ROI_signal[t,r,1:4] =  torch.sum(spins.M[:,0,0,:],[0]).flatten().detach().cpu()  # hard coded pixel id 0
                    self.ROI_signal[t,r,4] =  torch.sum(abs(spins.M[:,0,0,2]),[0]).flatten().detach().cpu()  # hard coded pixel id 0                        
                    
                    if t < self.F.shape[0]:
                         spins.M = FlipClass.apply(self.F[t,r,:,:,:],spins.M,self)
                    
                    self.set_relaxation_tensor(spins,delay)
                    self.set_freeprecession_tensor(spins,delay)
                    self.set_B0inhomogeneity_tensor(spins,delay)
                    
                    spins.M = RelaxRAMClass.apply(self.R,spins.M,delay,t,self,spins)
                    spins.M = DephaseClass.apply(self.P,spins.M,self)
                    spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                    
                    if compact_grad_tensor:
                        spins.M = GradPrecessClass.apply(G_cut[t,:,:,:],spins.M,self)
                    else:
                        spins.M = GradPrecessClass.apply(G_cut[t,r,:,:,:],spins.M,self)
                    
                    self.set_grad_intravoxel_precess_tensor(t,r)
                    spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                else:
                    if self.adc_mask[t-1] == 0:        # first sample in readout
                        total_delay = delay
                        start_t = t
                        
                        # set B0 inhomo precession tensor
                        S = torch.zeros((1,self.NCol,self.NVox,2,2), dtype=torch.float32)
                        S = self.setdevice(S)
                        
                        B0_inhomo = spins.B0inhomo.view([self.NVox]) * 2*np.pi
                        delay_ramp = delay * self.setdevice(torch.from_numpy(np.arange(self.NCol))).view([self.NCol,1]) * B0_inhomo.view([1, self.NVox])
                        
                        B0_nspins_cos = torch.cos(delay_ramp)
                        B0_nspins_sin = torch.sin(delay_ramp)
                         
                        S[0,:,:,0,0] = B0_nspins_cos
                        S[0,:,:,0,1] = -B0_nspins_sin
                        S[0,:,:,1,0] = B0_nspins_sin
                        S[0,:,:,1,1] = B0_nspins_cos
                        
                        self.SB0sig = S                        
                        
                    elif t == (self.T - half_read*2)//2 + half_read or self.adc_mask[t+1] == 0:
                        
                        self.set_relaxation_tensor(spins,total_delay)
                        self.set_freeprecession_tensor(spins,total_delay)
                        self.set_B0inhomogeneity_tensor(spins,total_delay)                        
                        
                        spins.M = RelaxRAMClass.apply(self.R, spins.M,total_delay,t,self,spins)
                        spins.M = DephaseClass.apply(self.P,spins.M,self)                        
                        
                        # do gradient precession (use adjoint as free kumulator)
                        if compact_grad_tensor:
                            REW = G_adj_cut[start_t,:,:,:]
                        else:
                            REW = G_adj_cut[start_t,r,:,:,:]                        
                        
                        # read signal
                        if t == (self.T - half_read*2)//2 + half_read:  # read signal
                            if compact_grad_tensor:
                                FWD = G_adj_cut[start_t:start_t+half_read*2,:,:,:].permute([0,1,3,2])
                            else:
                                FWD = G_adj_cut[start_t:start_t+half_read*2,r,:,:,:].permute([0,1,3,2])
                                
                            FWD = torch.matmul(REW, FWD)
                            
                            #intraSpins = spins.M
                            
                            kum_grad_intravoxel = torch.cumsum(self.gradm_event_for_intravoxel_precession[start_t:start_t+half_read*2,r,:],0)
                            
                            if not self.do_voxel_rand_ramp_distr:
                                intra_b0 = kum_grad_intravoxel.unsqueeze(0) * self.intravoxel_dephasing_ramp.unsqueeze(1)
                                intra_b0 = torch.sum(intra_b0,2)
                                
                                IVP_nspins_cos = torch.cos(intra_b0)
                                IVP_nspins_sin = torch.sin(intra_b0)                            
                                
                                IVP = torch.zeros((self.NSpins,self.NCol,1,3,3), dtype=torch.float32)
                                IVP = self.setdevice(IVP)
                                IVP[:,:,0,2,2] = 1
                                
                                IVP[:,:,0,0,0] = IVP_nspins_cos
                                IVP[:,:,0,0,1] = -IVP_nspins_sin
                                IVP[:,:,0,1,0] = IVP_nspins_sin
                                IVP[:,:,0,1,1] = IVP_nspins_cos
                            else:
                                intra_b0 = kum_grad_intravoxel.unsqueeze(0).unsqueeze(2) * self.intravoxel_dephasing_ramp.unsqueeze(1)
                                intra_b0 = torch.sum(intra_b0,3)
                                
                                IVP_nspins_cos = torch.cos(intra_b0)
                                IVP_nspins_sin = torch.sin(intra_b0)                            
                                
                                IVP = torch.zeros((self.NSpins,self.NCol,self.NVox,3,3), dtype=torch.float32)
                                IVP = self.setdevice(IVP)
                                IVP[:,:,:,2,2] = 1
                                
                                IVP[:,:,:,0,0] = IVP_nspins_cos
                                IVP[:,:,:,0,1] = -IVP_nspins_sin
                                IVP[:,:,:,1,0] = IVP_nspins_sin
                                IVP[:,:,:,1,1] = IVP_nspins_cos                            
                            #intraSpins = torch.matmul(IVP, intraSpins)
                            
                            S = torch.zeros((1,self.NCol,self.NVox,3,3), dtype=torch.float32)
                            S = self.setdevice(S)
                            
                            B0_inhomo = spins.B0inhomo.view([self.NVox]) * 2*np.pi
                            delay_ramp = delay * self.setdevice(torch.from_numpy(np.arange(self.NCol))).view([self.NCol,1]) * B0_inhomo.view([1, self.NVox])
                            
                            B0_nspins_cos = torch.cos(delay_ramp)
                            B0_nspins_sin = torch.sin(delay_ramp)
                             
                            S[0,:,:,0,0] = B0_nspins_cos
                            S[0,:,:,0,1] = -B0_nspins_sin
                            S[0,:,:,1,0] = B0_nspins_sin
                            S[0,:,:,1,1] = B0_nspins_cos
                            S[0,:,:,2,2] = 1
                             
                            #intraSpins = torch.matmul(S,intraSpins)
                            tmp = torch.einsum('ijklm,inomp->jolp', [IVP, spins.M]).unsqueeze(0)
                            intraSpins = torch.einsum('sjorl,ijolp->sjorp', [S, tmp])
                            
                            #intraSpins = torch.einsum("ijklm,njomp,irops->nrols",[IVP,S,spins.M])
                            signal = torch.matmul(FWD,intraSpins)
                            
                            signal = torch.matmul(self.B1, signal[:,:,:,:2,:1])
                            signal = torch.sum(signal,[2])
                            
                            self.signal[:,start_t:start_t+half_read*2,r,:2,0] = signal.squeeze() / self.NSpins 
                            
                        # do gradient precession (use adjoint as free kumulator)
                        if compact_grad_tensor:
                            FWD = G_adj_cut[t+1,:,:,:].permute([0,2,1])
                        else:
                            FWD = G_adj_cut[t+1,r,:,:,:].permute([0,2,1])
                            
                        # set grad intravoxel precession tensor
                        kum_grad_intravoxel = torch.sum(self.gradm_event_for_intravoxel_precession[start_t:t+1,r,:],0)
                        
                        if not self.do_voxel_rand_ramp_distr:
                            intra_b0 = kum_grad_intravoxel.unsqueeze(0) * self.intravoxel_dephasing_ramp
                            intra_b0 = torch.sum(intra_b0,1)
                            
                            IVP_nspins_cos = torch.cos(intra_b0)
                            IVP_nspins_sin = torch.sin(intra_b0)
                             
                            self.IVP[:,0,0,0,0] = IVP_nspins_cos
                            self.IVP[:,0,0,0,1] = -IVP_nspins_sin
                            self.IVP[:,0,0,1,0] = IVP_nspins_sin
                            self.IVP[:,0,0,1,1] = IVP_nspins_cos 
                        else:
                            intra_b0 = kum_grad_intravoxel.unsqueeze(0).unsqueeze(0) * self.intravoxel_dephasing_ramp
                            intra_b0 = torch.sum(intra_b0,2)
                            
                            IVP_nspins_cos = torch.cos(intra_b0)
                            IVP_nspins_sin = torch.sin(intra_b0)
                             
                            self.IVP[:,0,:,0,0] = IVP_nspins_cos
                            self.IVP[:,0,:,0,1] = -IVP_nspins_sin
                            self.IVP[:,0,:,1,0] = IVP_nspins_sin
                            self.IVP[:,0,:,1,1] = IVP_nspins_cos                             
                        
                        # do intravoxel precession
                        spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                        spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                            
                        FWD = torch.matmul(REW, FWD)
                        spins.M = torch.matmul(FWD,spins.M)                            
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay
                if self.grads[t,r,0] > 1.5*np.abs(self.sz[0]) and kill_transverse:
                    spins.M[:,0,:,:2,0] = 0                               
            
                    
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.lastM = spins.M.clone()
        
        if self.noise_std > 0:
            noise = self.noise_std*torch.randn(self.signal[:,:,:,:2,0].shape).float()
            noise = self.setdevice(noise)
            self.signal[:,:,:,:2,0] += noise * self.adc_mask.view([self.T,1,1])
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)               
            
    # run throw all repetition/actions and yield signal
    def forward_sparse_fast(self,spins,event_time,do_dummy_scans=False,compact_grad_tensor=True,kill_transverse=False):
        if self.do_voxel_rand_ramp_distr:
            class ExecutionControl(Exception): pass
            raise ExecutionControl('forward_sparse_fast:  do_voxel_rand_ramp_distr mode not supported')

        self.init_signal()
        if do_dummy_scans == False:
            spins.set_initial_magnetization()
        self.reco = 0
        
        half_read = np.int(torch.sum(self.adc_mask != 0) / 2)
        
        PD0_mask = spins.PD0_mask.flatten().bool()
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
                delay = torch.abs(event_time[t,r]) 
                
                if self.adc_mask[t] == 0:                         # regular pass
                    self.read_signal(t,r,spins)
                    if t < self.F.shape[0]:
                        if self.F.shape[2] > 1:
                              spins_cut = FlipClass.apply(self.F[t,r,PD0_mask,:,:],spins_cut,self)
                        else:
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
                        
                        spins_cut = RelaxSparseClass.apply(self.R[:,PD0_mask,:,:], spins_cut,total_delay,t,self,spins,PD0_mask)
                        spins_cut = DephaseClass.apply(self.P,spins_cut,self)
                        
                        # do gradient precession (use adjoint as free kumulator)
                        if compact_grad_tensor:
                            REW = G_adj_cut[start_t,:,:,:]
                        else:
                            REW = G_adj_cut[start_t,r,:,:,:]
                        
                        # read signal
                        if t == (self.T - half_read*2)//2 + half_read:  # read signal
                            if compact_grad_tensor:
                                FWD = G_adj_cut[start_t:start_t+half_read*2,:,:,:].permute([0,1,3,2])
                            else:
                                FWD = G_adj_cut[start_t:start_t+half_read*2,r,:,:,:].permute([0,1,3,2])
                                
                            FWD = torch.matmul(REW, FWD)
                            
                            #intraSpins = spins_cut
                            
                            kum_grad_intravoxel = torch.cumsum(self.gradm_event_for_intravoxel_precession[start_t:start_t+half_read*2,r,:],0)
                            intra_b0 = kum_grad_intravoxel.unsqueeze(0) * self.intravoxel_dephasing_ramp.unsqueeze(1)
                            intra_b0 = torch.sum(intra_b0,2)
                            
                            IVP_nspins_cos = torch.cos(intra_b0)
                            IVP_nspins_sin = torch.sin(intra_b0)                            
                            
                            IVP = torch.zeros((self.NSpins,self.NCol,1,3,3), dtype=torch.float32)
                            IVP = self.setdevice(IVP)
                            IVP[:,:,0,2,2] = 1
                            
                            IVP[:,:,0,0,0] = IVP_nspins_cos
                            IVP[:,:,0,0,1] = -IVP_nspins_sin
                            IVP[:,:,0,1,0] = IVP_nspins_sin
                            IVP[:,:,0,1,1] = IVP_nspins_cos
                            
                            #intraSpins = torch.matmul(IVP, intraSpins)
                            
                            B0_inhomo = spins.B0inhomo.view([self.NVox]) * 2*np.pi
                            B0_inhomo = B0_inhomo[PD0_mask]
                            delay_ramp = delay * self.setdevice(torch.from_numpy(np.arange(self.NCol))).view([self.NCol,1]) * B0_inhomo.view([1, B0_inhomo.shape[0]])
                            
                            B0_nspins_cos = torch.cos(delay_ramp)
                            B0_nspins_sin = torch.sin(delay_ramp)
                            
                            S = torch.zeros((1,self.NCol,B0_inhomo.shape[0],3,3), dtype=torch.float32)
                            S = self.setdevice(S)                            
                             
                            S[0,:,:,0,0] = B0_nspins_cos
                            S[0,:,:,0,1] = -B0_nspins_sin
                            S[0,:,:,1,0] = B0_nspins_sin
                            S[0,:,:,1,1] = B0_nspins_cos
                            S[0,:,:,2,2] = 1
                            
                            #intraSpins = torch.matmul(S,intraSpins)
                            
                            tmp = torch.einsum('ijklm,inomp->jolp', [IVP, spins_cut]).unsqueeze(0)
                            intraSpins = torch.einsum('sjorl,ijolp->sjorp', [S, tmp])                            
                            
                            #intraSpins = torch.einsum("ijklm,njomp,irops->nrols",[IVP,S,spins_cut])
                            signal = torch.matmul(FWD,intraSpins)
                            signal = torch.sum(signal,[2])
                            
                            self.signal[0,start_t:start_t+half_read*2,r,:,0] = signal.squeeze() / self.NSpins 
                            
                        # do gradient precession (use adjoint as free kumulator)
                        if compact_grad_tensor:
                            FWD = G_adj_cut[t+1,:,:,:].permute([0,2,1])
                        else:
                            FWD = G_adj_cut[t+1,r,:,:,:].permute([0,2,1])
                            
                        # set grad intravoxel precession tensor
                        kum_grad_intravoxel = torch.sum(self.gradm_event_for_intravoxel_precession[start_t:t+1,r,:],0)
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
                        spins_cut = B0InhomoClass.apply(self.SB0[:,:,PD0_mask,:,:],spins_cut,self)
                        
                        FWD = torch.matmul(REW, FWD)
                        spins_cut = torch.matmul(FWD,spins_cut)                            
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay
                if self.grads[t,r,0] > 1.5*np.abs(self.sz[0]) and kill_transverse:
                    spins_cut[:,0,:,:2,0] = 0                    
            
                    
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.tmask = self.tmask[PD0_mask]
        self.lastM = spins_cut.clone()
        
        if self.noise_std > 0:
            noise = self.noise_std*torch.randn(self.signal[0,:,:,:2].shape).float()
            noise = self.setdevice(noise)
            self.signal[0,:,:,:2] += noise * self.adc_mask.view([self.T,1,1])
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)            
            

    def forward_fast_supermem(self,spins,event_time):
        if self.do_voxel_rand_ramp_distr:
            class ExecutionControl(Exception): pass
            raise ExecutionControl('forward_fast_supermem:  do_voxel_rand_ramp_distr mode not supported')
        
        self.init_signal()
        spins.set_initial_magnetization()
        self.reco = 0
        
        half_read = np.int(torch.sum(self.adc_mask != 0) / 2)
        
        PD0_mask = spins.PD0_mask.flatten()
        PD0_mask[:] = 1
        
        class AuxGradMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, f, x, scanner):
                ctx.f = f.clone()
                ctx.scanner = scanner
                
                B0X = f[0] * scanner.rampX
                B0Y = f[1] * scanner.rampY
                
                B0_grad = (B0X + B0Y).view([scanner.NVox])
                
                B0_grad_cos = torch.cos(B0_grad)
                B0_grad_sin = torch.sin(B0_grad)          
               
                G = scanner.setdevice(torch.zeros((scanner.NVox,2,2), dtype=torch.float32))
                G[:,0,0] = B0_grad_cos
                G[:,0,1] = -B0_grad_sin
                G[:,1,0] = B0_grad_sin
                G[:,1,1] = B0_grad_cos    
                
                return torch.matmul(G, x)
            
            @staticmethod
            def backward(ctx, grad_output):
                scanner = ctx.scanner
                
                B0X = ctx.f[0] * scanner.rampX
                B0Y = ctx.f[1] * scanner.rampY
                
                B0_grad = (B0X + B0Y).view([scanner.NVox])
                
                B0_grad_cos = torch.cos(B0_grad)
                B0_grad_sin = torch.sin(B0_grad)          
               
                G = scanner.setdevice(torch.zeros((scanner.NVox,2,2), dtype=torch.float32))
                G[:,0,0] = B0_grad_cos
                G[:,0,1] = -B0_grad_sin
                G[:,1,0] = B0_grad_sin
                G[:,1,1] = B0_grad_cos
                
                ctx.scanner.lastM[:,:,:,:2,:] = torch.matmul(G.permute([0,2,1]),ctx.scanner.lastM[:,:,:,:2,:])
                gx = torch.matmul(G.permute([0,2,1]),grad_output[:,:,:,:2,:])
                
                gft = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
                gft = torch.sum(gft,[0,1])
                
                #B0_grad_cos_g = -torch.sin(gf[:,0,0])
                #B0_grad_sin_g = torch.cos(gf[:,1,0])
                
                GG = scanner.setdevice(torch.zeros((scanner.NVox,2,2), dtype=torch.float32))
                GG[:,0,0] = -B0_grad_sin
                GG[:,0,1] = -B0_grad_cos
                GG[:,1,0] = B0_grad_cos
                GG[:,1,1] = -B0_grad_sin  
                
                GG = gft[:,:,:2]*GG
                
                gf = scanner.setdevice(torch.zeros((2,), dtype=torch.float32))
                
                gf[0] = torch.sum(torch.sum(GG,[1,2]) * scanner.rampX.squeeze())
                gf[1] = torch.sum(torch.sum(GG,[1,2])  * scanner.rampY.squeeze())
                
                return (gf, gx, None)
            
        class AuxGetSignalGradMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, f, x, scanner, t1, t2):
                ctx.f = f.clone()
                ctx.scanner = scanner
                ctx.t1 = t1
                ctx.t2 = t2
                
                nmb_a = f.shape[0]
                
                # Intra-voxel grad precession
                intraSpins = x
                            
                kum_grad_intravoxel = torch.cumsum(ctx.scanner.gradm_event_for_intravoxel_precession[t1:t2,r,:],0)
                intra_b0 = kum_grad_intravoxel.unsqueeze(0) * ctx.scanner.intravoxel_dephasing_ramp.unsqueeze(1)
                intra_b0 = torch.sum(intra_b0,2)
                
                IVP_nspins_cos = torch.cos(intra_b0)
                IVP_nspins_sin = torch.sin(intra_b0)                            
                
                IVP = torch.zeros((ctx.scanner.NSpins,ctx.scanner.NCol,1,2,2), dtype=torch.float32)
                IVP = ctx.scanner.setdevice(IVP)
                
                IVP[:,:,0,0,0] = IVP_nspins_cos
                IVP[:,:,0,0,1] = -IVP_nspins_sin
                IVP[:,:,0,1,0] = IVP_nspins_sin
                IVP[:,:,0,1,1] = IVP_nspins_cos
                
                #intraSpins = torch.matmul(IVP, intraSpins)
                
                # Inter-voxel grad precession
                #presignal = torch.sum(intraSpins,0,keepdim=True)
                #presignal = torch.einsum('ijklm,inomp->jolp', [IVP, intraSpins]).unsqueeze(0)
                #presignal = torch.matmul(scanner.SB0sig,presignal)
                
                tmp = torch.einsum('ijklm,inomp->jolp', [IVP, intraSpins]).unsqueeze(0)
                presignal = torch.einsum('sjorl,ijolp->sjorp', [scanner.SB0sig, tmp])
                
                #presignal = torch.einsum('sjorl,ijklm,inomp->jorp', [scanner.SB0sig, IVP, intraSpins]).unsqueeze(0)
                
                B0X = torch.unsqueeze(torch.unsqueeze(f[:,0],1),1) * scanner.rampX
                B0Y = torch.unsqueeze(torch.unsqueeze(f[:,1],1),1) * scanner.rampY
                
                B0_grad = (B0X + B0Y).view([nmb_a,scanner.NVox])
                
                B0_grad_adj_cos = torch.cos(B0_grad)
                B0_grad_adj_sin = torch.sin(B0_grad)       
                
                G_adj = scanner.setdevice(torch.zeros((nmb_a,scanner.NVox,2,2), dtype=torch.float32))
                G_adj[:,:,0,0] = B0_grad_adj_cos
                G_adj[:,:,0,1] = B0_grad_adj_sin
                G_adj[:,:,1,0] = -B0_grad_adj_sin
                G_adj[:,:,1,1] = B0_grad_adj_cos
                
                REW = G_adj[0,:,:,:]
                FWD = G_adj[:,:,:,:].permute([0,1,3,2])
                FWD = torch.matmul(REW, FWD)                
                
                return torch.matmul(FWD, presignal)
            
            @staticmethod
            def backward(ctx, grad_output):
                scanner = ctx.scanner
                
                nmb_a = ctx.f.shape[0]
                
                # Inter-voxel grad precession
                B0X = torch.unsqueeze(torch.unsqueeze(ctx.f[:,0],1),1) * scanner.rampX
                B0Y = torch.unsqueeze(torch.unsqueeze(ctx.f[:,1],1),1) * scanner.rampY
                
                B0_grad = (B0X + B0Y).view([nmb_a,scanner.NVox])
                
                B0_grad_adj_cos = torch.cos(B0_grad)
                B0_grad_adj_sin = torch.sin(B0_grad)       
                
                G_adj = scanner.setdevice(torch.zeros((nmb_a,scanner.NVox,2,2), dtype=torch.float32))
                G_adj[:,:,0,0] = B0_grad_adj_cos
                G_adj[:,:,0,1] = B0_grad_adj_sin
                G_adj[:,:,1,0] = -B0_grad_adj_sin
                G_adj[:,:,1,1] = B0_grad_adj_cos
                
                REW = G_adj[0,:,:,:]
                FWD = G_adj[:,:,:,:].permute([0,1,3,2])
                FWD = torch.matmul(REW, FWD)  
                
                grad_output = torch.matmul(FWD.permute([0,1,3,2]),grad_output[:,:,:,:2,:])
                #grad_output = torch.repeat_interleave(grad_output,ctx.scanner.NSpins,0)
                
                # Intra-voxel grad precession
                kum_grad_intravoxel = torch.cumsum(ctx.scanner.gradm_event_for_intravoxel_precession[ctx.t1:ctx.t2,r,:],0)
                intra_b0 = kum_grad_intravoxel.unsqueeze(0) * ctx.scanner.intravoxel_dephasing_ramp.unsqueeze(1)
                intra_b0 = torch.sum(intra_b0,2)
                
                IVP_nspins_cos = torch.cos(intra_b0)
                IVP_nspins_sin = torch.sin(intra_b0)
                
                IVP = torch.zeros((ctx.scanner.NSpins,ctx.scanner.NCol,1,2,2), dtype=torch.float32)
                IVP = ctx.scanner.setdevice(IVP)
                
                IVP[:,:,0,0,0] = IVP_nspins_cos
                IVP[:,:,0,0,1] = -IVP_nspins_sin
                IVP[:,:,0,1,0] = IVP_nspins_sin
                IVP[:,:,0,1,1] = IVP_nspins_cos
                
                #gx = torch.matmul(IVP.permute([0,1,2,4,3]), grad_output)
                #gx = torch.sum(gx,1, keepdim=True)
                #gx = torch.matmul(ctx.scanner.SB0sig.permute([0,1,2,4,3]),grad_output)
                
                tmp = torch.einsum('sjolr,ijolp->sjorp', [scanner.SB0sig, grad_output])
                gx = torch.einsum('ijkml,sjomp->isolp', [IVP, tmp])
                
                
                #gx = torch.einsum('ijkmr,sjolm,njolp->inorp', [IVP,ctx.scanner.SB0sig,grad_output])
                
                
                return (None, gx, None, None, None)
            
        class AuxReadoutGradMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, f, x, scanner):
                ctx.f = f.clone()
                ctx.scanner = scanner
                
                nmb_a = f.shape[0]
                
                B0X = torch.unsqueeze(torch.unsqueeze(ctx.f[:,0],1),1) * scanner.rampX
                B0Y = torch.unsqueeze(torch.unsqueeze(ctx.f[:,1],1),1) * scanner.rampY
                
                B0_grad = (B0X + B0Y).view([nmb_a,scanner.NVox])
                
                B0_grad_adj_cos = torch.cos(B0_grad)
                B0_grad_adj_sin = torch.sin(B0_grad)       
                
                G_adj = scanner.setdevice(torch.zeros((nmb_a,scanner.NVox,2,2), dtype=torch.float32))
                G_adj[:,:,0,0] = B0_grad_adj_cos
                G_adj[:,:,0,1] = B0_grad_adj_sin
                G_adj[:,:,1,0] = -B0_grad_adj_sin
                G_adj[:,:,1,1] = B0_grad_adj_cos
                
                REW = G_adj[0,:,:,:]
                FWD = G_adj[1,:,:,:].permute([0,2,1])
                FWD = torch.matmul(REW, FWD)
                
                return torch.matmul(FWD,x)
            
            @staticmethod
            def backward(ctx, grad_output):
                scanner = ctx.scanner
                
                nmb_a = ctx.f.shape[0]
                
                B0X = torch.unsqueeze(torch.unsqueeze(ctx.f[:,0],1),1) * scanner.rampX
                B0Y = torch.unsqueeze(torch.unsqueeze(ctx.f[:,1],1),1) * scanner.rampY
                
                B0_grad = (B0X + B0Y).view([nmb_a,scanner.NVox])
                
                B0_grad_adj_cos = torch.cos(B0_grad)
                B0_grad_adj_sin = torch.sin(B0_grad)       
                
                G_adj = scanner.setdevice(torch.zeros((nmb_a,scanner.NVox,2,2), dtype=torch.float32))
                G_adj[:,:,0,0] = B0_grad_adj_cos
                G_adj[:,:,0,1] = B0_grad_adj_sin
                G_adj[:,:,1,0] = -B0_grad_adj_sin
                G_adj[:,:,1,1] = B0_grad_adj_cos
                
                REW = G_adj[0,:,:,:]
                FWD = G_adj[1,:,:,:].permute([0,2,1])
                FWD = torch.matmul(REW, FWD)
                
                ctx.scanner.lastM[:,:,:,:2,:] = torch.matmul(FWD.permute([0,2,1]),ctx.scanner.lastM[:,:,:,:2,:])
                gx = torch.matmul(FWD.permute([0,2,1]),grad_output[:,:,:,:2,:])
                
                return (None, gx, None)  
            
        class RelaxSupermemClass(torch.autograd.Function):
            @staticmethod
            def forward(ctx, f, x, delay, t, scanner, spins):
                ctx.f = f.clone()
                ctx.delay = delay
                ctx.scanner = scanner
                ctx.spins = spins
                ctx.t = t
                ctx.thresh = 1e-2
                
                if ctx.delay > ctx.thresh or ctx.t == 0 and False:
                    ctx.M = x.clone()
                    
                out = torch.matmul(f,x)
                out[:,:,:,2,0] += (1 - f[:,:,2,2]).view([1,1,scanner.NVox]) * spins.MZ0
                    
                return out
        
            @staticmethod
            def backward(ctx, grad_output):
                gx = torch.matmul(ctx.f.permute([0,1,3,2]),grad_output)
              
                gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
                gf[:,:,:,2,2] -= ctx.spins.MZ0 * grad_output[:,:,:,2,0]
                gf = torch.sum(gf,[0])
                
                if ctx.delay > ctx.thresh or ctx.t == 0 and False:
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
            
        class RelaxSupermemRAMClass(torch.autograd.Function):
            @staticmethod
            def forward(ctx, f, x, delay, t, r, scanner, spins):
                ctx.f = f.clone()
                ctx.delay = delay
                ctx.scanner = scanner
                ctx.spins = spins
                ctx.t = t
                ctx.r = r
                ctx.thresh = 1e-2
                
                #if ctx.delay > ctx.thresh or (np.mod(ctx.r,16) == 0 and ctx.t == 0):
                #if ctx.delay > ctx.thresh or ctx.t == 0:
                if ctx.delay > ctx.thresh or ctx.t == 0 or True:
                    ctx.M = x.clone().cpu()
                    
                out = torch.matmul(f,x)
                out[:,:,:,2,0] += (1 - f[:,:,2,2]).view([1,1,scanner.NVox]) * spins.MZ0
                    
                return out
        
            @staticmethod
            def backward(ctx, grad_output):
                gx = torch.matmul(ctx.f.permute([0,1,3,2]),grad_output)
              
                gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
                gf[:,:,:,2,2] -= ctx.spins.MZ0 * grad_output[:,:,:,2,0]
                gf = torch.sum(gf,[0])
                
                #if ctx.delay > ctx.thresh or (np.mod(ctx.r,16) == 0 and ctx.t == 0):
                #if ctx.delay > ctx.thresh:
                if ctx.delay > ctx.thresh or ctx.t == 0 or True:
                    ctx.scanner.lastM = ctx.scanner.setdevice(ctx.M)
                else:
                    d1 = ctx.f[0,:,0,0]
                    id1 = 1/d1
                    
                    d3 = ctx.f[0,:,2,2]
                    id3 = 1/d3
                    id3 = id3.view([1,ctx.scanner.NVox])
                    
                    ctx.scanner.lastM[:,0,:,:2,0] *= id1.view([1,ctx.scanner.NVox,1])
                    ctx.scanner.lastM[:,0,:,2,0] = ctx.scanner.lastM[:,0,:,2,0]*id3 + (1-id3)*ctx.spins.MZ0[:,0,:]
                    
                    ctx.scanner.lastM[:,:,ctx.scanner.tmask,:] = 0
                    
                return (gf, gx, None, None, None, None, None)           
        
        # scanner forward process loop
        for r in range(self.NRep):                         # for all repetitions
            total_delay = 0
            start_t = 0
            
            for t in range(self.T):                            # for all actions
                delay = torch.abs(event_time[t,r]) 
                
                if self.adc_mask[t] == 0:                         # regular pass
                    self.read_signal(t,r,spins)
                    if t < self.F.shape[0]:
                         spins.M = FlipClass.apply(self.F[t,r,:,:,:],spins.M,self)
                    
                    self.set_relaxation_tensor(spins,delay)
                    self.set_freeprecession_tensor(spins,delay)
                    self.set_B0inhomogeneity_tensor(spins,delay)
                    
                    #broken
                    spins.M = RelaxSupermemRAMClass.apply(self.R,spins.M,delay,t,r,self,spins)
                    #spins.M = RelaxClass.apply(self.R,spins.M,delay,t,self,spins)
                    
                    spins.M = DephaseClass.apply(self.P,spins.M,self)
                    spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                    
                    spins.M[:,:,:,:2,:] = AuxGradMul.apply(self.grads[t,r,:],spins.M[:,:,:,:2,:],self)
                    
                    self.set_grad_intravoxel_precess_tensor(t,r)
                    spins.M = GradIntravoxelPrecessClass.apply(self.IVP,spins.M,self)
                else:
                    if self.adc_mask[t-1] == 0:        # first sample in readout
                        total_delay = delay
                        start_t = t
                        
                        # set B0 inhomo precession tensor
                        S = torch.zeros((1,self.NCol,self.NVox,2,2), dtype=torch.float32)
                        S = self.setdevice(S)
                        
                        B0_inhomo = spins.B0inhomo.view([self.NVox]) * 2*np.pi
                        delay_ramp = delay * self.setdevice(torch.from_numpy(np.arange(self.NCol))).view([self.NCol,1]) * B0_inhomo.view([1, self.NVox])
                        
                        B0_nspins_cos = torch.cos(delay_ramp)
                        B0_nspins_sin = torch.sin(delay_ramp)
                         
                        S[0,:,:,0,0] = B0_nspins_cos
                        S[0,:,:,0,1] = -B0_nspins_sin
                        S[0,:,:,1,0] = B0_nspins_sin
                        S[0,:,:,1,1] = B0_nspins_cos
                        
                        self.SB0sig = S                        
                        
                    elif t == (self.T - half_read*2)//2 + half_read or self.adc_mask[t+1] == 0:
                        
                        self.set_relaxation_tensor(spins,total_delay)
                        self.set_freeprecession_tensor(spins,total_delay)
                        self.set_B0inhomogeneity_tensor(spins,total_delay)                        
                        
                        #spins.M = RelaxClass.apply(self.R, spins.M,total_delay,t,self,spins)
                        spins.M = RelaxSupermemRAMClass.apply(self.R,spins.M,total_delay,t,r,self,spins)
                        spins.M = DephaseClass.apply(self.P,spins.M,self)
                        
                        # read signal
                        if t == (self.T - half_read*2)//2 + half_read:  # read signal
                            signal = AuxGetSignalGradMul.apply(self.kspace_loc[start_t:start_t+half_read*2,r,:],spins.M[:,:,:,:2,:],self,start_t,start_t+half_read*2)
                                
                            signal = torch.sum(signal,[2])
                            signal *= self.adc_mask[start_t:start_t+half_read*2].view([1,signal.shape[1],1,1])
                            
                            self.signal[0,start_t:start_t+half_read*2,r,:2,0] = signal.squeeze() / self.NSpins 
                            
                        # set grad intravoxel precession tensor
                        kum_grad_intravoxel = torch.sum(self.gradm_event_for_intravoxel_precession[start_t:t+1,r,:],0)
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
                        spins.M[:,:,:,:2,:] = AuxReadoutGradMul.apply(self.kspace_loc[[start_t,t+1],r,:],spins.M[:,:,:,:2,:],self)
                        
                        spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay
                        
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.lastM = spins.M.clone() 
        
        if self.noise_std > 0:
            noise = self.noise_std*torch.randn(self.signal[0,:,:,:2].shape).float()
            noise = self.setdevice(noise)
            self.signal[0,:,:,:2] += noise        
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal) * self.adc_mask.view([self.T,1,1,1])    
            
        torch.cuda.empty_cache()
        
    def forward_sparse_fast_supermem(self,spins,event_time,kill_transverse=False):
        if self.do_voxel_rand_ramp_distr:
            class ExecutionControl(Exception): pass
            raise ExecutionControl('forward_sparse_fast_supermem:  do_voxel_rand_ramp_distr mode not supported')
        
        self.init_signal()
        spins.set_initial_magnetization()
        self.reco = 0
        
        PD0_mask = spins.PD0_mask.flatten().bool()
        spins_cut = spins.M[:,:,PD0_mask,:,:].clone()  
        
        nmb_svox = torch.sum(PD0_mask)
        half_read = np.int(torch.sum(self.adc_mask != 0) / 2)
        
        class AuxGradMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, f, x, scanner):
                ctx.f = f.clone()
                ctx.scanner = scanner
                
                B0X = f[0] * scanner.rampX[:,:,PD0_mask]
                B0Y = f[1] * scanner.rampY[:,:,PD0_mask]
                
                B0_grad = (B0X + B0Y).view([nmb_svox])
                
                B0_grad_cos = torch.cos(B0_grad)
                B0_grad_sin = torch.sin(B0_grad)          
               
                G = scanner.setdevice(torch.zeros((nmb_svox,2,2), dtype=torch.float32))
                G[:,0,0] = B0_grad_cos
                G[:,0,1] = -B0_grad_sin
                G[:,1,0] = B0_grad_sin
                G[:,1,1] = B0_grad_cos    
                
                return torch.matmul(G, x)
            
            @staticmethod
            def backward(ctx, grad_output):
                scanner = ctx.scanner
                
                B0X = ctx.f[0] * scanner.rampX[:,:,PD0_mask]
                B0Y = ctx.f[1] * scanner.rampY[:,:,PD0_mask]
                
                B0_grad = (B0X + B0Y).view([nmb_svox])
                
                B0_grad_cos = torch.cos(B0_grad)
                B0_grad_sin = torch.sin(B0_grad)          
               
                G = scanner.setdevice(torch.zeros((nmb_svox,2,2), dtype=torch.float32))
                G[:,0,0] = B0_grad_cos
                G[:,0,1] = -B0_grad_sin
                G[:,1,0] = B0_grad_sin
                G[:,1,1] = B0_grad_cos
                
                ctx.scanner.lastM[:,:,:,:2,:] = torch.matmul(G.permute([0,2,1]),ctx.scanner.lastM[:,:,:,:2,:])
                gx = torch.matmul(G.permute([0,2,1]),grad_output[:,:,:,:2,:])
                
                gft = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
                gft = torch.sum(gft,[0,1])
                
                GG = scanner.setdevice(torch.zeros((nmb_svox,2,2), dtype=torch.float32))
                GG[:,0,0] = -B0_grad_sin
                GG[:,0,1] = -B0_grad_cos
                GG[:,1,0] = B0_grad_cos
                GG[:,1,1] = -B0_grad_sin     
                
                GG = gft[:,:,:2]*GG
                
                #gff = -torch.sin(gft[:,0,0]) - torch.cos(gft[:,0,1]) + torch.cos(gft[:,1,0] - torch.sin(gft[:,1,1]))
                gf = scanner.setdevice(torch.zeros((2,), dtype=torch.float32))
                
                gf[0] = torch.sum(torch.sum(GG,[1,2]) * scanner.rampX[:,:,PD0_mask].squeeze())
                gf[1] = torch.sum(torch.sum(GG,[1,2])  * scanner.rampY[:,:,PD0_mask].squeeze())
                
                return (gf, gx, None)
            
        class AuxGetSignalGradMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, f, x, scanner, t1, t2):
                ctx.f = f.clone()
                ctx.scanner = scanner
                ctx.t1 = t1
                ctx.t2 = t2
                
                nmb_a = f.shape[0]
                
                # Intra-voxel grad precession
                intraSpins = x
                            
                kum_grad_intravoxel = torch.cumsum(ctx.scanner.gradm_event_for_intravoxel_precession[t1:t2,r,:],0)
                intra_b0 = kum_grad_intravoxel.unsqueeze(0) * ctx.scanner.intravoxel_dephasing_ramp.unsqueeze(1)
                intra_b0 = torch.sum(intra_b0,2)
                
                IVP_nspins_cos = torch.cos(intra_b0)
                IVP_nspins_sin = torch.sin(intra_b0)                            
                
                IVP = torch.zeros((ctx.scanner.NSpins,ctx.scanner.NCol,1,2,2), dtype=torch.float32)
                IVP = ctx.scanner.setdevice(IVP)
                
                IVP[:,:,0,0,0] = IVP_nspins_cos
                IVP[:,:,0,0,1] = -IVP_nspins_sin
                IVP[:,:,0,1,0] = IVP_nspins_sin
                IVP[:,:,0,1,1] = IVP_nspins_cos
                
                tmp = torch.einsum('ijklm,inomp->jolp', [IVP, intraSpins]).unsqueeze(0)
                presignal = torch.einsum('sjorl,ijolp->sjorp', [scanner.SB0sig, tmp])
                
                B0X = torch.unsqueeze(torch.unsqueeze(f[:,0],1),1) * scanner.rampX[:,:,PD0_mask]
                B0Y = torch.unsqueeze(torch.unsqueeze(f[:,1],1),1) * scanner.rampY[:,:,PD0_mask]
                
                B0_grad = (B0X + B0Y).view([nmb_a,nmb_svox])
                
                B0_grad_adj_cos = torch.cos(B0_grad)
                B0_grad_adj_sin = torch.sin(B0_grad)       
                
                G_adj = scanner.setdevice(torch.zeros((nmb_a,nmb_svox,2,2), dtype=torch.float32))
                G_adj[:,:,0,0] = B0_grad_adj_cos
                G_adj[:,:,0,1] = B0_grad_adj_sin
                G_adj[:,:,1,0] = -B0_grad_adj_sin
                G_adj[:,:,1,1] = B0_grad_adj_cos
                
                REW = G_adj[0,:,:,:]
                FWD = G_adj[:,:,:,:].permute([0,1,3,2])
                FWD = torch.matmul(REW, FWD)                
                
                return torch.matmul(FWD, presignal)
            
            @staticmethod
            def backward(ctx, grad_output):
                scanner = ctx.scanner
                
                nmb_a = ctx.f.shape[0]
                
                # Inter-voxel grad precession
                B0X = torch.unsqueeze(torch.unsqueeze(ctx.f[:,0],1),1) * scanner.rampX[:,:,PD0_mask]
                B0Y = torch.unsqueeze(torch.unsqueeze(ctx.f[:,1],1),1) * scanner.rampY[:,:,PD0_mask]
                
                B0_grad = (B0X + B0Y).view([nmb_a,nmb_svox])
                
                B0_grad_adj_cos = torch.cos(B0_grad)
                B0_grad_adj_sin = torch.sin(B0_grad)       
                
                G_adj = scanner.setdevice(torch.zeros((nmb_a,nmb_svox,2,2), dtype=torch.float32))
                G_adj[:,:,0,0] = B0_grad_adj_cos
                G_adj[:,:,0,1] = B0_grad_adj_sin
                G_adj[:,:,1,0] = -B0_grad_adj_sin
                G_adj[:,:,1,1] = B0_grad_adj_cos
                
                REW = G_adj[0,:,:,:]
                FWD = G_adj[:,:,:,:].permute([0,1,3,2])
                FWD = torch.matmul(REW, FWD)  
                
                grad_output = torch.matmul(FWD.permute([0,1,3,2]),grad_output[:,:,:,:2,:])
                #grad_output = torch.repeat_interleave(grad_output,ctx.scanner.NSpins,0)
                
                # Intra-voxel grad precession
                kum_grad_intravoxel = torch.cumsum(ctx.scanner.gradm_event_for_intravoxel_precession[ctx.t1:ctx.t2,r,:],0)
                intra_b0 = kum_grad_intravoxel.unsqueeze(0) * ctx.scanner.intravoxel_dephasing_ramp.unsqueeze(1)
                intra_b0 = torch.sum(intra_b0,2)
                
                IVP_nspins_cos = torch.cos(intra_b0)
                IVP_nspins_sin = torch.sin(intra_b0)
                
                IVP = torch.zeros((ctx.scanner.NSpins,ctx.scanner.NCol,1,2,2), dtype=torch.float32)
                IVP = ctx.scanner.setdevice(IVP)
                
                IVP[:,:,0,0,0] = IVP_nspins_cos
                IVP[:,:,0,0,1] = -IVP_nspins_sin
                IVP[:,:,0,1,0] = IVP_nspins_sin
                IVP[:,:,0,1,1] = IVP_nspins_cos
                
                tmp = torch.einsum('sjolr,ijolp->sjorp', [scanner.SB0sig, grad_output])
                gx = torch.einsum('ijkml,sjomp->isolp', [IVP, tmp])
                
                #gx = torch.einsum('ijkmr,sjolm,njolp->inorp', [IVP,ctx.scanner.SB0sig,grad_output])
                
                
                return (None, gx, None, None, None)
            
        class AuxReadoutGradMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, f, x, scanner):
                ctx.f = f.clone()
                ctx.scanner = scanner
                
                nmb_a = f.shape[0]
                
                B0X = torch.unsqueeze(torch.unsqueeze(ctx.f[:,0],1),1) * scanner.rampX[:,:,PD0_mask]
                B0Y = torch.unsqueeze(torch.unsqueeze(ctx.f[:,1],1),1) * scanner.rampY[:,:,PD0_mask]
                
                B0_grad = (B0X + B0Y).view([nmb_a,nmb_svox])
                
                B0_grad_adj_cos = torch.cos(B0_grad)
                B0_grad_adj_sin = torch.sin(B0_grad)       
                
                G_adj = scanner.setdevice(torch.zeros((nmb_a,nmb_svox,2,2), dtype=torch.float32))
                G_adj[:,:,0,0] = B0_grad_adj_cos
                G_adj[:,:,0,1] = B0_grad_adj_sin
                G_adj[:,:,1,0] = -B0_grad_adj_sin
                G_adj[:,:,1,1] = B0_grad_adj_cos
                
                REW = G_adj[0,:,:,:]
                FWD = G_adj[1,:,:,:].permute([0,2,1])
                FWD = torch.matmul(REW, FWD)
                
                return torch.matmul(FWD,x)
            
            @staticmethod
            def backward(ctx, grad_output):
                scanner = ctx.scanner
                
                nmb_a = ctx.f.shape[0]
                
                B0X = torch.unsqueeze(torch.unsqueeze(ctx.f[:,0],1),1) * scanner.rampX[:,:,PD0_mask]
                B0Y = torch.unsqueeze(torch.unsqueeze(ctx.f[:,1],1),1) * scanner.rampY[:,:,PD0_mask]
                
                B0_grad = (B0X + B0Y).view([nmb_a,nmb_svox])
                
                B0_grad_adj_cos = torch.cos(B0_grad)
                B0_grad_adj_sin = torch.sin(B0_grad)       
                
                G_adj = scanner.setdevice(torch.zeros((nmb_a,nmb_svox,2,2), dtype=torch.float32))
                G_adj[:,:,0,0] = B0_grad_adj_cos
                G_adj[:,:,0,1] = B0_grad_adj_sin
                G_adj[:,:,1,0] = -B0_grad_adj_sin
                G_adj[:,:,1,1] = B0_grad_adj_cos
                
                REW = G_adj[0,:,:,:]
                FWD = G_adj[1,:,:,:].permute([0,2,1])
                FWD = torch.matmul(REW, FWD)
                
                ctx.scanner.lastM[:,:,:,:2,:] = torch.matmul(FWD.permute([0,2,1]),ctx.scanner.lastM[:,:,:,:2,:])
                gx = torch.matmul(FWD.permute([0,2,1]),grad_output[:,:,:,:2,:])
                
                return (None, gx, None)  
            
        class RelaxSupermemRAMClass(torch.autograd.Function):
            @staticmethod
            def forward(ctx, f, x, delay, t, r, scanner, spins):
                ctx.f = f.clone()
                ctx.delay = delay
                ctx.scanner = scanner
                ctx.spins = spins
                ctx.t = t
                ctx.r = r
                ctx.thresh = 1e-2
                
                ctx.M = x.clone().cpu()
                    
                out = torch.matmul(f,x)
                out[:,:,:,2,0] += (1 - f[:,:,2,2]).view([1,1,x.shape[2]]) * spins.MZ0[:,:,PD0_mask]
                    
                return out
        
            @staticmethod
            def backward(ctx, grad_output):
                gx = torch.matmul(ctx.f.permute([0,1,3,2]),grad_output)
              
                gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
                gf[:,:,:,2,2] -= ctx.spins.MZ0[:,:,PD0_mask] * grad_output[:,:,:,2,0]
                gf = torch.sum(gf,[0])
                
                #if ctx.delay > ctx.thresh or (np.mod(ctx.r,16) == 0 and ctx.t == 0):
                #if ctx.delay > ctx.thresh or ctx.t == 0:
                ctx.scanner.lastM = ctx.scanner.setdevice(ctx.M)
                    
                return (gf, gx, None, None, None, None, None)
            
        # scanner forward process loop
        for r in range(self.NRep):                         # for all repetitions
            total_delay = 0
            start_t = 0
            
            for t in range(self.T):                            # for all actions
                delay = torch.abs(event_time[t,r]) 
                
                if self.adc_mask[t] == 0:                         # regular pass
                    if t < self.F.shape[0]:
                        if self.F.shape[2] > 1:
                              spins_cut = FlipClass.apply(self.F[t,r,PD0_mask,:,:],spins_cut,self)
                        else:
                              spins_cut = FlipClass.apply(self.F[t,r,:,:,:],spins_cut,self)
                    
                    self.set_relaxation_tensor(spins,delay)
                    self.set_freeprecession_tensor(spins,delay)
                    self.set_B0inhomogeneity_tensor(spins,delay)
                    
                    spins_cut = RelaxSupermemRAMClass.apply(self.R[:,PD0_mask,:,:],spins_cut,delay,t,r,self,spins)
                    spins_cut = DephaseClass.apply(self.P,spins_cut,self)
                    spins_cut = B0InhomoClass.apply(self.SB0[:,:,PD0_mask,:,:],spins_cut,self)
                    
                    spins_cut[:,:,:,:2,:] = AuxGradMul.apply(self.grads[t,r,:],spins_cut[:,:,:,:2,:],self)

                    self.set_grad_intravoxel_precess_tensor(t,r)
                    spins_cut = GradIntravoxelPrecessClass.apply(self.IVP,spins_cut,self)
                else:
                    if self.adc_mask[t-1] == 0:        # first sample in readout
                        total_delay = delay
                        start_t = t
                        
                        # set B0 inhomo precession tensor
                        B0_inhomo = spins.B0inhomo.view([self.NVox]) * 2*np.pi
                        B0_inhomo = B0_inhomo[PD0_mask]
                        
                        S = torch.zeros((1,self.NCol,nmb_svox,2,2), dtype=torch.float32)
                        S = self.setdevice(S)                        
                        
                        delay_ramp = delay * self.setdevice(torch.from_numpy(np.arange(self.NCol))).view([self.NCol,1]) * B0_inhomo.view([1, nmb_svox])
                        
                        B0_nspins_cos = torch.cos(delay_ramp)
                        B0_nspins_sin = torch.sin(delay_ramp)
                         
                        S[0,:,:,0,0] = B0_nspins_cos
                        S[0,:,:,0,1] = -B0_nspins_sin
                        S[0,:,:,1,0] = B0_nspins_sin
                        S[0,:,:,1,1] = B0_nspins_cos
                        
                        self.SB0sig = S
                        
                    elif t == (self.T - half_read*2)//2 + half_read or self.adc_mask[t+1] == 0:
                        self.set_relaxation_tensor(spins,total_delay)
                        self.set_freeprecession_tensor(spins,total_delay)
                        self.set_B0inhomogeneity_tensor(spins,total_delay)
                        
                        spins_cut = RelaxSupermemRAMClass.apply(self.R[:,PD0_mask,:,:],spins_cut,total_delay,t,r,self,spins)
                        spins_cut = DephaseClass.apply(self.P,spins_cut,self)
                        
                        
                        # read signal
                        if t == (self.T - half_read*2)//2 + half_read:  # read signal
                            signal = AuxGetSignalGradMul.apply(self.kspace_loc[start_t:start_t+half_read*2,r,:],spins_cut[:,:,:,:2,:],self,start_t,start_t+half_read*2)
                                
                            signal = torch.sum(signal,[2])
                            signal *= self.adc_mask[start_t:start_t+half_read*2].view([1,signal.shape[1],1,1])
                            
                            self.signal[0,start_t:start_t+half_read*2,r,:2,0] = signal.squeeze() / self.NSpins 
                            
                        # do gradient precession (use adjoint as free kumulator)
                        # set grad intravoxel precession tensor
                        kum_grad_intravoxel = torch.sum(self.gradm_event_for_intravoxel_precession[start_t:t+1,r,:],0)
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
                        spins_cut[:,:,:,:2,:] = AuxReadoutGradMul.apply(self.kspace_loc[[start_t,t+1],r,:],spins_cut[:,:,:,:2,:],self)
                        
                        spins_cut = B0InhomoClass.apply(self.SB0[:,:,PD0_mask,:,:],spins_cut,self)
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay
                        
            if kill_transverse:
                spins.M[:,0,:,:2,0] = 0

                    
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.tmask = self.tmask[PD0_mask]
        self.lastM = spins_cut.clone() 
        
        if self.noise_std > 0:
            noise = self.noise_std*torch.randn(self.signal[0,:,:,:2].shape).float()
            noise = self.setdevice(noise)
            self.signal[0,:,:,:2] += noise        
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal) * self.adc_mask.view([self.T,1,1,1])
            
        torch.cuda.empty_cache()        
            
    def do_dummy_scans(self,spins,event_time,compact_grad_tensor=True,nrep=0):
        class ExecutionControl(Exception): pass
        raise ExecutionControl('do_dummy_scans:  currently broken')
        
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
                    delay = torch.abs(event_time[t,r]) 
                    
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
                            kum_grad_intravoxel = torch.sum(self.gradm_event_for_intravoxel_precession[start_t:t+1,r,:],0)
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
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.lastM = spins.M.clone() 
            
    def do_dummy_scans_sparse(self,spins,event_time,compact_grad_tensor=True,nrep=0):
        class ExecutionControl(Exception): pass
        raise ExecutionControl('do_dummy_scans:  currently broken')        
        
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
                    delay = torch.abs(event_time[t,r]) 
                    
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
                            kum_grad_intravoxel = torch.sum(self.gradm_event_for_intravoxel_precession[start_t:t+1,r,:],0)
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
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.tmask = self.tmask[PD0_mask]
        self.lastM = spins_cut.clone()
                


    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,rep):
        sig = self.signal[:,:,rep,:,:]
        nrm = np.sqrt(np.prod(self.sz)) 
        
        if self.NCoils == 1:                                  # backward compat
            # for now we ignore parallel imaging options here (do naive sum sig over coil)
            sig = torch.sum(sig, 0)        
    
            r = torch.matmul(self.G_adj.permute([1,0,2,3]), sig)  / nrm
            self.reco = self.reco + torch.sum(r[:,:,:2,0],1)
        else:
            reco = torch.matmul(self.G_adj.permute([1,0,2,3]).unsqueeze(1), sig)  / nrm
            self.reco = self.reco + torch.sum(reco[:,:,:,:2,0],2).permute([1,0,2])
            
    def adjoint(self):
        if self.NCoils == 1:
            self.init_reco()
        else:
            reco = torch.zeros((self.NCoils,self.NVox,2), dtype = torch.float32)
            self.reco = self.setdevice(reco)              
        
        #adc_mask = self.adc_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #self.signal *= adc_mask        
        
        for rep in range(self.NRep):
            self.set_grad_adj_op(rep)
            self.do_grad_adj_reco(rep)
            
        # transpose for adjoint
        if self.NCoils > 1:                                             # do SOS
            self.reco = torch.sqrt(torch.sum(self.reco[:,:,0]**2 + self.reco[:,:,1]**2,0))
            self.reco = self.reco.reshape([self.sz[0],self.sz[1]]).flip([0,1]).permute([1,0]).reshape([self.NVox]).unsqueeze(1).repeat([1,2])
            self.reco[:,1] = 0
        else:
            self.reco = self.reco.reshape([self.sz[0],self.sz[1],2]).flip([0,1]).permute([1,0,2]).reshape([self.NVox,2])
        
    def adjoint_supermem(self):
        self.init_reco()
        
        class AuxGradMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, f, x, scanner):
                ctx.f = f.clone()
                ctx.scanner = scanner
                
                B0X = torch.unsqueeze(torch.unsqueeze(f[:,0],1),1) * scanner.rampX
                B0Y = torch.unsqueeze(torch.unsqueeze(f[:,1],1),1) * scanner.rampY
                
                B0_grad = (B0X + B0Y).view([scanner.T,scanner.NVox])
                
                B0_grad_adj_cos = torch.cos(B0_grad)
                B0_grad_adj_sin = torch.sin(B0_grad)       
                
                G_adj = scanner.setdevice(torch.zeros((scanner.T,scanner.NVox,3,3), dtype=torch.float32))
                G_adj[:,:,2,2] = 1
                G_adj[:,:,0,0] = B0_grad_adj_cos
                G_adj[:,:,0,1] = B0_grad_adj_sin
                G_adj[:,:,1,0] = -B0_grad_adj_sin
                G_adj[:,:,1,1] = B0_grad_adj_cos                
                
                return torch.matmul(G_adj.permute([1,0,2,3]), x)
            
            @staticmethod
            def backward(ctx, grad_output):
                
                scanner = ctx.scanner
                
                B0X = torch.unsqueeze(torch.unsqueeze(ctx.f[:,0],1),1) * scanner.rampX
                B0Y = torch.unsqueeze(torch.unsqueeze(ctx.f[:,1],1),1) * scanner.rampY
                
                B0_grad = (B0X + B0Y).view([scanner.T,scanner.NVox])
                
                B0_grad_adj_cos = torch.cos(B0_grad)
                B0_grad_adj_sin = torch.sin(B0_grad)       
                
                G_adj = scanner.setdevice(torch.zeros((scanner.T,scanner.NVox,3,3), dtype=torch.float32))
                G_adj[:,:,2,2] = 1
                G_adj[:,:,0,0] = B0_grad_adj_cos
                G_adj[:,:,0,1] = B0_grad_adj_sin
                G_adj[:,:,1,0] = -B0_grad_adj_sin
                G_adj[:,:,1,1] = B0_grad_adj_cos     
                
                gx = torch.matmul(G_adj.permute([1,0,3,2]),grad_output)
                
                return (None, gx, None)
        
        nrm = np.sqrt(np.prod(self.sz))
        
        for rep in range(self.NRep):
            s = self.signal[:,:,rep,:,:]
            s = torch.sum(s, 0)                                                  
            r = AuxGradMul.apply(self.kspace_loc[:,rep,:],s,self) / nrm
            
            self.reco = self.reco + torch.sum(r[:,:,:2,0],1)            
            
        # transpose for adjoint
        self.reco = self.reco.reshape([self.sz[0],self.sz[1],2]).flip([0,1]).permute([1,0,2]).reshape([self.NVox,2])
        
    # reconstruct image readout by readout            
    def do_grad_adj_reco_separable(self,r):
        s = self.signal[:,:,r,:,:]
         
        # for now we ignore parallel imaging options here (do naive sum sig over coil)
        s = torch.sum(s, 0)                                                  
        r = torch.matmul(self.G_adj.permute([1,0,2,3]), s)
        return torch.sum(r[:,:,:2,0],1)
        
    def adjoint_separable(self):
        self.init_reco()
        nrm = np.sqrt(np.prod(self.sz))
        
        r = self.setdevice(torch.zeros((self.NRep,self.NVox,2)).float())
        for rep in range(self.NRep):
            self.set_grad_adj_op(rep)
            r[rep,:,:] = self.do_grad_adj_reco_separable(rep)
            
        r = r.reshape([self.NRep,self.sz[0],self.sz[1],2]).flip([1,2]).permute([0,2,1,3]).reshape([self.NRep,self.NVox,2])
        r /=  nrm
        
        self.reco = torch.sum(r,0)
        return r 

    def get_base_path(self, experiment_id, today_datestr):

        if os.path.isfile(os.path.join('core','pathfile_local.txt')):
            pathfile ='pathfile_local.txt'
        else:
            pathfile ='pathfile.txt'
            print('You dont have a local pathfile in core/pathfile_local.txt, so we use standard file: pathfile.txt')

        with open(os.path.join('core',pathfile),"r") as f:
            path_from_file = f.readline()
        basepath = path_from_file
        basepath_seq = os.path.join(basepath, 'sequences')
        basepath_seq = os.path.join(basepath_seq, "seq" + today_datestr)
        basepath_seq = os.path.join(basepath_seq, experiment_id)

        return basepath, basepath_seq

    # interaction with real system
    def send_job_to_real_system(self, experiment_id, today_datestr, basepath_seq_override=None, jobtype="target", iterfile=None):
        basepath, basepath_seq = self.get_base_path(experiment_id, today_datestr)
        basepath_control  = os.path.join(basepath, 'control')
        
        if basepath_seq_override is not None:
            basepath_seq = basepath_seq_override
        
        if jobtype == "target":
            fn_pulseq = "target.seq"
        elif jobtype == "lastiter":
            fn_pulseq = "lastiter.seq"
        elif jobtype == "iter":
            fn_pulseq = iterfile + ".seq"
            
        fnpath = os.path.join(basepath_seq, fn_pulseq + ".dat")
        if os.path.isfile(fnpath):            
            os.remove(fnpath)
            
        #if os.path.isfile(os.path.join(basepath_seq, "data", fn_twix)):
        #    print('TWIX file already exists. Not sending job to scanner... ' + fn_twix)
        #    return            
        
        control_filename = "control.txt"
        position_filename = "position.txt"
        lock_filename = "lock"
        
        if os.path.isfile(os.path.join(basepath_seq, fn_pulseq)):
            pass
        else:
            class ExecutionControl(Exception): pass
            raise ExecutionControl('sequence file missing ' + os.path.join(basepath_seq, fn_pulseq))
            
        ready_flag = False
        fp_lock = os.path.join(basepath_control, lock_filename)
        while not ready_flag:
            if not os.path.isfile(fp_lock):
                open(fp_lock, 'a').close()
                ready_flag = True            
            
        with open(os.path.join(basepath_control,position_filename),"r") as f:
            position = int(f.read())
            
        with open(os.path.join(basepath_control,control_filename),"r") as f:
            control_lines = f.readlines()
            
        control_lines = [l.strip() for l in control_lines]
        
        if position >= len(control_lines) or len(control_lines) == 0 or control_lines.count('wait') > 1 or control_lines.count('quit') > 1:
            class ExecutionControl(Exception): pass
            raise ExecutionControl("control file is corrupt")
        
        basepath_out  = os.path.join(basepath_seq, "seq" + today_datestr,experiment_id)
        basepath_out = basepath_seq.replace('\\','//')
    
        # add sequence file
        if control_lines[-1] == 'wait':
            control_lines[-1] = basepath_out + "//" + fn_pulseq
            control_lines.append('wait')
            
        control_lines = [l+"\n" for l in control_lines]
        
        with open(os.path.join(basepath_control,control_filename),"w") as f:
            f.writelines(control_lines)
            
        os.remove(fp_lock)
            
    def get_signal_from_real_system(self, experiment_id, today_datestr, basepath_seq_override=None, jobtype="target", iterfile=None,single_folder=True):
        _, basepath_seq = self.get_base_path(experiment_id, today_datestr)
        
        if basepath_seq_override is not None:
            basepath_seq = basepath_seq_override        
            
        print('wating for TWIX file from the scanner...')
        
        if jobtype == "target":
            if single_folder:
                basepath_seq=os.path.dirname(basepath_seq)
                fn_twix = experiment_id+".seq"
            else:
                fn_twix = "target.seq"
        elif jobtype == "lastiter":
            fn_twix = "lastiter.seq"   
        elif jobtype == "iter":
            fn_twix = iterfile + ".seq"
        
        # if load from data folder
#        if jobtype == "target":
#            fn_twix = "target"
#        elif jobtype == "lastiter":
#            fn_twix = "lastiter"   
#        elif jobtype == "iter":
#            fn_twix = iterfile + ""
#        fn_twix = os.path.join("data", fn_twix)
        #end
            
        fn_twix += ".dat"
            
        print('waiting for TWIX file from the scanner... ' + fn_twix)
        
        # go into the infinite loop, checking if twix file is saved
        done_flag = False
        while not done_flag:
            fnpath = os.path.join(basepath_seq, fn_twix)
        
            if os.path.isfile(fnpath):
                # read twix file
                print("TWIX file arrived. Reading....")
                
                raw_file = os.path.join(basepath_seq, fn_twix)
                ncoils = 20
                
                time.sleep(0.2)
                
                raw = np.loadtxt(raw_file)

                
                dp_twix = os.path.dirname(fnpath)
#                shutil.move(fnpath, os.path.join(dp_twix,"data",fn_twix))
                
                heuristic_shift = 4
                print("raw size: {} ".format(raw.size) + "expected size: {} ".format("raw size: {} ".format(self.NRep*ncoils*(self.NCol+heuristic_shift)*2)) )
#                import pdb; pdb.set_trace();
                                
                if raw.size != self.NRep*ncoils*(self.NCol+heuristic_shift)*2:
                      print("get_signal_from_real_system: SERIOUS ERROR, TWIX dimensions corrupt, returning zero array..")
                      raw = np.zeros((self.NRep,ncoils,self.NCol+heuristic_shift,2))
                      raw = raw[:,:,:(self.NCol-heuristic_shift),0] + 1j*raw[:,:,:(self.NCol-heuristic_shift),1]
                else:
                      raw = raw.reshape([self.NRep,ncoils,self.NCol+heuristic_shift,2])                    
                      raw = raw[:,:,:(self.NCol),0] + 1j*raw[:,:,:(self.NCol),1]
#                      raw /= np.max(np.abs(raw))
#                      raw /= 0.05*0.014/9
                
                # inject into simulated scanner signal variable
                adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
                
                raw = raw.transpose([2,1,0])
                raw = np.copy(raw)
                #raw = np.roll(raw,1,axis=0)
            
                
                # assume for now a single coil
                
                if self.NCoils==ncoils:  # correct multi coil?
                    all_coil_reco = torch.zeros((ncoils,self.NVox,2), dtype = torch.float32)
                    for coil_idx in range(ncoils):
                        self.signal[coil_idx,adc_idx,:,0,0] = self.setdevice(torch.from_numpy(np.real(raw[:,coil_idx,:])))
                        self.signal[coil_idx,adc_idx,:,1,0] = self.setdevice(torch.from_numpy(np.imag(raw[:,coil_idx,:])))
                elif ncoils>2:  # pseudo mutli coil SOS
                    all_coil_reco = torch.zeros((ncoils,self.NVox,2), dtype = torch.float32)
                    for coil_idx in range(ncoils):
                        self.signal[0,adc_idx,:,0,0] = self.setdevice(torch.from_numpy(np.real(raw[:,coil_idx,:])))
                        self.signal[0,adc_idx,:,1,0] = self.setdevice(torch.from_numpy(np.imag(raw[:,coil_idx,:])))
                        self.adjoint()
                        all_coil_reco[coil_idx,:] = self.reco                        
                    self.init_reco()    
                    self.reco[:,0] = torch.sum(all_coil_reco[:,:,0]**2+all_coil_reco[:,:,1]**2, 0)
                else:
                    coil_idx=0
                    self.signal[0,adc_idx,:,0,0] = self.setdevice(torch.from_numpy(np.real(raw[:,coil_idx,:])))
                    self.signal[0,adc_idx,:,1,0] = self.setdevice(torch.from_numpy(np.imag(raw[:,coil_idx,:])))
                    
                
                done_flag = True
                
                time.sleep(0.5)
                
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
        if not self.do_voxel_rand_ramp_distr:
            IVP = torch.zeros((self.NSpins,1,1,3,3), dtype=torch.float32)
            IVP = self.setdevice(IVP)
            
            IVP[:,0,0,2,2] = 1
            
            self.IVP = IVP
        else:
            IVP = torch.zeros((self.NSpins,1,self.NVox,3,3), dtype=torch.float32)
            IVP = self.setdevice(IVP)
            IVP[:,0,:,2,2] = 1
            self.IVP = IVP  
        
    def set_gradient_precession_tensor(self,gradm_event,sequence_class):
        # we need to shift gradm_event to the right for adjoint pass, since at each repetition we have:
        # meas-signal, flip, relax,grad order  (signal comes before grads!)
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,gradm_event),0)
        temp = temp[:-1,:,:]
        k=torch.cumsum(temp,0)
        
        B0X = torch.unsqueeze(gradm_event[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(gradm_event[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        B0_grad_cos = torch.cos(B0_grad)
        B0_grad_sin = torch.sin(B0_grad)
        
        if sequence_class.lower() == "gre" or sequence_class.lower() == "bssfp":
            pass
        
        if sequence_class.lower() == "gre_dream" :
            temp[0:4,:,:]=0   # 3 is excitation pulse event ( still 0) 4 revinder is the first that counts
            k=torch.cumsum(temp,0)
        
        if sequence_class.lower() == "rare":
            refocusing_pulse_action_idx = 3
            kloc = 0
            for r in range(self.NRep):
                for t in range(self.T):
                    if refocusing_pulse_action_idx+1 == t:     # +1 because of right gradmom shift (see above )
                        kloc = -kloc
                    kloc += temp[t,r,:]
                    k[t,r,:] = kloc
                    
        if sequence_class.lower() == "se":
            refocusing_pulse_action_idx = 3
            for r in range(self.NRep):
                kloc = 0
                for t in range(self.T):
                    if refocusing_pulse_action_idx+1 == t:     # +1 because of right gradmom shift (see above )
                        kloc = -kloc
                    kloc += temp[t,r,:]
                    k[t,r,:] = kloc                    
                    
        if sequence_class.lower() == "epi":    
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
        
        # save gradm_event for intravoxel precession op
        self.gradm_event_for_intravoxel_precession = gradm_event
        self.grads = gradm_event
        
        self.kspace_loc = k
        
    def grad_precess(self,t,r,spins):
        spins.M = torch.matmul(self.G[t,r,:,:,:],spins.M)
        
    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,t):
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
        
    def forward_fast(self,spins,event_time,kill_transverse=False,do_dummy_scans=False):
        super().forward_fast(spins,event_time,do_dummy_scans,kill_transverse=kill_transverse,compact_grad_tensor=False)
        
    def forward_fast_supermem(self,spins,event_time,do_dummy_scans=False):
        super().forward_fast_supermem(spins,event_time)        
        
    def forward_sparse_fast(self,spins,event_time,do_dummy_scans=False,kill_transverse=False):
        super().forward_sparse_fast(spins,event_time,do_dummy_scans,kill_transverse=kill_transverse,compact_grad_tensor=False)
        
    def forward_sparse_fast_supermem(self,spins,event_time,do_dummy_scans=False,kill_transverse=False):
        super().forward_sparse_fast_supermem(spins,event_time,kill_transverse=kill_transverse)
        
    def do_dummy_scans(self,spins,event_time,nrep=0):
        super().do_dummy_scans(spins,event_time,compact_grad_tensor=False,nrep=nrep)

    def do_dummy_scans_sparse(self,spins,event_time,nrep=0):
        super().do_dummy_scans_sparse(spins,event_time,compact_grad_tensor=False,nrep=nrep)
        
    # compute adjoint encoding op-based reco    <            
    def adjoint(self):
        self.init_reco()
        
        #s = torch.sum(self.signal,0)
        #r = torch.matmul(self.G_adj.permute([2,3,0,1,4]).contiguous().view([self.NVox,3,self.T*self.NRep*3]), s.view([1,self.T*self.NRep*3,1]))
        #import pdb; pdb.set_trace()
        
        nrm = np.sqrt(np.prod(self.sz))
        r = torch.einsum("ijkln,oijnp->klp",[self.G_adj, self.signal])
        self.reco = r[:,:2,0] / nrm
        
        # transpose for adjoint
        self.reco = self.reco.reshape([self.sz[0],self.sz[1],2]).flip([0,1]).permute([1,0,2]).reshape([self.NVox,2])
        
    def generalized_adjoint(self, alpha=1e-4, nmb_iter=55):
        self.init_reco()
        nrm = np.sqrt(np.prod(self.sz))
        
        s = torch.sum(self.signal,0)
        adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
        s = s[adc_idx,:,:2]
        s = s.view([adc_idx.size*self.NRep*2,1])
        
        A = self.G_adj[adc_idx,:,:,:2,:2].permute([2,3,0,1,4]).contiguous().view([self.NVox*2,adc_idx.size*self.NRep*2]).permute([1,0])
        A /= nrm
        AtA = torch.matmul(A.permute([1,0]),A)
        
        t = AtA @ self.setdevice(torch.ones((AtA.shape[1],)))
        genalpha = 1e-2 * 192 / torch.mean(t)  
#        alpha = genalpha
        
        b = torch.matmul(A.permute([1,0]),s)
        r = alpha*b
        
        for i in range(nmb_iter):
            r = r - alpha*(torch.matmul(AtA,r) - b)
        
        r = r.view([self.NVox,2,1]) / alpha
        self.reco = r[:,:,0]
        
        # transpose for adjoint
        self.reco = self.reco.reshape([self.sz[0],self.sz[1],2]).flip([0,1]).permute([1,0,2]).reshape([self.NVox,2])
        
    def inverse(self):
        self.init_reco()
        
        s = torch.sum(self.signal,0)
        adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
        s = s[adc_idx,:,:2]
        s = s.view([adc_idx.size*self.NRep*2,1])
        
        A = self.G_adj[adc_idx,:,:,:2,:2].permute([2,3,0,1,4]).contiguous().view([self.NVox*2,adc_idx.size*self.NRep*2]).permute([1,0])
        r = torch.matmul(torch.inverse(A), s)
        
        r = r.view([self.NVox,2,1])
        self.reco = r[:,:,0]
        
        # transpose for adjoint
        self.reco = self.reco.reshape([self.sz[0],self.sz[1],2]).flip([0,1]).permute([1,0,2]).reshape([self.NVox,2])          
        
    def cholesky_inverse(self):
        self.init_reco()
        
        s = torch.sum(self.signal,0)
        adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
        s = s[adc_idx,:,:2]
        s = s.view([adc_idx.size*self.NRep*2,1])
        
        A = self.G_adj[adc_idx,:,:,:2,:2].permute([2,3,0,1,4]).contiguous().view([self.NVox*2,adc_idx.size*self.NRep*2]).permute([1,0])
        AtA = torch.mm(A, A.t()) + 1e-05 * self.setdevice(torch.eye(A.shape[0]))
        b = torch.matmul(A.permute([1,0]),s)
        u = torch.cholesky(AtA)
        AtAinv = torch.cholesky_inverse(u)
        
        r = torch.matmul(AtAinv, b)
        r = torch.matmul(torch.inverse(A), s)
        
        r = r.view([self.NVox,2,1])
        self.reco = r[:,:,0]
        
        # transpose for adjoint
        self.reco = self.reco.reshape([self.sz[0],self.sz[1],2]).flip([0,1]).permute([1,0,2]).reshape([self.NVox,2])        
        
    def adjoint_separable(self):
        nrm = np.sqrt(np.prod(self.sz))
        self.init_reco()
        
        s = torch.sum(self.signal,0)
        s = s.permute([1,0,2,3]).contiguous().view([self.NRep,1,self.T*3,1])
        g = self.G_adj.permute([1,2,3,0,4]).contiguous().view([self.NRep,self.NVox,3,self.T*3])
        
        r = torch.matmul(g, s)
        r = r[:,:,:2,0]
        r = r.reshape([self.NRep,self.sz[0],self.sz[1],2]).flip([1,2]).permute([0,2,1,3]).reshape([self.NRep,self.NVox,2])
        
        self.reco = torch.sum(r,0) / nrm
        return r
        
        

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
        #gf = torch.sum(gf,[0,2])
        gf = torch.sum(gf,[0,1]).reshape(ctx.f.shape)
        
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
        
        if ctx.delay > ctx.thresh or ctx.t == 0:
            ctx.M = x.clone()
            
        out = torch.matmul(f,x)
        out[:,:,:,2,0] += (1 - f[:,:,2,2]).view([1,1,scanner.NVox]) * spins.MZ0
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,1,3,2]),grad_output)
      
        gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
        gf[:,:,:,2,2] -= ctx.spins.MZ0 * grad_output[:,:,:,2,0]
        gf = torch.sum(gf,[0])
        
        if ctx.delay > ctx.thresh or ctx.t == 0:
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
    
class RelaxRAMClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, delay, t, scanner, spins):
        ctx.f = f.clone()
        ctx.delay = delay
        ctx.scanner = scanner
        ctx.spins = spins
        ctx.t = t
        
        ctx.M = x.clone().cpu()
                    
        out = torch.matmul(f,x)
        out[:,:,:,2,0] += (1 - f[:,:,2,2]).view([1,1,scanner.NVox]) * spins.MZ0
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,1,3,2]),grad_output)
        
        gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
        gf[:,:,:,2,2] -= ctx.spins.MZ0 * grad_output[:,:,:,2,0]
        gf = torch.sum(gf,[0])
        
        ctx.scanner.lastM = ctx.scanner.setdevice(ctx.M)
            
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
        
#        if ctx.delay > ctx.thresh or ctx.t == 0:
#            ctx.M = x.clone()
        
        ctx.M = x.clone().cpu()
            
        out = torch.matmul(f,x)
        out[:,:,:,2,0] += (1 - f[:,:,2,2]).view([1,1,x.shape[2]]) * spins.MZ0[:,:,mask]
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.matmul(ctx.f.permute([0,1,3,2]),grad_output)
      
        gf = ctx.scanner.lastM.permute([0,1,2,4,3]) * grad_output
        gf[:,:,:,2,2] -= ctx.spins.MZ0[:,:,ctx.mask] * grad_output[:,:,:,2,0]
        gf = torch.sum(gf,[0])
        
        ctx.scanner.lastM = ctx.scanner.setdevice(ctx.M)
        
#        if ctx.delay > ctx.thresh or ctx.t == 0:
#            ctx.scanner.lastM = ctx.M
#        else:
#            d1 = ctx.f[0,:,0,0]
#            id1 = 1/d1
#            
#            d3 = ctx.f[0,:,2,2]
#            id3 = 1/d3
#            id3 = id3.view([1,grad_output.shape[2]])
#            
#            ctx.scanner.lastM[:,0,:,:2,0] *= id1.view([1,grad_output.shape[2],1])
#            ctx.scanner.lastM[:,0,:,2,0] = ctx.scanner.lastM[:,0,:,2,0]*id3 + (1-id3)*ctx.spins.MZ0[:,0,ctx.mask]
#            
#            ctx.scanner.lastM[:,:,ctx.scanner.tmask,:] = 0
            
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