import numpy as np
import torch
import time
import os
import sys
from sys import platform
import scipy
import shutil
import socket
from auxutil.pseudoinverse import pinv_reg
import auxutil.cg_batch

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
    
def fftshift(x,axes):
    '''
    fftshift along arbitrary (multiple) axes of pytorch tensor x
    axes: tuple of axes along which fftshift is applied
    '''
    if type(axes) is not tuple:
        axes = (axes,)
    dims = x.shape
    for ax in axes:
        n = x.shape[ax] // 2 + 0 # very crucial point here, +-1 gives position shift in abs and additional phase wraps!
        x = x.permute((ax,) + tuple(range(0,ax)) + tuple(range(ax+1, len(dims))))
        x = torch.cat((x[-n:,:], x[:-n,:]))
        x = x.permute(tuple(range(1,ax+1)) + (0,) + tuple(range(ax+1, len(dims))))
    return x

def tonumpy(x):
    return x.detach().cpu().numpy()

# HOW we measure
class Scanner():
    def __init__(self,sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu,double_precision=False,coilCombineMode='sos',accel=1):
        
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
        self.coilCombineMode = coilCombineMode # how to combine coil signals (Ncoils > 1), sos
        self.coilWeights = None
        
        self.accel = accel # parallel imaging acceleration factor (R) in phase direction (wip)
        
        if accel >= NCoils and NCoils>1:
            print("Warning: acceleration factor larger than number of coils is most likely not a good idea!")

        self.signal = None                # measured signal (NCoils,T,NRep,3)
        self.reco =  None                       # reconstructed image (NVox,) 
        
        self.ROI_signal = None            # measured signal (NCoils,T,NRep,3)
        self.ROI_def = 1
        self.lastM = None
        self.AF = None
        self.use_gpu =  use_gpu
        self.double_precision = double_precision
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
    
    def do_SAR_test(self, flips, event_time):
        TACQ = torch.sum(event_time)*1000 + 2e3
        watchdog_norm = 100 / 0.45245
        SAR_watchdog = (torch.sum(flips[:,:,0]**2) / TACQ).cpu()
        print("SAR_watchdog = {}%".format(np.round(SAR_watchdog*watchdog_norm)))
        
    def set_B1plus(self,input_array):
        B1plus = torch.zeros((self.NCoils,1,self.NVox,1,1), dtype=torch.float32)
        if np.ndim(input_array) > 1:
            B1plus[:,0,:,0,0] = torch.from_numpy(input_array.reshape([self.NCoils, self.NVox]))
            B1plus[B1plus == 0] = 1    # set b1+ to one, where we dont have phantom measurements
        else:
            B1plus[:] = input_array
        self.B1plus = self.setdevice(B1plus)
        
        
    def set_adc_mask(self, adc_mask = None):
        if adc_mask is None:
            adc_mask = torch.from_numpy(np.ones((self.T,1))).float()
            self.adc_mask = self.setdevice(adc_mask)
        else:
            self.adc_mask = self.setdevice(adc_mask)
        
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
        if dim == torch.floor(dim):
            
            xv, yv = torch.meshgrid([torch.linspace(-1+off,1-off,dim.int()), torch.linspace(-1+off,1-off,dim.int())])
            # this generates an anti-symmetric distribution in x
            Rx1= torch.randn(torch.Size([dim.int()//2,dim.int()]))*off
            Rx2=-torch.flip(Rx1, [0])
            Rx= torch.cat((Rx1, Rx2),0)
            # this generates an anti-symmetric distribution in y
            Ry1= torch.randn(torch.Size([dim.int(),dim.int()//2]))*off
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
            
        self.intravoxel_dephasing_ramp = self.setdevice(intravoxel_dephasing_ramp)    
        
    # function is obsolete: deprecate in future, this dummy is for backward compat
    def get_ramps(self):
        pass
    
    
    def get_phase_cycler(self, n, dphi):
        out = np.cumsum(np.arange(n) * dphi)
        out = torch.from_numpy(np.mod(out, 360).astype(np.float32))
        
        return out    
        
    def init_coil_sensitivities(self, B1=None):
        # input B1: Ncoils x Nx x Ny
        # handle complex mul as matrix mul
        B1_init = torch.zeros((self.NCoils,1,self.NVox,2,2), dtype=torch.float32)
        
        if B1 is None:
            B1 = torch.ones((self.NCoils,self.sz[0],self.sz[1],2))     # last dimension real/imag for B1 minus
            B1[:,:,:,1] = 0 # imag=0
        
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
         
    def set_flip_tensor(self,flips):
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        self.F[:,:,0,0,0] = flips_cos
        self.F[:,:,0,0,2] = flips_sin
        self.F[:,:,0,2,0] = -flips_sin
        self.F[:,:,0,2,2] = flips_cos 
        
    def set_flipXY_tensor(self,input_flips):
        
        class ExecutionControl(Exception): pass
        raise ExecutionControl('set_flipXY_tensor method is deprecated! use set_flip_tensor_withB1plus instead')
        
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
        
        # WARNING TODO, not future proof
#        if self.sz[0] > 64:
#            theta = theta[:(self.T - self.NCol)//2,:,:,:,:]
#            Fglob = Fglob[:(self.T - self.NCol)//2,:,:,:,:]
#            F2 = F2[:(self.T - self.NCol)//2,:,:,:,:]        
        
        theta = theta * self.B1plus.view([1,1,self.NVox,1,1])
        
        F = torch.sin(theta)*Fglob + (1 - torch.cos(theta))*F2
        self.F = self.setdevice(F)       
    
        self.F[:,:,:,0,0] += 1
        self.F[:,:,:,1,1] += 1
        self.F[:,:,:,2,2] += 1
        
    def set_flip_tensor_withB1plus_and_theta(self,input_flips):
        if not hasattr(self,'B1plus'):
            class ExecutionControl(Exception): pass
            raise ExecutionControl('set_flip_tensor_withB1plus_and_offres: set B1plus before use')        # flip operator with B1plus inhomogeneity
    
        Fglob = torch.zeros((self.T,self.NRep,1,3,3), dtype=torch.float32)
        
        Fglob[:,:,:,1,1] = 1
         
        Fglob = self.setdevice(Fglob)
        
        vx = torch.cos(input_flips[:,:,1])*torch.cos(input_flips[:,:,2])
        vy = torch.sin(input_flips[:,:,1])*torch.cos(input_flips[:,:,2])
        vz = torch.sin(input_flips[:,:,2]) #angle between xy plane and Beff = pi/2 - atan(B1/delta B0), 0 deg for on-resonant pulse
        
        Fglob[:,:,0,0,0] = 0
        Fglob[:,:,0,0,1] = -vz
        Fglob[:,:,0,0,2] = vy
        Fglob[:,:,0,1,0] = vz
        Fglob[:,:,0,1,1] = 0
        Fglob[:,:,0,1,2] = -vx
        Fglob[:,:,0,2,0] = -vy
        Fglob[:,:,0,2,1] = vx
        Fglob[:,:,0,2,2] = 0
    
        # matrix square
        F2 = torch.matmul(Fglob,Fglob)
        
        # theta = angle around Beff = FA/sin(atan(B1/delta B0))
        theta = input_flips[:,:,0]/torch.sin(np.pi/2-input_flips[:,:,2]) 
        theta = theta.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        
        # WARNING TODO, not future proof
#        if self.sz[0] > 64:
#            theta = theta[:(self.T - self.NCol)//2,:,:,:,:]
#            Fglob = Fglob[:(self.T - self.NCol)//2,:,:,:,:]
#            F2 = F2[:(self.T - self.NCol)//2,:,:,:,:]        
        
        theta = theta * self.B1plus.view([1,1,self.NVox,1,1])
        
        F = torch.sin(theta)*Fglob + (1 - torch.cos(theta))*F2
        self.F = self.setdevice(F)       
    
        self.F[:,:,:,0,0] += 1
        self.F[:,:,:,1,1] += 1
        self.F[:,:,:,2,2] += 1    
               
    def set_flip_tensor_withB1plus_and_offres(self,RF_event, B0inhomo,t_pulse):
        if not hasattr(self,'B1plus'):
            class ExecutionControl(Exception): pass
            raise ExecutionControl('set_flip_tensor_withB1plus_and_offres: set B1plus before use')        # flip operator with B1plus inhomogeneity
    
        input_flips = torch.zeros((self.T,self.NRep,self.NVox,3), dtype=torch.float32)
        
        input_flips[...,0] = RF_event[...,0].unsqueeze(2)*self.B1plus.view([1,1,self.NVox])/t_pulse # w1
        input_flips[...,1] = RF_event[...,1].unsqueeze(2)                                           # phase
        input_flips[...,2]= RF_event[...,2].unsqueeze(2) + B0inhomo.view([1,1,self.NVox])*2*np.pi  # dw
        
        #Beff = pi/2 - atan(B1/delta B0),
        theta = np.pi/2 - torch.atan2(input_flips[...,0],input_flips[...,2])
                # theta = angle around Beff = FA/sin(atan(B1/delta B0))
        phi_eff = input_flips[...,0]/torch.sin(np.pi/2-theta)*t_pulse 
        phi_eff[torch.isnan(phi_eff)] = 0 # nan is where w1 = 0
        phi_eff[torch.isinf(phi_eff)] = 0 # inf is where w1 = inf
        phi_eff = phi_eff.unsqueeze(3).unsqueeze(3)
        
        Fglob = torch.zeros((self.T,self.NRep,self.NVox,3,3), dtype=torch.float32)
        
        Fglob = self.setdevice(Fglob)
        
        vx = torch.cos(input_flips[:,:,:,1])*torch.cos(theta)
        vy = torch.sin(input_flips[:,:,:,1])*torch.cos(theta)
#        input_flips[:,:,2] # frequency dwa in Hz

        vz = torch.sin(theta) #angle between xy plane and Beff = pi/2 - atan(B1/delta B0), 0 deg for on-resonant pulse

        Fglob[:,:,:,0,0] = 0
        Fglob[:,:,:,0,1] = -vz
        Fglob[:,:,:,0,2] = vy
        Fglob[:,:,:,1,0] = vz
        Fglob[:,:,:,1,1] = 0
        Fglob[:,:,:,1,2] = -vx
        Fglob[:,:,:,2,0] = -vy
        Fglob[:,:,:,2,1] = vx
        Fglob[:,:,:,2,2] = 0
    
        # matrix square
        F2 = torch.matmul(Fglob,Fglob)
#        F2 = torch.einsum("ijklm,ijkmn->ijkln",[Fglob,Fglob])
        
        
        # WARNING TODO, not future proof
#        if self.sz[0] > 64:
#            theta = theta[:(self.T - self.NCol)//2,:,:,:,:]
#            Fglob = Fglob[:(self.T - self.NCol)//2,:,:,:,:]
#            F2 = F2[:(self.T - self.NCol)//2,:,:,:,:]        
               
        F = torch.sin(phi_eff)*Fglob + (1 - torch.cos(phi_eff))*F2
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
        
#        IVP = torch.zeros((self.NSpins,1,self.NVox,3,3), dtype=torch.float32)
#        IVP = self.setdevice(IVP)
#        IVP[:,0,:,2,2] = 1
#        self.IVP = IVP          
        
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
        
    def set_gradient_precession_tensor_super(self,grad_moms,flips):
        # 0: flip, 1: phase, 2: offset freq. 3: usage 
        grads=grad_moms
        
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
        temp = temp[:-1,:,:]
        k=torch.cumsum(temp,0)  # only for initialisation
        
        kloc = 0
        for r in range(self.NRep):
            for t in range(self.T):
                if t != 0:
                    if flips[t-1,r,3]==1:  #is excitation         # +1 because of right gradmom shift (see above )
                        kloc=0
                    if flips[t-1,r,3]==2:  #is refocus pulse      # +1 because of right gradmom shift (see above )
                        kloc = -kloc
                kloc += temp[t,r,:]
                k[t,r,:] = kloc
        
        self.grads = grads
                # save grad_moms for intravoxel precession op
        self.grad_moms_for_intravoxel_precession = grad_moms
        
        self.kspace_loc = k
        
        
        
    def set_gradient_precession_tensor(self,grad_moms,sequence_class):
        
        # deprecated warning
        import warnings
#        warnings.warn('set_gradient_precession_tensor method is deprecated! use set_gradient_precession_tensor_super instead')
        
        grads=grad_moms
        
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
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
        
        # save grad_moms for intravoxel precession op
        self.grad_moms_for_intravoxel_precession = grad_moms
        
        self.kspace_loc = k
        
    def get_kmatrix(self,extraRep=1):
        '''
        reorder scanner signal according to kspace trajectory, works only for
        cartesian (under)sampling (where kspace grid points are hit exactly)
        '''
        sz = self.sz
        NCol = self.NCol
        NRep = self.NRep
        NCoils = self.NCoils
   
        adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
        # spectrum = self.signal[:,self.adc_mask.flatten()!=0,:,:2,0]
        spectrum = self.signal[:,self.adc_mask.flatten()!=0,:,:2,0].detach().cpu()
        
        if extraRep > 1:
            measRepStep = NRep//extraRep
            meas_indices=np.zeros((extraRep,measRepStep))
            kspace = self.setdevice(torch.zeros([extraRep,NCoils, sz[0], sz[1], 2]))
            for i in range(0,extraRep):
                meas_indices[i,:] = np.arange(i*measRepStep,(i+1)*measRepStep)
            for j in range(extraRep):
                        grid = self.kspace_loc[adc_idx,j*measRepStep:(j+1)*measRepStep,:].detach().cpu().numpy()
                        grid[:,:,0] += int(sz[0] / 2)
                        grid[:,:,1] += int(sz[1] / 2)
                        grid = grid.astype(int)
                        kspace[j,:,grid[:,:,0], grid[:,:,1],:] = spectrum[:,:,j*measRepStep:(j+1)*measRepStep,:]
        else:
            grid = self.kspace_loc[adc_idx,:,:].detach().cpu().numpy()
    #        grid = torch.flip(grid, [2]).detach().cpu().numpy()
            grid[:,:,0] += int(sz[0] / 2)
            grid[:,:,1] += int(sz[1] / 2)
            grid = grid.astype(int)
            kspace = torch.zeros([NCoils, sz[0], sz[1], 2])
            kspace[:,grid[:,:,0], grid[:,:,1],:] = spectrum
    
        return kspace
        
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
        intra_b0 = self.grad_moms_for_intravoxel_precession[t,r,:].unsqueeze(0) * self.intravoxel_dephasing_ramp
        intra_b0 = torch.sum(intra_b0,1)
        
        IVP_nspins_cos = torch.cos(intra_b0)
        IVP_nspins_sin = torch.sin(intra_b0)
         
        self.IVP[:,0,0,0,0] = IVP_nspins_cos
        self.IVP[:,0,0,0,1] = -IVP_nspins_sin
        self.IVP[:,0,0,1,0] = IVP_nspins_sin
        self.IVP[:,0,0,1,1] = IVP_nspins_cos
        
#    def set_grad_intravoxel_precess_tensor(self,t,r):
#        intra_b0 = self.grad_moms_for_intravoxel_precession[t,r,:].unsqueeze(0).unsqueeze(0) * self.intravoxel_dephasing_ramp
#        intra_b0 = torch.sum(intra_b0,2)
#        
#        IVP_nspins_cos = torch.cos(intra_b0)
#        IVP_nspins_sin = torch.sin(intra_b0)
#        
#        self.IVP[:,0,:,0,0] = IVP_nspins_cos
#        self.IVP[:,0,:,0,1] = -IVP_nspins_sin
#        self.IVP[:,0,:,1,0] = IVP_nspins_sin
#        self.IVP[:,0,:,1,1] = IVP_nspins_cos
   
        
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
    
    def do_reordered_ifft_reco(self, kmatrix=None):
        '''
        for cartesian (under-)sampling: reorder kspace signal according to 
        trajectory and perform ifft reco (similar as done in grappa_imspace,
        but without any grappa weights applied).
        
        either get kmatrix as argument, or use scanner signal reordered by
        get_kmatrix()
        
        return non-coil-combined reco

        '''
        if kmatrix is None:
            kspace = self.get_kmatrix()
        else:
            kspace = kmatrix
        
        # go to image space
        sig = fftshift(torch.ifft(fftshift(kspace,(1,2)),2), (1,2))

        # strange normalization
        sig = sig * torch.prod(torch.from_numpy(self.sz))
        
        sig = sig.flip([1,2]).permute([0,2,1,3]) 
        
        return sig
    
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
#            class ExecutionControl(Exception): pass
#            raise ExecutionControl('do_ifft_rec: parallel imaging not implemented!')
            
            # rough implementation of multicoil FFT
            reco = torch.zeros((self.NCoils,self.NVox,2), dtype = torch.float32)
            self.reco = self.setdevice(reco)              
            for jj in range(self.NCoils):

                spectrum = self.signal[jj,self.adc_mask.flatten()!=0,:,:2,0].clone()
                
                # fftshift
                pmOne = 0 # originally, roll was performed by NCol//2-1, however it seems to me that just NCol//2 gives rather correct phase images and avoids shift in image space by one pixel
                                   
                spectrum = roll(spectrum,self.NCol//2+pmOne,0)
                spectrum = roll(spectrum,self.NRep//2+pmOne,1)
                
                space = torch.ifft(spectrum,2)
                
                if self.NCol > self.sz[0]:
                    print("do_ifft_reco: oversampled singal detected, doing crop around center in space...")
                    hsz = (self.NCol - self.sz[0])//2
                    space = space[hsz:hsz+self.sz[0]]
                
                # fftshift
                space = roll(space,self.sz[0]//2+pmOne,0)
                space = roll(space,self.sz[1]//2+pmOne,1)
               
                space = torch.flip(space.permute([1,0,2]), (0,1)) #  flip / rot to match adjoint's orientation
                
                self.reco[jj,:,:] = space.reshape([self.NVox,2])
                
            self.reco = self.coil_combine(self.reco) # do coil combi here in order to have same behavior as adjoint()
        
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
                                
                            signal = torch.matmul(self.B1, signal[:,:,:,:2,:1])
                            signal = torch.sum(signal,[2])
                            signal *= self.adc_mask[start_t:start_t+half_read*2].view([1,signal.shape[1],1,1])
                            
                            self.signal[:,start_t:start_t+half_read*2,r,:,0] = signal.squeeze() / self.NSpins 
                            
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
        
        half_read = np.int(torch.sum(self.adc_mask != 0) //2)
        
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
                delay = torch.abs(event_time[t,r]) + 1e-6
                
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
                            
                            kum_grad_intravoxel = torch.cumsum(self.grad_moms_for_intravoxel_precession[start_t:start_t+half_read*2,r,:],0)
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
                        spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                            
                        FWD = torch.matmul(REW, FWD)
                        spins.M = torch.matmul(FWD,spins.M)                            
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay
                if self.grads[t,r,0] > 1.5*np.abs(self.sz[0]) and kill_transverse==1: # mode 1 kill transverse when gradient is high
                    spins.M[:,0,:,:2,0] = 0        
                spins.M_history.append(spins.M)
            if kill_transverse==2:      # mode 2: kill transverse at end of TR
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
                delay = torch.abs(event_time[t,r]) + 1e-6
                
                
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
                            
                            kum_grad_intravoxel = torch.cumsum(self.grad_moms_for_intravoxel_precession[start_t:start_t+half_read*2,r,:],0)
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
                            signal = torch.matmul(self.B1[:,:,PD0_mask,:,:], signal[:,:,:,:2,:1])
                            signal = torch.sum(signal,[2])
                            
                            self.signal[:,start_t:start_t+half_read*2,r,:2,0] = signal.squeeze() / self.NSpins 
                            
                        # do gradient precession (use adjoint as free kumulator)
                        if compact_grad_tensor:
                            FWD = G_adj_cut[t+1,:,:,:].permute([0,2,1])
                        else:
                            FWD = G_adj_cut[t+1,r,:,:,:].permute([0,2,1])
                            
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
                        spins_cut = B0InhomoClass.apply(self.SB0[:,:,PD0_mask,:,:],spins_cut,self)
                        
                        FWD = torch.matmul(REW, FWD)
                        spins_cut = torch.matmul(FWD,spins_cut)                            
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay
                if self.grads[t,r,0] > 1.5*np.abs(self.sz[0]) and kill_transverse==1: # mode 1 kill transverse when gradient is high
                    spins_cut[:,0,:,:2,0] = 0        
            if kill_transverse==2:      # mode 2: kill transverse at end of TR
                spins_cut[:,0,:,:2,0] = 0
                    
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.tmask = self.tmask[PD0_mask]
        self.lastM = spins_cut.clone()
        
        if self.noise_std > 0:
            noise = self.noise_std*torch.randn(self.signal[:,:,:,:2].shape).float()
            noise = self.setdevice(noise)
            self.signal[:,:,:,:2] += noise * self.adc_mask.view([1,self.T,1,1])
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal)            
            

    def forward_fast_supermem(self,spins,event_time,kill_transverse=False):
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
                            
                kum_grad_intravoxel = torch.cumsum(ctx.scanner.grad_moms_for_intravoxel_precession[t1:t2,r,:],0)
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
                kum_grad_intravoxel = torch.cumsum(ctx.scanner.grad_moms_for_intravoxel_precession[ctx.t1:ctx.t2,r,:],0)
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
                delay = torch.abs(event_time[t,r]) + 1e-6
                
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
                                
                            signal = torch.matmul(self.B1, signal[:,:,:,:2,:1])
                            signal = torch.sum(signal,[2])
                            signal *= self.adc_mask[start_t:start_t+half_read*2].view([1,signal.shape[1],1,1])
                            
                            self.signal[:,start_t:start_t+half_read*2,r,:2,0] = signal.squeeze() / self.NSpins 
                            
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
                        spins.M[:,:,:,:2,:] = AuxReadoutGradMul.apply(self.kspace_loc[[start_t,t+1],r,:],spins.M[:,:,:,:2,:],self)
                        
                        spins.M = B0InhomoClass.apply(self.SB0,spins.M,self)
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay
                if self.grads[t,r,0] > 1.5*np.abs(self.sz[0]) and kill_transverse==1: # mode 1 kill transverse when gradient is high
                    spins.M[:,0,:,:2,0] = 0        
            if kill_transverse==2:      # mode 2: kill transverse at end of TR
                spins.M[:,0,:,:2,0] = 0
                
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.lastM = spins.M.clone() 
        
        if self.noise_std > 0:
            noise = self.noise_std*torch.randn(self.signal[:,:,:,:2].shape).float()
            noise = self.setdevice(noise)
            self.signal[:,:,:,:2] += noise        
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal) * self.adc_mask.view([self.T,1,1,1])    
            
        torch.cuda.empty_cache()
        
    def forward_sparse_fast_supermem(self,spins,event_time,kill_transverse=False):
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
                            
                kum_grad_intravoxel = torch.cumsum(ctx.scanner.grad_moms_for_intravoxel_precession[t1:t2,r,:],0)
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
                kum_grad_intravoxel = torch.cumsum(ctx.scanner.grad_moms_for_intravoxel_precession[ctx.t1:ctx.t2,r,:],0)
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
                delay = torch.abs(event_time[t,r]) + 1e-6
                
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
                                
                            signal = torch.matmul(self.B1[:,:,PD0_mask,:,:], signal[:,:,:,:2,:1])
                            signal = torch.sum(signal,[2])
                            signal *= self.adc_mask[start_t:start_t+half_read*2].view([1,signal.shape[1],1,1])
                            
                            self.signal[:,start_t:start_t+half_read*2,r,:2,0] = signal.squeeze() / self.NSpins 
                            
                        # do gradient precession (use adjoint as free kumulator)
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
                        spins_cut[:,:,:,:2,:] = AuxReadoutGradMul.apply(self.kspace_loc[[start_t,t+1],r,:],spins_cut[:,:,:,:2,:],self)
                        
                        spins_cut = B0InhomoClass.apply(self.SB0[:,:,PD0_mask,:,:],spins_cut,self)
                        
                        # reset readout position tracking vars
                        start_t = t + 1
                        total_delay = delay
                    else:                                       # keep accumulating
                        total_delay += delay
                if self.grads[t,r,0] > 1.5*np.abs(self.sz[0]) and kill_transverse==1: # mode 1 kill transverse when gradient is high
                    spins_cut[:,0,:,:2,0] = 0        
            if kill_transverse==2:      # mode 2: kill transverse at end of TR
                spins_cut[:,0,:,:2,0] = 0          

                    
        # kill numerically unstable parts of M vector for backprop
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.tmask = self.tmask[PD0_mask]
        self.lastM = spins_cut.clone() 
        
        if self.noise_std > 0:
            noise = self.noise_std*torch.randn(self.signal[:,:,:,:2].shape).float()
            noise = self.setdevice(noise)
            self.signal[:,:,:,:2] += noise
        
        # rotate ADC phase according to phase of the excitation if necessary
        if self.AF is not None:
            self.signal = torch.matmul(self.AF,self.signal) * self.adc_mask.view([self.T,1,1,1])
            
        torch.cuda.empty_cache()        
            
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
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
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
        self.tmask = torch.zeros((self.NVox))
        self.tmask = self.setdevice(self.tmask).byte()
        
        self.tmask[spins.T1 < 1e-2] = 1
        self.tmask[spins.T2 < 1e-2] = 1
        
        self.tmask = self.tmask[PD0_mask]
        self.lastM = spins_cut.clone()
                

    def get_adapt_coil_weights(self, modeSVD=False):
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
        '''
        im = self.reco.reshape([self.NCoils,self.sz[0],self.sz[1],2]).detach()
        sz = im.size()
        n = torch.tensor(sz[1:-1]) # only spatial

        # block size for sliding window estimation of coil weights
        bs = torch.min(torch.tensor([7]),n) 
        
        # receiver noise matrix (may be measured by prescan noise calibration)
        rn = np.eye(self.NCoils) # receiver noise matrix
        inv_rn = np.linalg.inv(rn)
        
        if modeSVD: # intuitively: if more then 12 coils, use low rank approximation of coil images to determine coil weights
            nc_svd = min(min(12,max(9,np.floor(self.NCoils/2))),self.NCoils)
        
        weights = 1j*np.zeros([self.NCoils, n.prod()])
        
        # get coil with maximum signal for relative phase calibration
        maxcoil = torch.argmax(torch.sum(torch.sqrt(im[:,:,:,0]**2+im[:,:,:,1]**2).reshape([self.NCoils, n.prod()]),1))
        cnt = 0 # count voxels

        if modeSVD: # projection on first nc_svd singular vectors of coil images
            tmp = np.array(im[:,:,:,0]) + 1j * np.array(im[:,:,:,1]) # write as complex number, b/c not sure how to implement svd with real/imag handeled separately
            tmp = tmp.reshape([self.NCoils, -1])
            [U,S,VH] = np.linalg.svd(np.matmul(tmp, tmp.T.conjugate()))
            V = VH.T.conjugate()
            V = V[:,0:nc_svd]
        else:
            V = np.eye(self.NCoils)
            
        # sliding window calculation of coil weights
        for x in range(n[0]):
            for y in range(n[1]):
                # current position of sliding window
                ix = torch.tensor([x,y])
                imin = torch.max(ix - torch.floor(bs.float()/2), torch.tensor([0.])).int()
                imax = torch.min(ix + torch.ceil(bs.float()/2) -1, (n-1.)).int()
                
                # signal in sliding window
                m1 = im[:, imin[0]:imax[0], imin[1]:imax[1], :].reshape([self.NCoils, -1, 2])
                m1 = np.array(m1[:,:,0]) + 1j * np.array(m1[:,:,1]) # complex eigenvalues / eigenvectors for separately stored Re / Im?
                m1 = np.matmul(V.T.conjugate(), m1) # project on low-rank coil image space
                
                # signal covariance, hermitian!
                m = np.matmul(m1, m1.T.conjugate())
                
                # eigenvector of rn^-1 * m with largest modulus gives coil weights that maximize SNR
                d, v = np.linalg.eigh(np.matmul(inv_rn, m))
                tmp = v[:,-1] # for eigh eigenvalues are already sorted in ascending order
                tmp = np.matmul(V, tmp) # back-trafo in original coil space
                
                # correct phase based on coil with max intensity
                tmp = tmp * np.exp(-1j * np.angle(tmp[maxcoil])) 
                
                weights[:, cnt] = np.conj(tmp) / np.matmul(tmp.T.conjugate(), np.matmul(inv_rn, tmp))
                
                cnt += 1
        
        weights = weights.reshape([self.NCoils, n[0], n[1]])
        
        # prepare for complex multiplication in pytorch
        weights_torch = torch.zeros([self.NCoils, n[0], n[1], 2, 2])
        weights_torch[:,:,:,0,0] = torch.tensor(np.real(weights))
        weights_torch[:,:,:,0,1] = torch.tensor(-np.imag(weights))
        weights_torch[:,:,:,1,0] = torch.tensor(np.imag(weights))
        weights_torch[:,:,:,1,1] = torch.tensor(np.real(weights))
        
        return weights_torch
    
    def coil_combine_ext(self, im, weights_torch, donorm=True):
        '''
        apply externally calculated coil weights to combine coil images
        
        im:             uncombined coil images [NCoils, Nx, Ny, 2]
        weights_torch:  coil weights prepared for complex multiplication
        donorm:         do intensity normalization according to Griswold et al, the Use of an Adaptive Reconstruction for Array Coil Sensitivity Mapping and Intensity Normalization, Proceedings of the Tenth  Scientific Meeting of the International Society for Magnetic Resonance in Medicine pg 2410 (2002)
        
        return coil combined image [Nx, Ny, 2]
        '''
        weights_torch = self.setdevice(weights_torch)
        if im.dim() > 4:
            recon = self.setdevice(torch.zeros((im.shape[0],self.sz[0],self.sz[1],2)))
            for i in range(im.shape[0]):
                recon[i,:] = torch.einsum('ijklm,ijkm->jkl', weights_torch, im[i,:]) # [NCoils, Nx, Ny, 2, 2] x [NCoils, Nx, Ny, 2] -> [Nx, Ny, 2]
                recon[i,:] = recon[i,:].reshape([self.sz[0],self.sz[1],2]) 
                if donorm: # intensity normalization
                    # weights = self.tonumpy(weights_torch[:,:,:,0,0]) + 1j * self.tonumpy(weights_torch[:,:,:,1,0])
                    # norm = np.sum(np.abs(weights), 0) ** 2
                    
                    norm = self.setdevice(torch.sum(torch.sqrt(weights_torch[:,:,:,0,0]**2 + weights_torch[:,:,:,1,0]**2),0) ** 2)
                    
                    recon[i,:,:,0] *= norm
                    recon[i,:,:,1] *= norm
                
            return recon.flip([1,2]).permute([0,2,1,3])
        else:        
            recon = torch.einsum('ijklm,ijkm->jkl', weights_torch, im) # [NCoils, Nx, Ny, 2, 2] x [NCoils, Nx, Ny, 2] -> [Nx, Ny, 2]
            recon = recon.reshape([self.sz[0],self.sz[1],2])
            
            if donorm: # intensity normalization
                # weights = self.tonumpy(weights_torch[:,:,:,0,0]) + 1j * self.tonumpy(weights_torch[:,:,:,1,0])
                # norm = np.sum(np.abs(weights), 0) ** 2
                
                norm = torch.sum(torch.sqrt(weights_torch[:,:,:,0,0]**2 + weights_torch[:,:,:,1,0]**2),0) ** 2
                
                recon[:,:,0] *= norm
                recon[:,:,1] *= norm
    
            return recon.flip([0,1]).permute([1,0,2])
    

    def coil_combine(self, reco, donorm=True):
        '''
        perform coil combination as specified in coilCombineMode
        reco:   [NCoils, Nx*Ny, 2], e.g. from adjoint reco
        
        return coil combined image [Nx, Ny, 2]
        '''
        if self.coilCombineMode.lower() == "complex_naive": # just complex sum (coilWeights=1), for testing only
                tmp = torch.sum(reco,0)
                return self.reco.reshape([self.sz[0],self.sz[1],2]).flip([0,1]).permute([1,0,2]).reshape([self.NVox,2])
                
        elif self.coilCombineMode.lower() == "adaptive": # Adaptive recon based on Walsh et al. Adaptive reconstruction of phased array MR imagery. Magnetic Resonance in Medicine. 2000;43(5):682-690. doi:10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G
#            import pdb; pdb.set_trace()
            self.reco = reco
            weights_torch = self.get_adapt_coil_weights(modeSVD=False)
            self.coilWeights = weights_torch
            recon = self.coil_combine_ext(reco.reshape([self.NCoils,self.sz[0],self.sz[1],2]), weights_torch, donorm=donorm)
            return recon.reshape([self.NVox,2])
                    
        elif self.coilCombineMode.lower() == "roemer": # use full complex sensitivity maps as coil weighting factors, this should give best SNR (Roemer et al. The NMR phased array. Magnetic Resonance in Medicine. 1990;16(2):192-225. doi:10.1002/mrm.1910160203)
            coilWeights = self.B1;
            # complex conjugate as sign switch when writing complex multiplication in matrix form
            coilWeights[:,:,:,0,1] = -coilWeights[:,:,:,0,1]
            coilWeights[:,:,:,1,0] = -coilWeights[:,:,:,1,0]
            recon = torch.einsum('ijklm,ikm->kl', coilWeights, reco) # [NCoils,1,Nvox,2,2] x [NCoils, Nvox, 2] -> [Nvox, 2]
            return recon.reshape([self.sz[0],self.sz[1],2]).flip([0,1]).permute([1,0,2]).reshape([self.NVox,2])
            
        else: # sum-of-squares
            # do SOS
            recon = reco.reshape([self.NCoils,self.sz[0],self.sz[1],2])
            recon = torch.sqrt(torch.sum(recon[:,:,:,0]**2 + recon[:,:,:,1]**2,0))
            recon = recon.reshape([self.sz[0],self.sz[1]]).flip([0,1]).permute([1,0]).reshape([self.NVox]).unsqueeze(1).repeat([1,2]) # results in Nvox x 2, both containing SOS magnitude
            recon[:,1] = 0 # set phase to 0
            return recon

    def create_grappa_weight_set(self, kernel_size, accel, full_sz, lambd=0.01, NACS=None):
        '''
        2D implementation of Felix Breuer's grappa_imspace, as included in recoVBVD
        ---------
        create grappa weight set from ACS scan
        
        accel:          acceleration factor in phase direction
        kernel_size:    (sblx, sbly)
        full_sz:        full size [nx,ny] of the reconstructed image
        lambd:          regularization factor for pseudoinverse
        NACS:           number of ACS lines: optional
                        if this is not given, all non-zero PE lines are used
        
        returns ws, wsKernel, ws_mult
        ws:         grappa weight set (for kspace reco by convolution)
        wsKernel:   weight set in image space for reco by multiplication
        ws_mult:    weight set in image space for reco by multiplication with
                    last two dims prepared for complex multiplication
        '''
        
        NCoils = self.NCoils
        
        # crop calibration kspace (remove zeros in the outer parts)
        kspace = self.get_kmatrix().numpy()
        
        if NACS is not None: # manual cropping
            acs = kspace[:,:,int(full_sz[1]//2-NACS/2):int(full_sz[1]//2+NACS/2),:]
        else:
            keep_indices = np.argwhere(np.sum(kspace,(0,-1))[0,:]) # keep only PE lines with nonzero signal (sum over coils and real/imag)
            acs = kspace[:,:,np.min(keep_indices):np.max(keep_indices)+1,:]
        
        # to complex (need to do complex matrix inversion later based on numpy)
        # -> not clear how to do that only in pytorch
        acs = acs[:,:,:,0] + 1j *acs[:,:,:,1]
        
        nc, nxacs, nyacs = acs.shape  
        sblx, sbly = kernel_size
        afy = accel
        
        SrcCell, TrgCell, tbly, shift = self.get_source_and_target_cell(accel)
        
        patSrc = np.tile(SrcCell[:,0].T, (sblx, int(np.ceil(sbly/afy/2)*2+1)))
        patSrc = patSrc[:,int(np.floor((patSrc.shape[1] - sbly)/2)):int(np.floor((patSrc.shape[1] + sbly)/2))]
        patTrg = np.zeros((1, sbly))
        patTrg[0, int(np.ceil((sbly - afy)/2)):int(np.ceil((sbly + afy)/2))] = 1
        
        SrcIdx = patSrc.flatten() == 1
        TrgIdx = patTrg.flatten() == 1
        
        nsp = SrcIdx[SrcIdx==1].shape[0]
        
        
        # Collect all the source and target replicates 
        # within the ACS data in the src and trg-matrix 
        src = np.zeros((nc*nsp, (nxacs-sblx+1)*(nyacs-sbly+1)), dtype=acs.dtype)
        trg = np.zeros((nc*afy, (nxacs-sblx+1)*(nyacs-sbly+1)), dtype=acs.dtype)
        
        cnt = 0
        for y in range(nyacs-sbly+1):
            for x in range(nxacs-sblx+1):
                # source points
                s = acs[:, x:x+sblx, y:y+sbly]
                s = s.reshape((NCoils,-1))[:, SrcIdx]
                
                # target points
                t = acs[:, x+int(np.floor((sblx-1)/2)), y:y+sbly]
                t = t[:, TrgIdx]
                
                if (0 in np.sum(s,0)) or (0 in np.sum(t,0)): # check whether there is missing data (e.g. in case elliptical scanning was used in ref. scan)
                    continue
                
                src[:,cnt] = s.flatten()
                trg[:,cnt] = t.flatten()
                cnt += 1
                
        src = src[:,0:cnt]
        trg = trg[:,0:cnt]
        
        # now solve for weights using regularized pseudo inverse
        ws = trg.dot(pinv_reg(src, lambd))
        ws = ws.reshape((NCoils, afy, NCoils, nsp))
        
        # convert to convolution kernel
        ws_tmp = np.zeros((NCoils, afy, NCoils, sblx, sbly), dtype=acs.dtype)
        ws_tmp[:,:,:,patSrc==1] = ws
        
        ksz_y = int(np.min([int(np.floor((sbly + tbly - 1)/2))*2 + 1, full_sz[1]]))
        kernel = np.zeros((NCoils,afy,NCoils,sblx,ksz_y), dtype=acs.dtype)
        kernel[:,:,:,:,int(np.floor((ksz_y - sbly)/2))+np.arange(sbly)] = ws_tmp
        
        # Flip the weights in x,y
        kernel = np.flip(np.flip(kernel, 3), 4)
        wsKernel = np.zeros((NCoils,NCoils,sblx,ksz_y), dtype=acs.dtype)
        for k in range(afy):
            wsKernel += np.roll(kernel[:,k,:,:,:], (0,0,0, shift[k]))
        
        
        nx, ny = full_sz
       
        _, _, nxw, nyw = wsKernel.shape
        ws_imspace = np.zeros((NCoils,NCoils,nx,ny),dtype=acs.dtype)
        ix_x = 1
        ix_y = 1
        if nx > 1:
            ix_x = int(np.floor((nx - nxw)/2+1))
        if ny > 1:
            ix_y = int(np.floor((ny - nyw)/2+1))
            
        ws_imspace[:,:,ix_x:ix_x+nxw,ix_y:ix_y+nyw] = wsKernel
        
        ws_imspace = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ws_imspace, (-2,-1))), (-2,-1))
        
        ws_imspace_t = torch.zeros(ws_imspace.shape + (2,))
        ws_imspace_t[:,:,:,:,0] = torch.from_numpy(np.real(ws_imspace))
        ws_imspace_t[:,:,:,:,1] = torch.from_numpy(np.imag(ws_imspace))
        
        # prepare weights for complex multiplication
        ws_mult = torch.zeros(ws_imspace_t.shape+(2,))
        ws_mult[:,:,:,:,0,0] = ws_imspace_t[:,:,:,:,0]
        ws_mult[:,:,:,:,0,1] = -ws_imspace_t[:,:,:,:,1]
        ws_mult[:,:,:,:,1,0] = ws_imspace_t[:,:,:,:,1]
        ws_mult[:,:,:,:,1,1] = ws_imspace_t[:,:,:,:,0]
        
        return wsKernel, ws_imspace_t, ws_mult
        
                
    def get_source_and_target_cell(self, afy):
        '''
        2D implementation of Felix Breuer's grappa_imspace, as included in recoVBVD
        ---------
        get indices of source and target kspace points
        
        simplified version without any CAIPI capability
        (since we are doing only 2D anyway up to now)
        '''
        
        # define elementary sampling pattern cell
        pat = np.zeros([afy, afy])
        pat[int(np.floor(afy / 2)),:] = 1
        tbly = afy # redundant for 2D Grappa?
        
        SrcCell = pat
        TrgCell = pat.T
        
        shift = np.arange(-np.floor(afy/2), -np.floor(afy/2)+afy, dtype=int) # only in y!
        
        return SrcCell, TrgCell, tbly, shift
    

    def grappa_imspace(self, ws_imspace, extraRep=1):
        '''
        2D implementation of Felix Breuer's grappa_imspace, as included in recoVBVD
        ---------
        
        do grappa reco of scanner signal with previously calculated weight set wsKernel
        apply grappa weights in image space (multiplication)
        '''
        kspace = self.get_kmatrix(extraRep)
        
        if extraRep > 1:
            sig = torch.zeros_like(kspace)
            for i in range(extraRep):
                sig[i,:] = fftshift(torch.ifft(fftshift(kspace[i,:],(1,2)),2), (1,2))   
                sig[i,:] = torch.einsum('ijklmn,jkln->iklm', ws_imspace, sig[i,:])
                sig[i,:] = sig[i,:] * torch.prod(torch.from_numpy(self.sz))
                
                sig[i,:] = sig[i,:].flip([1,2]).permute([0,2,1,3]) 
        else:
        # go to image space
            sig = fftshift(torch.ifft(fftshift(kspace,(1,2)),2), (1,2))
    
            # apply weights matrix and complex multiplication matrix simultaneously
            sig = torch.einsum('ijklmn,jkln->iklm', self.setdevice(ws_imspace), sig)
            sig = sig * torch.prod(torch.from_numpy(self.sz))
            
            sig = sig.flip([1,2]).permute([0,2,1,3]) 
        
        return sig
        
        # calc g factor from weights
#        g = self.grappa_gfactor(wsKernel)
        


    def grappa_gfactor(self, wsKernel, R=None, coilWeights=None):
        '''
        analytical g factor calculation according to
        Breuer et al. MRM 2009;62(3):739-746. doi:10.1002/mrm.22066
        
        wsKernel:       grappa weight set in image space (from create_grappa_weight_set)
        R:              noise covariance matrix (eye if None is given)
        coilWeights:    if given, combine coil gfactors as described in the paper,
                        if not given, return g factor for each coil individually
        '''
        af = self.accel
        sig_sz = self.sz
        
        st = 1 #interpolation factor -> not implemented yet, keep at 1
        nx = int(2*np.floor(sig_sz[0]/(2*st)))
        ny = int(2*np.floor(nx*sig_sz[1]/sig_sz[0]/2))
        
        nc, _, nxw, nyw = wsKernel.shape
        
        if R is None:
            R = np.eye(nc)
            
        wsImg = np.zeros((nc,nc,nx,ny), dtype=wsKernel.dtype)
        wsImg[:,:,int(np.floor((nx-nxw)/2)):int(np.floor((nx-nxw)/2)+nxw), int(np.floor((ny-nyw)/2)):int(np.floor((ny-nyw)/2)+nyw)] = wsKernel*nx*ny
        wsImg = np.fft.fftshift(np.fft.ifft2(wsImg),(-2,-1))
        
        
        if coilWeights is None: # calculate gfator individually in each coil
            g = np.zeros((nc,nx,ny))
            for y in range(ny):
                for x in range(nx):
                    W = wsImg[:,:,x,y]
                    g[:,x,y] = np.sqrt(np.abs(np.diag(W.dot(R).dot(W.conj().T)))) / np.sqrt(np.diag(R)) / af # why /af ?
        else: # if coil weights are given, use them to calculate proper single g factor
            g = np.zeros((nx,ny))
            if coilWeights.ndim == 5: # from torch to complex numpy
                coilWeights = self.tonumpy(coilWeights[:,:,:,0,0]) + 1j * self.tonumpy(coilWeights[:,:,:,1,0])
            for y in range(ny):
                for x in range(nx):
                    W = wsImg[:,:,x,y]
                    p = coilWeights[:,x,y]
                    pT = p.T
                    upper = (pT.dot(W)).dot(R).dot((pT.dot(W)).conj().T)
                    lower = pT.dot(R).dot(pT.conj().T)
                    g[x,y] = np.sqrt(np.abs(upper)) / np.sqrt(np.abs(lower)) / af # why /af ?
                
        return np.abs(g)
        
    

    def conjugate_grad_pi(self, maxiter, rtol, sensitivities, reg=0, calc_gfactor=False, gfactor_reg_eps = 1e-3):
        '''
        CG SENSE reco (arbitrary trajectory with parallel imaging)
        
        maxiter:        maximum number of CG iterations
        rtol:           relative tolerance for CG
        sensitivities:  receive coil sensitivities assumed for reco
                        (NCoils x 1 x NVox x 2 x 2), shape like scanner.B1
                        last two axes Re and Im prepared for complex multiplication
        reg:            Tikhonov regularization parameter
        calc_gfactor:   whether to also calculate g factor (heavy computation!)
        gfactor_reg_eps: EXPERIMENTAL: try to regularize gfactor calc, as it tends to diverge

        '''
        self.init_reco()
    
        s = self.signal
        adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
        s = s[:,adc_idx,:,:2] # kx x ky x 2 x 1
        s = s.view([self.NCoils*adc_idx.size*self.NRep*2,1])
        
        # Fourier part of forward model
        # NVox x 2 x Kx x Ky x 2
        A_singlechannel = self.G_adj[adc_idx,:,:,:2,:2].permute([2,3,0,1,4])
        
        # NCoils x 1 x NVox x 2 x 2
        # sensitivities = scanner.B1 # realistically, here the measured sensitivities should be used 
        
        # complex multiplication of sensitivities and fourier terms (last axis of A_singlechannel)
        A = torch.einsum('cijkl,jmxyl->cjmxyk', sensitivities, A_singlechannel)
        
        # bring to shape [NCoils*Kx*Ky*2, NVox*2]
        A = A.permute([0,3,4,5,1,2]).contiguous().view([self.NCoils*adc_idx.size*self.NRep*2, self.NVox*2])
        
        
        AtA = torch.matmul(A.permute([1,0]),A)
        b = torch.matmul(A.permute([1,0]),s)
        
        AtA *= 0.01
        b *= 0.01
        
        AtA_reg = AtA + self.setdevice(torch.eye(AtA.shape[0])) * reg
        def cgmm(x):
            return torch.matmul(AtA_reg.unsqueeze(0),x)
            
        r = auxutil.cg_batch.cg_batch(cgmm,b.unsqueeze(0),maxiter=maxiter,rtol=rtol,verbose=True)[0] 
    
        self.reco = r.view([self.NVox,2,1])[:,:,0]
        self.reco = self.reco.reshape([self.sz[0],self.sz[1],2]).flip([0,1]).permute([1,0,2]).reshape([self.NVox,2])
        
        if calc_gfactor:
            AtA_reg2 = AtA + self.setdevice(gfactor_reg_eps * torch.eye(AtA.shape[0]))
        
            self.gfactor = (torch.diag(torch.inverse(AtA)) * torch.diag(AtA)) ** (0.5) # torch.sqrt had problems with backpropagation here
            self.gfactor_reg = (torch.diag(torch.inverse(AtA_reg2)) * torch.diag(AtA_reg2)) ** (0.5)
            self.reco_cond = torch.norm(torch.inverse(AtA)) * torch.norm(AtA) # this is condition number with respect to Frobenius norm, which is easier to calculate than the one for the operator / spectral norm


    def conjugate_grad(self, maxiter, rtol):
        ''' single channel version of CG reco '''
        self.init_reco()
        
        s = torch.sum(self.signal,0)
        adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
        s = s[adc_idx,:,:2]
        s = s.view([adc_idx.size*self.NRep*2,1])
        
        A = self.G_adj[adc_idx,:,:,:2,:2].permute([2,3,0,1,4]).contiguous().view([scanner.self*2,adc_idx.size*scanner.self*2]).permute([1,0])
        AtA = torch.matmul(A.permute([1,0]),A)
        b = torch.matmul(A.permute([1,0]),s)
        
        AtA *= 0.01
        b *= 0.01
        
        def cgmm(x):
            return torch.matmul(AtA.unsqueeze(0),x)
            
        r = auxutil.cg_batch.cg_batch(cgmm,b.unsqueeze(0),maxiter=maxiter,rtol=rtol,verbose=False)[0] 
        self.reco = r.view([self.NVox,2,1])[:,:,0]
        self.reco = self.reco.reshape([self.sz[0],self.sz[1],2]).flip([0,1]).permute([1,0,2]).reshape([self.NVox,2])


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
        if self.NCoils > 1:
            self.reco_uncombined = self.reco.reshape((self.NCoils, self.sz[0], self.sz[1], 2))
            self.reco = self.coil_combine(self.reco)
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
    def do_grad_adj_reco_separable_parallel(self,r):
        s = self.signal[:,:,r,:,:]
        s = s.contiguous().view([self.NCoils,1,self.T*3,1])
        g = self.G_adj.permute([1,2,0,3]).contiguous().view([self.NVox,3,self.T*3])
        r = torch.matmul(g, s)
        return r
    
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
    
    def adjoint_separable_parallel(self, extraRep_idx=-1):
        nrm = np.sqrt(np.prod(self.sz))
        self.init_reco()
        r = self.setdevice(torch.zeros((self.NRep,self.NCoils,self.NVox,3,1)).float())
        # s = self.signal.permute([0,2,1,3,4]).contiguous().view([self.NCoils,self.NRep,1,self.T*3,1])
        # g = self.G_adj.permute([1,2,3,0,4]).contiguous().view([self.NRep,self.NVox,3,self.T*3])
        
        # r = torch.matmul(g, s)
        for rep in range(self.NRep):
            self.set_grad_adj_op(rep)
            r[rep,:,:,:,:] = self.do_grad_adj_reco_separable_parallel(rep)        
        
        r = r[:,:,:,:2,0].permute(1,0,2,3)
        extraRep = (self.NRep/self.sz[0]).astype(int)
        meas_indices=np.zeros((extraRep,self.sz[0]))
        for i in range(0,extraRep):
            meas_indices[i,:] = np.arange(i*self.sz[0],(i+1)*self.sz[0])

        reco_all_rep=torch.zeros((self.NCoils,extraRep,self.NVox,2))
        for j in range(0,extraRep):
            reco_all_rep[:,j,:,:] = r[:,meas_indices[j,:],:,:].sum(1)
        if self.coilCombineMode.lower() == "adaptive":
            recon_comb = torch.zeros((extraRep,self.NVox,2))
            self.reco = reco_all_rep[:,extraRep_idx,:,:]
            weights_torch = self.get_adapt_coil_weights(modeSVD=False)
            for rep in range(extraRep):                
                self.reco = reco_all_rep[:,rep,:,:]
                self.coilWeights = weights_torch
                recon_comb[rep,:,:] = self.coil_combine_ext(self.reco.reshape([self.NCoils,self.sz[0],self.sz[1],2]), weights_torch, donorm=True).reshape(self.NVox,2)
                
        else:
            #sum of squares
            recon_comb = torch.zeros((extraRep,self.NVox,2))
            recon_comb[:,:,0] = torch.sqrt(torch.sum(reco_all_rep[:,:,:,0]**2+reco_all_rep[:,:,:,1]**2,0))
            recon_comb = recon_comb.reshape([extraRep,self.sz[0],self.sz[1],2]).flip([1,2]).permute([0,2,1,3]).reshape([extraRep,self.NVox,2])
            
            self.reco = torch.sum(r,0) / nrm
    
        return recon_comb

    def get_base_path(self, experiment_id, today_datestr):

        if os.path.isfile(os.path.join('core','pathfile_local.txt')):
            pathfile ='pathfile_local.txt'
        else:
            pathfile ='pathfile.txt'
            print('You dont have a local pathfile in core/pathfile_local.txt, so we use standard file: pathfile.txt')

        with open(os.path.join('core',pathfile),"r") as f:
            path_from_file = f.readline()
        if platform == 'linux':
            hostname = socket.gethostname()
            if hostname == 'vaal' or hostname == 'madeira4' or hostname == 'gadgetron':
                basepath = '/media/upload3t/CEST_seq/pulseq_zero'
            else:                                                     # cluster
                basepath = 'out'
        else:
            basepath = path_from_file
        basepath_seq = os.path.join(basepath, 'sequences')
        basepath_seq = os.path.join(basepath_seq, "seq" + today_datestr)
        basepath_seq = os.path.join(basepath_seq, experiment_id)

        return basepath, basepath_seq

    # interaction with real system
    def send_job_to_real_system(self, experiment_id, today_datestr, basepath_seq_override=None, jobtype="target", iterfile=None):
        basepath, basepath_seq = self.get_base_path(experiment_id, today_datestr)
        if platform == 'linux':
            basepath_control = '/media/upload3t/CEST_seq/pulseq_zero/control'
        else:
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
            
    def get_signal_from_real_system(self, experiment_id, today_datestr, basepath_seq_override=None, jobtype="target", iterfile=None,single_folder=False):
        _, basepath_seq = self.get_base_path(experiment_id, today_datestr)
        
        if basepath_seq_override is not None:
            basepath_seq = basepath_seq_override        
            
        print('waiting for TWIX file from the scanner...')
        
        
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
            
        print('waiting for TWIX file from the scanner... ' + os.path.join(basepath_seq, fn_twix))
        
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
                print('raw shape', raw.shape)
                
                dp_twix = os.path.dirname(fnpath)
                shutil.move(fnpath, os.path.join(dp_twix,"data",fn_twix))
#                import pdb; pdb.set_trace();
                heuristic_shift = 4
                print("raw size: {}, ".format(raw.size) + "expected size: {} ".format(self.NRep*ncoils*(self.NCol+heuristic_shift)*2) )
                
                                
                if raw.size != self.NRep*ncoils*(self.NCol+heuristic_shift)*2: # only zeros!
                    print("get_signal_from_real_system: SERIOUS ERROR, TWIX dimensions corrupt, returning zero array..")
                    raw = np.zeros((self.NRep,ncoils,self.NCol+heuristic_shift,2))
                    raw = raw[:,:,:(self.NCol-heuristic_shift),0] + 1j*raw[:,:,:(self.NCol-heuristic_shift),1]
                else:
                      raw = raw.reshape([self.NRep,ncoils,self.NCol+heuristic_shift,2])
                      raw = raw[:,:,:self.NCol,0] + 1j*raw[:,:,:self.NCol,1]
                      #raw = raw[:,:,:-heuristic_shift,0] + 1j*raw[:,:,:-heuristic_shift,1]
#                      raw /= np.max(np.abs(raw))
#                      raw /= 0.05*0.014/9
                
                # inject into simulated scanner signal variable
                adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
                
                raw = raw.transpose([2,1,0])
                raw = np.copy(raw)
                #raw = np.roll(raw,1,axis=0)
            
                
                # assume for now a single coil
#                import pdb; pdb.set_trace();
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
          
    def get_data_from_file(self, fnpath):
            
        if os.path.isfile(fnpath):
            # read twix file
 
            raw_file = fnpath
            ncoils = 20

            raw = np.loadtxt(raw_file)
            print('raw shape', raw.shape)

         
            heuristic_shift = 4
            print("raw size: {}, ".format(raw.size) + "expected size: {} ".format(self.NRep*ncoils*(self.NCol+heuristic_shift)*2) )
            
                            
            if raw.size != self.NRep*ncoils*(self.NCol+heuristic_shift)*2: # only zeros!
                print("get_signal_from_real_system: SERIOUS ERROR, TWIX dimensions corrupt, returning zero array..")
                raw = np.zeros((self.NRep,ncoils,self.NCol+heuristic_shift,2))
                raw = raw[:,:,:(self.NCol-heuristic_shift),0] + 1j*raw[:,:,:(self.NCol-heuristic_shift),1]
            else:
                  raw = raw.reshape([self.NRep,ncoils,self.NCol+heuristic_shift,2])
                  raw = raw[:,:,:self.NCol,0] + 1j*raw[:,:,:self.NCol,1]
                  #raw = raw[:,:,:-heuristic_shift,0] + 1j*raw[:,:,:-heuristic_shift,1]
#                      raw /= np.max(np.abs(raw))
#                      raw /= 0.05*0.014/9
            
            # inject into simulated scanner signal variable
            adc_idx = np.where(self.adc_mask.cpu().numpy())[0]
            
            raw = raw.transpose([2,1,0])
            raw = np.copy(raw)
            #raw = np.roll(raw,1,axis=0)
        
            
            # assume for now a single coil
#                import pdb; pdb.set_trace();
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
                
            
           
            
    def get_signal_from_real_system_raw(self, experiment_id, today_datestr, basepath_seq_override=None, jobtype="target", iterfile=None):
        _, basepath_seq = self.get_base_path(experiment_id, today_datestr)
        
        if basepath_seq_override is not None:
            basepath_seq = basepath_seq_override        
            
        print('wating for TWIX file from the scanner...')
        
        if jobtype == "target":
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
                shutil.move(fnpath, os.path.join(dp_twix,"data",fn_twix))
                
                heuristic_shift = 4
                print("raw size: {} ".format(raw.size) + "expected size: {} ".format("raw size: {} ".format(self.NRep*ncoils*(self.NCol+heuristic_shift)*2)) )
#                import pdb; pdb.set_trace();
                                
                if raw.size != self.NRep*ncoils*(self.NCol+heuristic_shift)*2:
                      print("get_signal_from_real_system: SERIOUS ERROR, TWIX dimensions corrupt, returning zero array..")
                      raw = np.zeros((self.NRep,ncoils,self.NCol+heuristic_shift,2))
                      raw = raw[:,:,:self.NCol,0] + 1j*raw[:,:,:self.NCol,1]
                else:
                      raw = raw.reshape([self.NRep,ncoils,self.NCol+heuristic_shift,2])
                      raw = raw[:,:,:self.NCol,0] + 1j*raw[:,:,:self.NCol,1]
                      #raw = raw[:,:,:-heuristic_shift,0] + 1j*raw[:,:,:-heuristic_shift,1]
#                      raw /= np.max(np.abs(raw))
#                      raw /= 0.05*0.014/9
                
                # inject into simulated scanner signal variable
                
                raw = raw.transpose([2,1,0])
        return raw    
                
# Fast, but memory inefficient version 
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
    def set_gradient_precession_tensor_super(self,grad_moms,flips):
        # 0: flip, 1: phase, 2: offset freq. 3: usage 
        grads=grad_moms
        
        padder = torch.zeros((1,self.NRep,2),dtype=torch.float32)
        padder = self.setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
        temp = temp[:-1,:,:]
        k=torch.cumsum(temp,0)  # only for initialisation
        
        B0X = torch.unsqueeze(grad_moms[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        B0_grad_cos = torch.cos(B0_grad)
        B0_grad_sin = torch.sin(B0_grad)
        
        kloc = 0
        for r in range(self.NRep):
            for t in range(self.T):
                if t != 0:
                    if flips[t-1,r,3]==1:  #is excitation         # +1 because of right gradmom shift (see above )
                        kloc=0
                    if flips[t-1,r,3]==2:  #is refocus pulse      # +1 because of right gradmom shift (see above )
                        kloc = -kloc
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
        
        self.grads = grads
        # save grad_moms for intravoxel precession op
        self.grad_moms_for_intravoxel_precession = grad_moms
        
        self.kspace_loc = k
        
    def set_gradient_precession_tensor(self,grad_moms,sequence_class):
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
        self.grads = grad_moms
        
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
        
    def adjoint_multicoil(self):
        self.init_reco()
        
        #s = torch.sum(self.signal,0)
        #r = torch.matmul(self.G_adj.permute([2,3,0,1,4]).contiguous().view([self.NVox,3,self.T*self.NRep*3]), s.view([1,self.T*self.NRep*3,1]))
        #import pdb; pdb.set_trace()
        
        nrm = np.sqrt(np.prod(self.sz))
        # [Nevnt x NRep x (Nx*Ny) x 3 x 3] , [NCoils x Nevnt x NRep x 3 x 1] -> [NCoils x (Nx*Ny) x 3 x 1]
        r = torch.einsum("ijkln,oijnp->oklp",[self.G_adj, self.signal])
        self.reco = r[:,:,:2,0] / nrm
        
        # transpose for adjoint
        self.reco = self.reco.reshape([self.NCoils,self.sz[0],self.sz[1],2]).flip([1,2]).permute([0,2,1,3]).reshape([self.NCoils,self.NVox,2])
        
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
        
    def adjoint_separable_parallel(self, extraRep_idx=-1):
        nrm = np.sqrt(np.prod(self.sz))
        self.init_reco()
        
        s = self.signal.permute([0,2,1,3,4]).contiguous().view([self.NCoils,self.NRep,1,self.T*3,1])
        g = self.G_adj.permute([1,2,3,0,4]).contiguous().view([self.NRep,self.NVox,3,self.T*3])
        
        r = torch.matmul(g, s)
        r = r[:,:,:,:2,0]
        extraRep = (self.NRep/self.sz[0]).astype(int)
        meas_indices=np.zeros((extraRep,self.sz[0]))
        for i in range(0,extraRep):
            meas_indices[i,:] = np.arange(i*self.sz[0],(i+1)*self.sz[0])

        reco_all_rep=torch.zeros((self.NCoils,extraRep,self.NVox,2))
        for j in range(0,extraRep):
            reco_all_rep[:,j,:,:] = r[:,meas_indices[j,:],:,:].sum(1)
        if self.coilCombineMode.lower() == "adaptive":
            recon_comb = torch.zeros((extraRep,self.NVox,2))
            self.reco = reco_all_rep[:,extraRep_idx,:,:]
            weights_torch = self.get_adapt_coil_weights(modeSVD=False)
            for rep in range(extraRep):                
                self.reco = reco_all_rep[:,rep,:,:]
                self.coilWeights = weights_torch
                recon_comb[rep,:,:] = self.coil_combine_ext(self.reco.reshape([self.NCoils,self.sz[0],self.sz[1],2]), weights_torch, donorm=True).reshape(self.NVox,2)
        else:
            #sum of squares
            recon_comb = torch.zeros((extraRep,self.NVox,2))
            recon_comb[:,:,0] = torch.sqrt(torch.sum(reco_all_rep[:,:,:,0]**2+reco_all_rep[:,:,:,1]**2,0))
            recon_comb = recon_comb.reshape([extraRep,self.sz[0],self.sz[1],2]).flip([1,2]).permute([0,2,1,3]).reshape([extraRep,self.NVox,2])
            
            self.reco = torch.sum(r,0) / nrm
        
        return recon_comb
        
    def adjoint_separable(self):
        nrm = np.sqrt(np.prod(self.sz))
        self.init_reco()
        
#        s = torch.sum(self.signal,0) # sum over coil signals
        s = self.signal
        s = s.permute([0,2,1,3,4]).contiguous().view([self.NCoils,self.NRep,1,self.T*3,1])
        g = self.G_adj.permute([1,2,3,0,4]).contiguous().view([self.NRep,self.NVox,3,self.T*3])
        
        r = torch.matmul(g, s)
        r = r[:,:,:,:2,0]
        r = r.reshape([self.NCoils,self.NRep,self.sz[0],self.sz[1],2]).flip([2,3]).permute([0,1,3,2,4]).reshape([self.NCoils,self.NRep,self.NVox,2])
            
        if self.NCoils > 1: # coil combination for each repetition separately
            rr = torch.zeros([self.NRep, self.NVox, 2])
            for jj in range(self.NRep):
                rr[jj,:,:] = self.coil_combine(r[:,jj,:,:])
            self.reco = torch.sum(rr,0) / nrm
            return rr
        else:
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