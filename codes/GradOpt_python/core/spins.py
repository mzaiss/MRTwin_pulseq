import numpy as np
import torch
import cv2

def throw(msg):
    class ExecutionControl(Exception): pass
    raise ExecutionControl(msg)

# WHAT we measure
class SpinSystem():
    
    def __init__(self,sz,NVox,NSpins,use_gpu):
        
        self.sz = sz                                             # image size
        self.NVox = sz[0]*sz[1]                                 # voxel count
        self.NSpins = NSpins              # number of spin sims in each voxel
        
        self.PD = None                        # proton density tensor (NVox,)
        self.T1 = None                          # T1 relaxation times (NVox,)
        self.T2 = None                          # T2 relaxation times (NVox,)
        self.omega = None                        # spin off-resonance (NSpins,)
        
        self.M0 = None     # initial magnetization state (NSpins,NRep,NVox,4)
        self.M = None       # curent magnetization state (NSpins,NRep,NVox,4)
        
        # aux
        self.R2 = None
        self.use_gpu = use_gpu
        
    # device setter
    def setdevice(self,x):
        if self.use_gpu:
            x = x.cuda(0)
            
        return x
    
    # get magnitude image
    def magimg(self, x):
      return np.sqrt(np.sum(np.abs(x)**2,2))    
    
    def set_system(self, input_array=None):
        
        if np.any(input_array < 0):
            throw('ERROR: SpinSystem: set_system: some of the values in <input_array> are smaller than zero')
            
        if input_array.shape[-1] == 3:                    # full specificiation
            PD = input_array[...,0]
            T1 = input_array[...,1]
            T2 = input_array[...,2]
        else:                                                         # only PD
            PD = input_array
            T1 = np.ones((self.NVox,), dtype=np.float32)*4
            T2 = np.ones((self.NVox,), dtype=np.float32)*2            
        
        PD = torch.from_numpy(PD.reshape([self.NVox])).float()
        T1 = torch.from_numpy(T1.reshape([self.NVox])).float()
        T2 = torch.from_numpy(T2.reshape([self.NVox])).float()   
            
        # set NSpins offresonance (from R2)
        factor = (0*1e0*np.pi/180) / self.NSpins
        omega = torch.from_numpy(factor*np.random.rand(self.NSpins,self.NVox).reshape([self.NSpins,self.NVox])).float()
        
        self.T1 = self.setdevice(T1)
        self.T2 = self.setdevice(T2)
        self.PD = self.setdevice(PD)
        self.omega = self.setdevice(omega)
        
    def set_initial_magnetization(self):
        
        M0 = torch.zeros((self.NSpins,1,self.NVox,4), dtype=torch.float32)
        M0 = self.setdevice(M0)
        
        # set initial longitudinal magnetization value
        M0[:,:,:,2:] = 1
        M0[:,:,:,2:] = M0[:,:,:,2:] * self.PD.view([self.NVox,1])    # weight by proton density
        
        M = M0.clone().view([self.NSpins,1,self.NVox,4,1])
        
        self.M0 = M0
        self.M = self.setdevice(M)
        
        

# child SpinSystem class for batch image processing        
# variation for supervised learning
class SpinSystem_batched(SpinSystem):
    
    def __init__(self,sz,NVox,NSpins,batch_size,use_gpu):
        
        super(SpinSystem, self).__init__(sz,NVox,NSpins,use_gpu)
        self.batch_size = batch_size
        
    def set_system(self, input_array):
        
        if np.any(input_array < 0):
            throw('ERROR: SpinSystem: set_system: some of the values in <input_array> are smaller than zero')
            
        if input_array.shape[-1] == 3:                    # full specificiation
            PD = input_array[...,0]
            T1 = input_array[...,1]
            T2 = input_array[...,2]
        else:                                                         # only PD
            PD = input_array
            T1 = np.ones((self.batch_size, self.NVox,), dtype=np.float32)*4
            T2 = np.ones((self.batch_size, self.NVox,), dtype=np.float32)*2            
        
        PD = torch.from_numpy(PD.reshape([self.batch_size, self.NVox])).float()
        T1 = torch.from_numpy(T1.reshape([self.batch_size, self.NVox])).float()
        T2 = torch.from_numpy(T2.reshape([self.batch_size, self.NVox])).float()   
            
        # set NSpins offresonance (from R2)
        factor = 0
        omega = torch.from_numpy(factor*np.random.rand(self.NSpins,self.NVox).reshape([self.NSpins,self.NVox])).float()
        
        self.T1 = self.setdevice(T1)
        self.T2 = self.setdevice(T2)
        self.PD = self.setdevice(PD)
        self.omega = self.setdevice(omega)
        
    def set_initial_magnetization(self):
        
        M0 = torch.zeros((self.batch_size,self.NSpins,1,self.NVox,4), dtype=torch.float32)
        M0 = self.setdevice(M0)
        
        # set initial longitudinal magnetization value
        M0[:,:,:,:,2:] = 1
        M0[:,:,:,:,2:] = M0[:,:,:,:,2:] * self.PD.view([self.batch_size,1,1,self.NVox,1])    # weight by proton density
        
        M = M0.clone().view([self.batch_size,self.NSpins,1,self.NVox,4,1])
        
        self.M0 = M0
        self.M = self.setdevice(M)
        
        
        
