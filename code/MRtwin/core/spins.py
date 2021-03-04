import numpy as np
import torch
import torchvision
import cv2
import scipy
import scipy.io
from  scipy import ndimage
import matplotlib.pyplot as plt

def throw(msg):
    class ExecutionControl(Exception): pass
    raise ExecutionControl(msg)
    
# WHAT we measure
class SpinSystem():
    def __init__(self,sz,NVox,NSpins,use_gpu,double_precision=False):
        
        self.sz = sz                                             # image size
        self.NVox = sz[0]*sz[1]                                 # voxel count
        self.NSpins = NSpins              # number of spin sims in each voxel
        
        self.PD = None                        # proton density tensor (NVox,)
        self.T1 = None                          # T1 relaxation times (NVox,)
        self.T2 = None                          # T2 relaxation times (NVox,)
        self.omega = None                        # spin off-resonance (NSpins,)
        
        self.M0 = None     # initial magnetization state (NSpins,NRep,NVox,3)
        self.M = None       # curent magnetization state (NSpins,NRep,NVox,3)
        self.M_history = []
        # aux
        self.R2 = None
        self.use_gpu = use_gpu
        self.double_precision = double_precision
        
    # device setter
    def setdevice(self,x):
        if self.double_precision:
            x = x.double()
        else:
            x = x.float()
        if self.use_gpu > 0:
            x = x.cuda(self.use_gpu-1)
            
        return x
    
        # device setter
    def get_phantom(self,szx,szy,type='object1',interpolation=cv2.INTER_CUBIC, plot=False): # type='object1'
        if type=='object1':
            real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
        elif type=='brain1':
            real_phantom = scipy.io.loadmat('../../data/numerical_brain_cropped.mat')['cropped_brain']
        
        real_phantom_resized = np.zeros((szx,szy,5), dtype=np.float32)
        cutoff = 1e-12
        for i in range(5):
            t = cv2.resize(real_phantom[:,:,i], dsize=(szx,szy), interpolation=interpolation)
            if i == 0:
                t[t < 0] = 0
            elif i == 1 or i == 2:
                t[t < cutoff] = cutoff        
            real_phantom_resized[:,:,i] = t
            
        if plot==True:
            plt.figure("""phantom""")
            param=['PD','T1','T2','dB0','rB1']
            for i in range(5):
                plt.subplot(151+i), plt.title(param[i])
                ax=plt.imshow(real_phantom_resized[:,:,i], interpolation='none')
                fig = plt.gcf()
                fig.colorbar(ax) 
            fig.set_size_inches(18, 3)
            plt.show()
            
        return real_phantom_resized

    def get_phantom_torch(self,szx,szy,type='object1',interpolation=cv2.INTER_CUBIC, plot=False): # type='object1'
        if type=='object1':
            real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
        elif type=='brain1':
            real_phantom = scipy.io.loadmat('../../data/numerical_brain_cropped.mat')['cropped_brain']
        
        real_phantom_resized = torch.from_numpy(real_phantom.transpose(2,0,1))
        real_phantom_resized = torchvision.transforms.Resize((szx,szy),interpolation=3)(real_phantom_resized)
        cutoff = 1e-12
        real_phantom_resized[0,real_phantom_resized[0]<0]=0
        real_phantom_resized[1,real_phantom_resized[1]<cutoff]=cutoff
        real_phantom_resized[2,real_phantom_resized[2]<cutoff]=cutoff
        real_phantom_resized = real_phantom_resized.detach().cpu().numpy().transpose(1,2,0)
            
        if plot==True:
            plt.figure("""phantom""")
            param=['PD','T1','T2','dB0','rB1']
            for i in range(5):
                plt.subplot(151+i), plt.title(param[i])
                ax=plt.imshow(real_phantom_resized[:,:,i], interpolation='none')
                fig = plt.gcf()
                fig.colorbar(ax) 
            fig.set_size_inches(18, 3)
            plt.show()
            
        return real_phantom_resized
       
    def set_system(self, input_array=None, R2dash=30.0 ):
        
        if np.any(input_array[:,:,:3] < 0):
            throw('ERROR: SpinSystem: set_system: some of the values in <input_array> are smaller than zero')
            
        if input_array.shape[2] > 2:                    # full specificiation
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
            
        # set NSpins offresonance (from R2dash)
        factor = (0*1e0*np.pi/180) / self.NSpins
        omega = torch.from_numpy(factor*np.random.rand(self.NSpins,self.NVox).reshape([self.NSpins,self.NVox])).float()
        

        omega = np.linspace(0,1,self.NSpins) - 0.5   # cutoff might bee needed for opt.
        omega = np.expand_dims(omega[:],1).repeat(self.NVox, axis=1)
        omega*=0.99 # cutoff large freqs
        omega = R2dash * np.tan ( np.pi  * omega)
        omega = torch.from_numpy(omega.reshape([self.NSpins,self.NVox])).float()
        
        B0inhomo = torch.zeros((self.NVox)).float()
        
        # also susceptibilities
        if input_array.shape[2] > 3:
            B0inhomo = input_array[...,3]
            B0inhomo = torch.from_numpy(B0inhomo.reshape([self.NVox])).float()
            
        # find and store in mask locations with zero PD
        PD0_mask = PD.reshape([self.sz[0],self.sz[1]]) > 1e-6

        self.PD0_mask = self.setdevice(PD0_mask).byte()
        self.T1 = self.setdevice(T1)
        self.T2 = self.setdevice(T2)
        self.PD = self.setdevice(PD)
        self.omega = self.setdevice(omega)
        self.B0inhomo = self.setdevice(B0inhomo)
        
    def set_initial_magnetization(self):
        
        M0 = torch.zeros((self.NSpins,1,self.NVox,3), dtype=torch.float32)
        M0 = self.setdevice(M0)
        
        # set initial longitudinal magnetization value
        M0[:,:,:,2] = 1
        M0[:,:,:,2] = M0[:,:,:,2] * self.PD.view([1,1,self.NVox])    # weight by proton density
        
        M = M0.clone().view([self.NSpins,1,self.NVox,3,1])
        
        MZ0 = M0[:,:,:,2].clone()
        
        self.M0 = M0
        self.M = self.setdevice(M)
        self.MZ0 = self.setdevice(MZ0)
        
        

# child SpinSystem class for batch image processing        
# variation for supervised learning
class SpinSystem_batched(SpinSystem):
      
    def __init__(self,sz,NVox,NSpins,batch_size,use_gpu):
        super(SpinSystem_batched, self).__init__(sz,NVox,NSpins,use_gpu)
        self.batch_size = batch_size
        
    def set_system(self, input_array=None):
        
        if np.any(input_array[:,:,:,:3] < 0):
            throw('ERROR: SpinSystem: set_system: some of the values in <input_array> are smaller than zero')
            
        PD = input_array[...,0]
        T1 = input_array[...,1]
        T2 = input_array[...,2]
        
        PD = torch.from_numpy(PD.reshape([self.batch_size,self.NVox])).float()
        T1 = torch.from_numpy(T1.reshape([self.batch_size,self.NVox])).float()
        T2 = torch.from_numpy(T2.reshape([self.batch_size,self.NVox])).float()   
            
        B0inhomo = torch.zeros((self.batch_size,self.NVox)).float()
        
        # also susceptibilities
        if input_array.shape[2] > 3:
            B0inhomo = input_array[...,3]
            B0inhomo = torch.from_numpy(B0inhomo.reshape([self.batch_size,self.NVox])).float()
            
        # find and store in mask locations with zero PD
        PD0_mask = PD.reshape([self.batch_size,self.sz[0],self.sz[1]]) > 1e-6
        
        omega = torch.from_numpy(0*np.random.rand(self.batch_size,self.NSpins,self.NVox).reshape([self.batch_size,self.NSpins,self.NVox])).float()
        

        self.PD0_mask = self.setdevice(PD0_mask).byte()
        self.T1 = self.setdevice(T1)
        self.T2 = self.setdevice(T2)
        self.PD = self.setdevice(PD)
        self.B0inhomo = self.setdevice(B0inhomo)
        self.omega = self.setdevice(omega)
        
    def set_initial_magnetization(self):
        
        M0 = torch.zeros((self.batch_size,self.NSpins,1,self.NVox,3), dtype=torch.float32)
        M0 = self.setdevice(M0)
        
        # set initial longitudinal magnetization value
        M0[:,:,:,:,2] = 1
        M0[:,:,:,:,2] = M0[:,:,:,:,2] * self.PD.view([self.batch_size,1,1,self.NVox])    # weight by proton density
        
        M = M0.clone().view([self.batch_size,self.NSpins,1,self.NVox,3,1])
        
        MZ0 = M0[:,:,:,:,2].clone()
        
        self.M0 = M0
        self.M = self.setdevice(M)
        self.MZ0 = self.setdevice(MZ0)        
        


        
