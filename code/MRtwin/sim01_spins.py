"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""
experiment_id = 'sim01_spins'
sequence_class = "gre_dream"
experiment_description = """
check simulation setup
"""
excercise = """
"""

import os, sys
import numpy as np
import scipy
import scipy.io
from  scipy import ndimage
import torch
import cv2
import matplotlib.pyplot as plt
from torch import optim
import core.spins
import core.scanner
import core.nnreco
import core.target_seq_holder
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


from importlib import reload
reload(core.scanner)

double_precision = False
do_scanner_query = False

use_gpu = 1
gpu_dev = 0

if sys.platform != 'linux':
    use_gpu = 0
    gpu_dev = 0
print(experiment_id)    
print('use_gpu = ' +str(use_gpu)) 

# NRMSE error function
def e(gt,x):
    return 100*np.linalg.norm((gt-x).ravel())/np.linalg.norm(gt.ravel())
    
# torch to numpy
def tonumpy(x):
    return x.detach().cpu().numpy()

# get magnitude image
def magimg(x):
  return np.sqrt(np.sum(np.abs(x)**2,2))

def magimg_torch(x):
  return torch.sqrt(torch.sum(torch.abs(x)**2,1))

def tomag_torch(x):
    return torch.sqrt(torch.sum(torch.abs(x)**2,-1))

# device setter
def setdevice(x):
    if double_precision:
        x = x.double()
    else:
        x = x.float()
    if use_gpu:
        x = x.cuda(gpu_dev)    
    return x 

#############################################################################
## S0: define image and simulation settings::: #####################################
sz = np.array([16,16])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                      # number of total repetitions
NRep = 4                                  # number of total repetitions
szread=128
NEvnt = szread + 5 + 2                               # number of events F/R/P
NSpins = 16**2                               # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*1e-3                          # additive Gaussian noise std
kill_transverse = False                     #
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*sz[1]

#############################################################################
## S1: Init spin system and phantom::: #####################################
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)

cutoff = 1e-12
#real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
#real_phantom = scipy.io.loadmat('../../data/numerical_brain_cropped.mat')['cropped_brain']

phantom = np.zeros((sz[0],sz[1],5), dtype=np.float32)
phantom[8,8,:]=np.array([1, 1, 0.1, 0,0])
#phantom[7,7,:]=np.array([0.25, 1, 0.1, 0,0])
    
phantom[:,:,1] *= 1 # Tweak T1
phantom[:,:,2] *= 1 # Tweak T2
phantom[:,:,3] += 0 # Tweak dB0
phantom[:,:,4] *= 1 # Tweak rB1

spins.set_system(phantom)

if 0:
    plt.figure("""phantom""")
    param=['PD','T1','T2','dB0','rB1']
    for i in range(5):
        plt.subplot(151+i), plt.title(param[i])
        ax=plt.imshow(phantom[:,:,i], interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax) 
    fig.set_size_inches(18, 3)
    plt.show()
   
#begin nspins with R2* = 1/T2*
R2star = 0.0
omega = np.linspace(0,1,NSpins) - 0.5   # cutoff might bee needed for opt.
omega = np.expand_dims(omega[:],1).repeat(NVox, axis=1)
omega*=0.99 # cutoff large freqs
omega = R2star * np.tan ( np.pi  * omega)
spins.omega = torch.from_numpy(omega.reshape([NSpins,NVox])).float()
spins.omega = setdevice(spins.omega)
## end of S1: Init spin system and phantom ::: #####################################


#############################################################################
## S2: Init scanner system ::: #####################################
scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,NEvnt,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)

B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(phantom[:,:,4].reshape([scanner.NCoils, scanner.NVox]))
B1plus[B1plus == 0] = 1    # set b1+ to one, where we dont have phantom measurements
B1plus[:] = 1
scanner.B1plus = setdevice(B1plus)

#############################################################################
## S3: MR sequence definition ::: #####################################
# begin sequence definition
# allow for extra events (pulses, relaxation and spoiling) in the first five and last two events (after last readout event)
adc_mask = torch.from_numpy(np.ones((NEvnt,1))).float()
adc_mask[:5]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

# RF events: rf_event and phases
rf_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
rf_event[3,:,0] = 90*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
rf_event = setdevice(rf_event)
scanner.init_flip_tensor_holder()    
scanner.set_flip_tensor_withB1plus(rf_event)
# rotate ADC according to excitation phase
rfsign = ((rf_event[3,:,0]) < 0).float()
scanner.set_ADC_rot_tensor(-rf_event[3,0,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.08*1e-3*np.ones((NEvnt,NRep))).float()
event_time[:,0] =  0.08*1e-3
event_time[-1,:] =  5
event_time = setdevice(event_time)

# gradient-driver precession
# Cartesian encoding
gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
#gradm_event[4,:,0] = -0.5*szread
gradm_event[5:-2,:,0] = 1
gradm_event = setdevice(gradm_event)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(gradm_event,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes
## end S3: MR sequence definition ::: #####################################


#############################################################################
## S4: MR simulation forward process ::: #####################################
scanner.init_signal()
#scanner.forward(spins, event_time)
fig=plt.figure("""intravoxel_dephasing_ramp""")
plt.imshow(scanner.intravoxel_dephasing_ramp[:,0].view(int(np.sqrt(NSpins)),int(np.sqrt(NSpins))))

# %% ### # from   def init_intravoxel_dephasing_ramps(self):
NSpins=8**2
dim = setdevice(torch.sqrt(torch.tensor(NSpins).float()))
        
off = 1 / dim
if dim == torch.floor(dim):
    xv, yv = torch.meshgrid([torch.linspace(-1+off,1-off,dim.int()), torch.linspace(-1+off,1-off,dim.int())])
    Rx1= torch.randn(torch.Size([dim.int()//2,dim.int()]))*off
    Rx2=-torch.flip(Rx1, [0])
    Rx= torch.cat((Rx1, Rx2),0)
    
    Ry1= torch.randn(torch.Size([dim.int(),dim.int()//2]))*off
    Ry2=-torch.flip(Ry1, [1])
    Ry= torch.cat((Ry1, Ry2),1)
                
    xv = xv + Rx
    yv = yv + Ry
#    yv = yv + (torch.randn(yv.shape))*off
plt.subplot(221)
plt.imshow(Rx); plt.title('Rx')
plt.subplot(222)
plt.imshow(Ry); plt.title('Ry')
plt.show()
plt.subplot(212)
plt.plot(xv.flatten(),yv.flatten(),'x'); plt.title('distribution and its center of mass')
plt.plot(xv.flatten().mean(),yv.flatten().mean(),'d')
fig.set_size_inches(64, 7)
plt.show()
#
#fig=plt.figure("""seq and signal""")
#plt.subplot(311)
#ax=plt.plot(np.tile(tonumpy(adc_mask),NRep).transpose().ravel(),'.',label='ADC')
#ax=plt.plot(tonumpy(event_time).transpose().ravel(),'.',label='time')
#ax=plt.plot(tonumpy(rf_event[:,:,0]).transpose().ravel(),label='RF')
#plt.legend()
#plt.subplot(312)
#ax=plt.plot(tonumpy(gradm_event[:,:,0]).transpose().ravel(),label='gx')
#ax=plt.plot(tonumpy(gradm_event[:,:,1]).transpose().ravel(),label='gy')
#plt.legend()
#plt.subplot(313)
#ax=plt.plot(tonumpy(scanner.signal[0,:,:,0,0]).transpose().ravel(),label='real')
#plt.plot(tonumpy(scanner.signal[0,:,:,1,0]).transpose().ravel(),label='imag')
#plt.title('signal')
#plt.legend()
#plt.ion()
#
#fig.set_size_inches(64, 7)
#plt.show()
                        
            