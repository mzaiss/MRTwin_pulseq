"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""
experiment_id = 'solA06_gradientecho_freq_enc_FFT'
sequence_class = "super"
experiment_description = """
GRE or 1 D imaging
"""
excercise = """
this file starts from solA05. we want now to create the 1D - fourier transform of the signal to get a 1D image.
This is also called frequency encoding.
A06.1. to speed things up: use scanner.forward_fast(spins, event_time), NRep =1
A06.2  isolate only the signal when ADC is on: spectrum = tonumpy(scanner.signal[0,adc_mask.flatten()!=0,:,:2,0].clone()) 
A06.3. to separate different frequencies, perform a fourier transform of the signal. Learn from sim02_fft.py
A06.4. compare your result to the upsampled phantom:
    
        plt.subplot(313); plt.title('phantom projection')
        t = cv2.resize(phantom[:,:,0], dsize=(sz[0],szread), interpolation=cv2.INTER_NEAREST)
        t=np.flipud(np.roll(t,-szread//sz[1]//2+1,0))  # this is needed due to the oversampling of the phantom, szread>sz
        plt.plot(np.sum(t,axis=1).flatten('F'),label='real')
        plt.show()  

A06.5. what happens if you change y to x gradient encoding
A06.6. what happens if you use both x and y gradients?
A06.7. decrease szread.
"""
print(excercise)
#%%
#matplotlib.pyplot.close(fig=None)
#%%
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

def phaseimg(x):
    return np.angle(1j*x[:,:,1]+x[:,:,0])

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
sz = np.array([12,12])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                      # number of total repetitions
NRep = 1                                  # number of total repetitions
szread=128
NEvnt = szread + 5 + 2                               # number of events F/R/P
NSpins = 16**2                               # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*1e-3                          # additive Gaussian noise std
kill_transverse = False                     #
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*szread


#############################################################################
## S0: define image and simulation settings::: #####################################
sz = np.array([12,12])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                      # number of total repetitions
NRep = 8                                  # number of total repetitions
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

# either (i) load phantom (third dimension: PD, T1 T2 dB0 rB1)
phantom = spins.get_phantom(sz[0],sz[1],type='object1')  # type='object1' or 'brain1'

# or (ii) set phantom  manually to single pixel phantom
phantom = np.zeros((sz[0],sz[1],5), dtype=np.float32); 
phantom[6,7,:]=np.array([1, 1, 0.1, 0,0])  # pixel 1,  third dimension: PD, T1 T2 dB0 rB1
phantom[4,3:5,:]=np.array([0.5, 1, 0.1, 0,0]) # position 2 (two pixels)


# adjust phantom
phantom[:,:,1] *= 1 # Tweak T1
phantom[:,:,2] *= 1 # Tweak T2
phantom[:,:,3] += 0 # Tweak dB0
phantom[:,:,4] *= 1 # Tweak rB1

if 1: # switch on for plot
    plt.figure("""phantom"""); plt.clf();  param=['PD','T1 [s]','T2 [s]','dB0 [Hz]','rB1 [rel.]']
    for i in range(5):
        plt.subplot(151+i), plt.title(param[i])
        ax=plt.imshow(phantom[:,:,i], interpolation='none')
        fig = plt.gcf(); fig.colorbar(ax) 
    fig.set_size_inches(18, 3); plt.show()

spins.set_system(phantom,R2dash=30.0)  # set phantom variables with overall constant R2' = 1/T2'  (R2*=R2+R2')

## end of S1: Init spin system and phantom ::: #####################################

#############################################################################
## S2: Init scanner system ::: #####################################
scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,NEvnt,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)
#scanner.set_B1plus(phantom[:,:,4])  # use as defined in phantom
scanner.set_B1plus(1)                 # overwrite with homogeneous excitation       

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
rf_event[3,:,0] = 90*np.pi/180  # 90deg excitation now for every rep
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
gradm_event[4,:,1] = -0.5*szread
gradm_event[5:-2,:,1] = 1
#gradm_event[4,:,0] = -0.5*szread *0.7
#gradm_event[5:-2,:,0] = 1 *0.7
gradm_event = setdevice(gradm_event)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(gradm_event,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes
## end S3: MR sequence definition ::: #####################################



#############################################################################
## S4: MR simulation forward process ::: #####################################
scanner.init_signal()
scanner.forward(spins, event_time)

  
# sequence and signal plotting
targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,scanner.signal)
#targetSeq.print_seq_pic(True,plotsize=[12,9])
#targetSeq.print_seq(plotsize=[12,9])
targetSeq.print_seq(plotsize=[12,9], time_axis=1)


#%% ############################################################################
## S5: MR reconstruction of signal ::: #####################################
fig=plt.figure("""Fourier Transform""")
plt.subplot(311); plt.title('ADC signal')
spectrum = tonumpy(scanner.signal[0,adc_mask.flatten()!=0,:,:2,0].clone()) 
spectrum = spectrum[:,:,0]+spectrum[:,:,1]*1j # get all ADC signals as complex numpy array
plt.plot(np.real(spectrum).flatten('F'),label='real')
plt.plot(np.imag(spectrum).flatten('F'),label='imag')
major_ticks = np.arange(0, szread*NRep, szread) # this adds ticks at the correct position szread
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()

space = np.zeros_like(spectrum)

plt.subplot(312); plt.title('FFT')
plt.plot(np.abs(space.flatten('F')))
plt.plot(np.imag(space.flatten('F')))
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()
           
