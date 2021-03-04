"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""
experiment_id = 'solA02_spinecho'
sequence_class = "super"
experiment_description = """
SE or 1 D imaging / spectroscopy
"""
excercise = """
A02.1. increase Nreps to 4
A02.2. add a second flip event to create an echo. which flip angle generates the highest echo?
A02.3. try to find a way so that the echo forms in the middle of the acquisition phase
A02.4. try to adjust rf phase / adc rotation to get only real and positive echoes
A02.5. uncomment FITTING BLOCK, fit signal, what is the decay rate of the envelope alter R2star
A02.6. increase Nreps and check the fit result
A02.7. decrease R2star and check the influence on the fit result (realistic R2star=30)
A02.8. remove again refocusing pulses, and look at the FID again
"""
#%%
#matplotlib.pyplot.close(fig=None)
#%%
import os, sys
import numpy as np
import scipy
import scipy.io
from  scipy import ndimage
from scipy.optimize import curve_fit
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
sz = np.array([4,4])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                      # number of total repetitions
NRep = 16                                    # number of total repetitions
szread=128
NEvnt = szread + 5 + 2                               # number of events F/R/P
NSpins = 26**2                               # number of spin sims in each voxel
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

# or (ii) set phantom  manually
phantom = np.zeros((sz[0],sz[1],5), dtype=np.float32); 
phantom[1,1,:]=np.array([1, 1, 0.1, 0, 1]) # third dimension: PD, T1 T2 dB0 rB1

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

spins.set_system(phantom,R2dash=250.0)  # set phantom variables with overall constant R2' = 1/T2'  (R2*=R2+R2')

## end of S1: Init spin system and phantom ::: #####################################

#############################################################################
## S2: Init scanner system ::: #####################################
scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,NEvnt,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)
#scanner.set_B1plus(phantom[:,:,4])  # use as defined in phantom
scanner.set_B1plus(1)               # overwriet with homogeneous excitation

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
rf_event[3,0,0] = 90*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
rf_event[3,0,1] = 90*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
rf_event[3,1:,0] = 180*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
rf_event = setdevice(rf_event)
scanner.init_flip_tensor_holder()    
scanner.set_flip_tensor_withB1plus(rf_event)
# rotate ADC according to excitation phase
rfsign = ((rf_event[3,:,0]) < 0).float()
scanner.set_ADC_rot_tensor(-rf_event[3,0,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.08*1e-3*np.ones((NEvnt,NRep))).float()
event_time[:,0] =  0.04*1e-3
event_time = setdevice(event_time)

# gradient-driver precession
# Cartesian encoding
gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
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
# targetSeq.print_seq(plotsize=[12,9], time_axis=1)     
      
fig=plt.figure("""signals""")
ax1=plt.subplot(131)
ax=plt.plot(tonumpy(scanner.signal[0,:,:,0,0]).transpose().ravel(),label='real')
plt.plot(tonumpy(scanner.signal[0,:,:,1,0]).transpose().ravel(),label='imag')
plt.title('signal')
plt.legend()
plt.ion()


plt.show()
                        
#%%  FITTING BLOCK
tfull=np.cumsum(tonumpy(event_time).transpose().ravel())
yfull=tonumpy(scanner.signal[0,:,:,0,0]).transpose().ravel()
idx=tonumpy(scanner.signal[0,:,:,0,0]).transpose().argmax(1)
idx=idx + np.linspace(0,(NRep-1)*len(event_time[:,0]),NRep,dtype=np.int64)
t=tfull[idx]
y=yfull[idx]
def fit_func(t, a, R,c):
    return a*np.exp(-R*t) + c   

p=scipy.optimize.curve_fit(fit_func,t,y,p0=(np.mean(y), 1,np.min(y)))
print(p[0][1])

fig=plt.figure("""fit""")
ax1=plt.subplot(131)
ax=plt.plot(tfull,yfull,label='fulldata')
ax=plt.plot(t,y,label='data')
plt.plot(t,fit_func(t,p[0][0],p[0][1],p[0][2]),label="f={:.2}*exp(-{:.2}*t)+{:.2}".format(p[0][0], p[0][1],p[0][2]))
plt.title('fit')
plt.legend()
plt.ion()


plt.show()
            