"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""
experiment_id = 'exA07_gradientecho_FFT_2'
sequence_class = "gre_dream"
experiment_description = """
this is a gre after a 80 degree prepulse
"""
excercise = """
inrease the NSPINS until image quality is good.

"""
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
import core.opt_helper
import core.target_seq_holder
import core.FID_normscan
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

from importlib import reload
reload(core.scanner)

double_precision = False
do_scanner_query = False

do_voxel_rand_ramp_distr = True
do_voxel_rand_r2_distr = False

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
sz = np.array([24,24])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                      # number of total repetitions
szread=24
T = szread + 5 + 2                               # number of events F/R/P
NSpins = 30**2                               # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*1e-3                          # additive Gaussian noise std
kill_transverse = False                     #
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*szread

#############################################################################
## S1: Init spin system and phantom::: #####################################
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)

cutoff = 1e-12
real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
#real_phantom = scipy.io.loadmat('../../data/numerical_brain_cropped.mat')['cropped_brain']

real_phantom_resized = np.zeros((sz[0],sz[1],5), dtype=np.float32)
for i in range(5):
    t = cv2.resize(real_phantom[:,:,i], dsize=(sz[0],sz[1]), interpolation=cv2.INTER_NEAREST)
    if i == 0:
        t[t < 0] = 0
    elif i == 1 or i == 2:
        t[t < cutoff] = cutoff        
    real_phantom_resized[:,:,i] = t
    
#real_phantom_resized = np.zeros((sz[0],sz[1],5), dtype=np.float32)
#real_phantom_resized[6,6,:]=np.array([1.0, 1, 0.1, 0,0])
#real_phantom_resized[2,3,:]=np.array([0.5,    1, 0.1, 0,0])
    
real_phantom_resized[:,:,1] *= 1 # Tweak T1
real_phantom_resized[:,:,2] *= 1 # Tweak T2
real_phantom_resized[:,:,3] *= 3 # Tweak dB0
real_phantom_resized[:,:,4] *= 1 # Tweak rB1

spins.set_system(real_phantom_resized)

if 0:
    plt.figure("""phantom""")
    param=['PD','T1','T2','dB0','rB1']
    for i in range(5):
        plt.subplot(151+i), plt.title(param[i])
        ax=plt.imshow(real_phantom_resized[:,:,i], interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax) 
    fig.set_size_inches(18, 3)
    plt.show()
   
#begin nspins with R2* = 1/T2*
R2star = 30.0
if not do_voxel_rand_r2_distr:
    omega = np.linspace(0,1,NSpins) - 0.5   # cutoff might bee needed for opt.
    omega = np.expand_dims(omega[:],1).repeat(NVox, axis=1)
else:
    omega = np.random.rand(NSpins,NVox) - 0.5   # cutoff might bee needed for opt.
    
omega*=0.99 # cutoff large freqs
omega = R2star * np.tan ( np.pi  * omega)
spins.omega = torch.from_numpy(omega.reshape([NSpins,NVox])).float()
spins.omega = setdevice(spins.omega)
## end of S1: Init spin system and phantom ::: #####################################


#############################################################################
## S2: Init scanner system ::: #####################################
scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision,do_voxel_rand_ramp_distr=do_voxel_rand_ramp_distr,do_voxel_rand_r2_distr=do_voxel_rand_r2_distr)

B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([scanner.NCoils, scanner.NVox]))
B1plus[B1plus == 0] = 1    # set b1+ to one, where we dont have phantom measurements
B1plus[:] = 1
scanner.B1plus = setdevice(B1plus)

#############################################################################
## S3: MR sequence definition ::: #####################################
# begin sequence definition
# allow for extra events (pulses, relaxation and spoiling) in the first five and last two events (after last readout event)
adc_mask = torch.from_numpy(np.ones((T,1))).float()
adc_mask[:5]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

# RF events: flips and phases
flips = torch.zeros((T,NRep,2), dtype=torch.float32)
flips[3,:,0] = 5*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
flips[0,0,0] = 80*np.pi/180 
flips = setdevice(flips)
scanner.init_flip_tensor_holder()    
scanner.set_flip_tensor_withB1plus(flips)
# rotate ADC according to excitation phase
rfsign = ((flips[3,:,0]) < 0).float()
scanner.set_ADC_rot_tensor(-flips[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.08*1e-3*np.ones((scanner.T,scanner.NRep))).float()
event_time[:,0] =  0.08*1e-3
event_time[-1,:] =  0.01
event_time = setdevice(event_time)

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32)
grad_moms[1,:,:] = 30*szread
grad_moms[1,:,0] = 0
grad_moms[4,:,1] = -0.5*szread
grad_moms[5:-2,:,1] = 1
grad_moms[4,:,0] = torch.arange(0,sz[0],1)-sz[0]/2
grad_moms[-2,:,1] = -0.5*szread
grad_moms[-2,:,0] = -grad_moms[4,:,0] 
grad_moms = setdevice(grad_moms)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes
## end S3: MR sequence definition ::: #####################################



#############################################################################
## S4: MR simulation forward process ::: #####################################
scanner.init_signal()
scanner.forward_fast(spins, event_time)
#scanner.forward(spins, event_time)
  
#%% ############################################################################
## S5: MR reconstruction of signal ::: #####################################

plt.subplot(311)
spectrum = tonumpy(scanner.signal[0,adc_mask.flatten()!=0,:,:2,0].clone()) # get all complex DC signals
spectrum = spectrum[:,:,0]+spectrum[:,:,1]*1j # generate complex signal
plt.plot(np.transpose(np.real(spectrum)).flatten(),label='real')
plt.plot(np.transpose(np.imag(spectrum)).flatten(),label='imag')
major_ticks = np.arange(0, szread*NRep, szread)
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()
space = np.zeros_like(spectrum)
spectrum = np.roll(spectrum,szread//2,axis=0)
spectrum = np.roll(spectrum,NRep//2,axis=1)

for i in range(0,NRep):
    space[:,i] = np.fft.ifft(spectrum[:,i])
space = np.fft.ifft2(spectrum)
# fftshift
space= np.roll(space,szread//2-1,axis=0)
space = np.roll(space,NRep//2-1,axis=1)
plt.subplot(312)
plt.plot(np.abs(np.transpose(space).ravel()))
plt.plot(np.imag(np.transpose(space).ravel()))
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()
            
plt.subplot(3,5,11)
plt.imshow(real_phantom_resized[:,:,0], interpolation='none')

space = np.fft.ifft2(spectrum)
space = np.roll(space,szread//2-1,axis=0)
space = np.roll(space,NRep//2-1,axis=1)
space = np.flip(space,(0,1))
        
plt.subplot(3,5,12)
plt.imshow(np.abs(space), interpolation='none',aspect = sz[0]/szread)
plt.subplot(3,5,13)
mask=(np.abs(space)>0.2*np.max(np.abs(space)))
plt.imshow(np.angle(space)*mask, interpolation='none',aspect = sz[0]/szread)
#plt.imshow(np.imag(space), interpolation='none')
scanner.adjoint()
plt.subplot(3,5,14)
plt.imshow(magimg(tonumpy(scanner.reco).reshape([sz[0],sz[1],2])), interpolation='none')
plt.subplot(3,5,15)
plt.imshow(phaseimg(tonumpy(scanner.reco).reshape([sz[0],sz[1],2]))*np.transpose(np.flip(mask)), interpolation='none')

print(phaseimg(tonumpy(scanner.reco).reshape([sz[0],sz[1],2]))[10,10])
plt.show()                     
#%% FITTING BLOCK
#tfull=np.cumsum(tonumpy(event_time).transpose().ravel())
#yfull=tonumpy(scanner.signal[0,:,:,0,0]).transpose().ravel()
##yfull=tonumpy(scanner.signal[0,:,:,1,0]).transpose().ravel()
#idx=tonumpy(scanner.signal[0,:,:,0,0]).transpose().argmax(1)
#idx=idx + np.linspace(0,(NRep-1)*len(event_time[:,0]),NRep,dtype=np.int64)
#t=tfull[idx]
#y=yfull[idx]
#def fit_func(t, a, R,c):
#    return a*np.exp(-R*t) + c   
#
#p=scipy.optimize.curve_fit(fit_func,t,y,p0=(np.mean(y), 1,np.min(y)))
#print(p[0][1])
#
#fig=plt.figure("""fit""")
#ax1=plt.subplot(131)
#ax=plt.plot(tfull,yfull,label='fulldata')
#ax=plt.plot(t,y,label='data')
#plt.plot(t,fit_func(t,p[0][0],p[0][1],p[0][2]),label="f={:.2}*exp(-{:.2}*t)+{:.2}".format(p[0][0], p[0][1],p[0][2]))
#plt.title('fit')
#plt.legend()
#plt.ion()
#
#fig.set_size_inches(64, 7)
#plt.show()
#            