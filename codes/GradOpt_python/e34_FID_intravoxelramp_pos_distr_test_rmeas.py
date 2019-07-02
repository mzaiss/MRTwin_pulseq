#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
"""

experiment_id = 't01_tgtGRESP_tsk_GRESP_no_grad_noflip_kspaceloss_new'
sequence_class = "GRE"
experiment_description = """
tgt FLASHspoiled_relax20ms, with spoilers and random phase cycling
task find all grads except read ADC grads
opt: SARloss, kloss, 

this is the same as e05_tgtGRE_tskGREnogspoil.py, but now with more automatic restarting
and high initial learning rate
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
import core.opt_helper
import core.target_seq_holder

from importlib import reload
reload(core.scanner)

print('32x float forwardfast oS')

double_precision = False
use_supermem = False
do_scanner_query = True

use_gpu = 1
gpu_dev = 0

if sys.platform != 'linux':
    use_gpu = 0
    gpu_dev = 0

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
def phaseimg(x):
    return np.angle(1j*x[:,:,1]+x[:,:,0])

# device setter
def setdevice(x):
    if double_precision:
        x = x.double()
    else:
        x = x.float()
    if use_gpu:
        x = x.cuda(gpu_dev)    
    return x
    
def stop():
    sys.tracebacklimit = 0
    class ExecutionControl(Exception): pass
    raise ExecutionControl('stopped by user')
    sys.tracebacklimit = 1000
# define setup
sz = np.array([8,8])                                           # image size
NRep =3                                          # number of repetitions
T = 512 + 4                                        # number of events F/R/P
NSpins = 22**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements

noise_std = 0*1e0                               # additive Gaussian noise std
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*sz[1]

#############################################################################
## Init spin system ::: #####################################

# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)
cutoff = 1e-12
real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
real_phantom_resized = np.zeros((sz[0],sz[1],5), dtype=np.float32)
for i in range(5):
    t = cv2.resize(real_phantom[:,:,i], dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
    if i == 0:
        t[t < 0] = 0
    elif i == 1 or i == 2:
        t[t < cutoff] = cutoff
        
    real_phantom_resized[:,:,i] = t

real_phantom_resized[:,:,1] *= 1 # Tweak T1
real_phantom_resized[:,:,2] *= 1 # Tweak T2
real_phantom_resized[:,:,3] *= 1 # Tweak dB0

spins.set_system(real_phantom_resized)
# end initialize scanned object

plt.subplot(141)
plt.imshow(real_phantom_resized[:,:,0], interpolation='none')
plt.title("PD")
plt.subplot(142)
plt.imshow(real_phantom_resized[:,:,1], interpolation='none')
plt.title("T1")
plt.subplot(143)
plt.imshow(real_phantom_resized[:,:,2], interpolation='none')
plt.title("T2")
plt.subplot(144)
plt.imshow(real_phantom_resized[:,:,3], interpolation='none')
plt.title("inhom")
plt.show()
print('use_gpu = ' +str(use_gpu)) 

#begin nspins with R*
R2 = 30.0
omega = np.linspace(0+1e-5,1-1e-5,NSpins) - 0.5    # cutoff might bee needed for opt.
#omega = np.random.rand(NSpins,NVox) - 0.5
omega = np.expand_dims(omega[:],1).repeat(NVox, axis=1)
omega*=0.9  # cutoff large freqs
omega = R2 * np.tan ( np.pi  * omega)

if NSpins==1:
    omega[:,:]=0
    
spins.omega = torch.from_numpy(omega.reshape([NSpins,NVox])).float()
spins.omega = setdevice(spins.omega)
#end nspins with R*


#############################################################################
## Init scanner system ::: #####################################
scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)
adc_mask = torch.from_numpy(np.ones((T,1))).float()
adc_mask[:2]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

# RF events: flips and phases
flips = torch.zeros((T,NRep,2), dtype=torch.float32)
flips[0,0,0] = 90*np.pi/180 
flips[0,1,0] = 180*np.pi/180 
flips[0,1,1] = 180*np.pi/180 

flips = setdevice(flips)

scanner.init_flip_tensor_holder()

B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([scanner.NCoils, scanner.NVox]))
B1plus[B1plus == 0] = 1    # set b1+ to one, where we dont have phantom measurements
B1plus[:] = 1
scanner.B1plus = setdevice(B1plus)    
scanner.set_flip_tensor_withB1plus(flips)

# rotate ADC according to excitation phase
scanner.set_ADC_rot_tensor(-flips[0,:,1] + -np.pi/2) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.2*1e-3*np.ones((scanner.T,scanner.NRep))).float()
event_time[0,:] =  2e-3
event_time[1,:] =  0.5*1e-3
event_time[-2,:] = 1e-1*1e-3
event_time[-1,:] = 0*1e-3

#event_time[1,1] =  0*1e-3

timevec = 0.2*1e-3*np.ones((scanner.T-4,))
timevec = np.cumsum(timevec)

event_time = setdevice(event_time)

TR=torch.sum(event_time[:,1])
TE=torch.sum(event_time[:11,1])

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 
grad_moms[2:-2,:,0] = 1.0*1e-1*torch.ones(int(T-4)).view(int(T-4),1).repeat([1,NRep]) # ADC open, readout, freq encoding
grad_moms[2:-2,1:,0] *= 1
grad_moms = setdevice(grad_moms)

# end sequence 

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
# forward/adjoint pass
scanner.forward(spins, event_time)
scanner.adjoint()

plt.clf()            
ax1=plt.subplot(311)
plt.plot(timevec, tonumpy(scanner.signal[0,2:-2,:,0,0]))
plt.title("sim X")

ax1=plt.subplot(312)
plt.plot(timevec,tonumpy(scanner.signal[0,2:-2,:,1,0]))
plt.title("sim Y")

ax1=plt.subplot(313)
plt.plot(timevec,np.log((tonumpy(scanner.signal[0,2:-2,:,:,0])**2).sum(2)))
plt.title("sim log mag")
fig = plt.gcf()
fig.set_size_inches(18, 6)

plt.ion()
plt.show()   


hfhfghgf

# try to fit this
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target)
targetSeq.export_to_pulseq(experiment_id,today_datestr,sequence_class,plot_seq=False)

scanner.send_job_to_real_system(experiment_id,today_datestr)
scanner.get_signal_from_real_system(experiment_id,today_datestr)

scanner.adjoint()

targetSeq.meas_sig = scanner.signal.clone()
targetSeq.meas_reco = scanner.reco.clone()

plt.clf()            
ax1=plt.subplot(311)
plt.plot(timevec,tonumpy(scanner.signal[0,2:-2,:,0,0]))
plt.title("meas X")

ax1=plt.subplot(312)
plt.plot(timevec,tonumpy(scanner.signal[0,2:-2,:,1,0]))
plt.title("meas Y")

ax1=plt.subplot(313)
plt.plot(timevec,np.log((tonumpy(scanner.signal[0,2:-2,:,:,0])**2).sum(2)))
plt.title("meas log mag")
fig = plt.gcf()
fig.set_size_inches(18, 6)

plt.ion()
plt.show()   
        



