#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: mzaiss

experiment desciption:

2D imaging: GRE with spoilers and random phase cycling
GRE90spoiled_relax2s

"""

experiment_id = 'e15_tgtGRESP_cartesian_rotated_radial'
experiment_description = """
rotate cartesian grid
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

if sys.version_info[0] < 3:
    reload(core.spins)
    reload(core.scanner)
    reload(core.opt_helper)
else:
    import importlib

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

# device setter
def setdevice(x):
    if use_gpu:
        x = x.cuda(gpu_dev)
        
    return x
    
def imshow(x, title=None):
    plt.imshow(x, interpolation='none')
    if title != None:
        plt.title(title)
    plt.ion()
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    plt.show()     

def stop():
    sys.tracebacklimit = 0
    class ExecutionControl(Exception): pass
    raise ExecutionControl('stopped by user')
    sys.tracebacklimit = 1000

# define setup
sz = np.array([32,32])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 35**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]


#############################################################################
## Init spin system ::: #####################################

# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev)

real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
real_phantom_resized = np.zeros((sz[0],sz[1],5), dtype=np.float32)
for i in range(5):
    t = cv2.resize(real_phantom[:,:,i], dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
    if i != 3:
        t[t < 0] = 0
    real_phantom_resized[:,:,i] = t
    
spins.set_system(real_phantom_resized)

cutoff = 1e-12
spins.T1[spins.T1<cutoff] = cutoff
spins.T2[spins.T2<cutoff] = cutoff
# end initialize scanned object
spins.T1*=1
spins.T2*=1
plt.subplot(121)
plt.imshow(real_phantom_resized[:,:,0], interpolation='none')
plt.title("PD")
plt.subplot(122)
plt.imshow(real_phantom_resized[:,:,3], interpolation='none')
plt.title("inhom")
plt.show()

#begin nspins with R*
R2 = 0.0
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

scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev)
scanner.set_adc_mask()

# begin sequence definition

# allow for relaxation and spoiling in the first two and last two events (after last readout event)
scanner.adc_mask[:2]  = 0
scanner.adc_mask[-2:] = 0

# RF events: flips and phases
flips = torch.zeros((T,NRep,2), dtype=torch.float32)
flips[0,:,0] = 5*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
#flips[0,:,1] = torch.rand(flips.shape[1])*90*np.pi/180

# randomize RF phases
flips[0,:,1] = torch.tensor(scanner.phase_cycler[:NRep]).float()*np.pi/180

flips = setdevice(flips)

scanner.init_flip_tensor_holder()
scanner.set_flipXY_tensor(flips)

# rotate ADC according to excitation phase
scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.2*1e-3*np.ones((scanner.T,scanner.NRep))).float()
event_time[1,:] = 1e-3
event_time[-2,:] = 2*1e-3
event_time[-1,:] = 13.4*1e-3
#event_time[-1,:] = 1.2           # GRE/FID specific, GRE relaxation time: choose large for fully relaxed  >=1, choose small for FLASH e.g 10ms
event_time = setdevice(event_time)

TR=torch.sum(event_time[:,1])
TE=torch.sum(event_time[:11,1])

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

grad_moms[1,:,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
grad_moms[1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))  # phase encoding in second event block
grad_moms[2:-2,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep]) # ADC open, readout, freq encoding

alpha = torch.tensor(0 * np.pi / 180)

rotomat = torch.zeros((2,2)).float()
rotomat[0,0] = torch.cos(alpha)
rotomat[0,1] = -torch.sin(alpha)
rotomat[1,0] = torch.sin(alpha)
rotomat[1,1] = torch.cos(alpha)

# rotate grid
grad_moms[1,:,:] = (torch.matmul(rotomat.unsqueeze(0),grad_moms[1,:,:].unsqueeze(2))).squeeze()
grad_moms[2:-2,:,:] = (torch.matmul(rotomat.unsqueeze(0).unsqueeze(0),grad_moms[2:-2,:,:].unsqueeze(3))).squeeze()

grad_moms[-2,:,0] = torch.ones(1)*sz[0]*2      # GRE/FID specific, SPOILER
grad_moms[-2,:,1] = -grad_moms[1,:,1]      # GRE/FID specific, SPOILER
grad_moms = setdevice(grad_moms)

# end sequence 
scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,refocusing=False,wrap_k=False)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
# forward/adjoint pass
scanner.forward_fast(spins, event_time)
#scanner.forward_fast(spins, event_time)
scanner.adjoint(spins)

# scanner.reco = scanner.do_ifft_reco()

# try to fit this2
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target)

if True: # check sanity: is target what you expect and is sequence what you expect
    targetSeq.print_status(True, reco=None)
    
    #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(tonumpy(scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(scanner.T)*scanner.NRep]) )
        plt.title("ROI_def %d" % scanner.ROI_def)
        fig = plt.gcf()
        fig.set_size_inches(16, 3)
    plt.show()
    
    targetSeq.export_to_matlab(experiment_id)
    
    stop()
    
