#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: aloktyus

experiment desciption:
optimize for flip and gradient events and also for time delays between those
assume irregular event grid where flip and gradient events are interleaved with
relaxation and free pression events subject to free variable (dt) that specifies
the duration of each relax/precess event
assume very long TR and return of magnetization to initial state at the beginning of each repetition
attach NN trainable reco module to the output of adjoint operator
train on a database of <PD/T1/T2> -- <target image> pairs

"""

import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
import torch.nn.functional as fnn

import core.spins
import core.scanner
import core.nnreco
import core.opt_helper

if sys.version_info[0] < 3:
    reload(core.spins)
    reload(core.scanner)
    reload(core.nnreco)
    reload(core.opt_helper)
else:
    import importlib
    importlib.reload(core.spins)
    importlib.reload(core.scanner)
    importlib.reload(core.nnreco)
    importlib.reload(core.opt_helper)    

use_gpu = 1

# NRMSE error function
def e(gt,x):
    return 100*np.linalg.norm((gt-x).ravel())/np.linalg.norm(gt.ravel())

# get magnitude image
def magimg(x):
  return np.sqrt(np.sum(np.abs(x)**2,2))

# device setter
def setdevice(x):
    if use_gpu:
        x = x.cuda(0)
        
    return x


batch_size = 32     # number of images used at one optimization gradient step

# define setup
sz = np.array([16,16])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 2                                        # number of events F/R/P
NSpins = 2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
#dt = 0.0001                        # time interval between actions (seconds)

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]

#############################################################################
## Init spin system and the scanner ::: #####################################

dir_data = '/agbs/cpr/mr_motion/RIMphase/data/'
fn_data_tensor = 'T1w_10subjpack_16x16_cmplx.npy'                    # inputs
fn_tgt_tensor = "T1w_10subjpack_16x16_tgt_cmplx.npy"                # targets

# load and normalize
data_tensor_numpy_cmplx = np.load(os.path.join(dir_data, fn_data_tensor))
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx / np.max(data_tensor_numpy_cmplx)
tgt_tensor_numpy_cmplx = np.load(os.path.join(dir_data, fn_tgt_tensor))
ssz = data_tensor_numpy_cmplx.shape
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx.reshape([ssz[0]*ssz[1],ssz[2],ssz[3],ssz[4]])

# initialize scanned object
spins = core.spins.SpinSystem_batched(sz,NVox,NSpins,batch_size,use_gpu)

batch_idx = np.random.choice(batch_size,batch_size,replace=False)
spins.set_system(data_tensor_numpy_cmplx[batch_idx,:,:,:])

scanner = core.scanner.Scanner_batched(sz,NVox,NSpins,NRep,T,NCoils,noise_std,batch_size,use_gpu)
scanner.get_ramps()
scanner.set_adc_mask()

scanner.init_coil_sensitivities()

# init tensors
flips = torch.ones((T,NRep), dtype=torch.float32) * 0 * np.pi/180
flips[0,:] = 90*np.pi/180
     
flips = setdevice(flips)
     
scanner.init_flip_tensor_holder()
scanner.set_flip_tensor(flips)

# gradient-driver precession
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

# Cartesian encoding
grad_moms[T-sz[0]:,:,0] = torch.linspace(-sz[0]/2,sz[0]/2-1,sz[0]).view(sz[0],1).repeat([1,NRep])
grad_moms[T-sz[0]:,:,1] = torch.linspace(-sz[1]/2,sz[1]/2-1,NRep).repeat([sz[0],1])

grad_moms = setdevice(grad_moms)

# event timing vector 
event_time = torch.from_numpy(1e-2*np.zeros((scanner.T,1))).float()
event_time[0,0] = 1e-1
event_time = setdevice(event_time)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms)

#############################################################################
## Forward process ::: ######################################################
    
scanner.init_signal()
spins.set_initial_magnetization(scanner.NRep)

# always flip 90deg on first action (test)
if False:                                 
    flips_base = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
    scanner.custom_flip_allRep(0,flips_base,spins)
    scanner.custom_relax(spins,dt=0.06)                # relax till ADC (sec)
    
# scanner forward process loop
for t in range(T):                                          # for all actions

    # flip/relax/dephase only if adc is closed
    if scanner.adc_mask[t] == 0:
        scanner.flip_allRep(t,spins)
              
        delay = torch.abs(event_time[t] + 1e-6)
        scanner.set_relaxation_tensor(spins,delay)
        scanner.set_freeprecession_tensor(spins,delay)
        scanner.relax_and_dephase(spins)
        
    scanner.set_grad_op(t)
    scanner.grad_precess_allRep(spins)
    scanner.read_signal_allRep(t,spins)

# init reconstructed image
scanner.init_reco()

#############################################################################
## Inverse pass, reconstruct image with adjoint operator ::: ################
# WARNING: so far adjoint is pure gradient-precession based

for t in range(T-1,-1,-1):
    if scanner.adc_mask[t] > 0:
        scanner.set_grad_adj_op(t)
        scanner.do_grad_adj_reco(t,spins)

    
# try to fit this
target = scanner.reco.clone()
  
reco = scanner.reco.cpu().numpy().reshape([batch_size,sz[0],sz[1],2])

img_id = 0


plt.imshow(magimg(spins.images[img_id,:,:,:]))
plt.title('original (proton density) image')
plt.ion()
plt.show()

plt.imshow(magimg(reco[img_id,:,:,:]))
plt.title('output of the adjoint operator')
plt.ion()
plt.show()






