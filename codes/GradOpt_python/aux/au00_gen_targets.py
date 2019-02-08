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

"""

import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
import torch.nn.functional as fnn

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import core.spins; reload(core.spins)
import core.scanner; reload(core.scanner)

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

# define setup
sz = np.array([32,32])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 2                                        # number of events F/R/P
NSpins = 2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
#dt = 0.0001                         # time interval between actions (seconds)

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]

#############################################################################
## Init spin system and the scanner ::: #####################################

dir_data = '/agbs/cpr/mr_motion/RIMphase/data/'
fn_data_tensor = 'T1w_10subjpack_32x32_cmplx.npy'
fn_out = "T1w_10subjpack_32x32_tgt_cmplx.npy"

data_tensor_numpy_cmplx = np.load(os.path.join(dir_data, fn_data_tensor))
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx / np.max(data_tensor_numpy_cmplx)

ssz = data_tensor_numpy_cmplx.shape
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx.reshape([ssz[0]*ssz[1],ssz[2],ssz[3],ssz[4]])

NSubj = ssz[0]*ssz[1]

target_arr = np.zeros([NSubj,ssz[2],ssz[3],ssz[4]],dtype=np.float32)

for subjid in range(NSubj):

    # initialize scanned object
    spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu)
    spins.set_system(img=data_tensor_numpy_cmplx[subjid,:,:,:])
    
    scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu)
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
       
    reco = scanner.reco.cpu().numpy().reshape([sz[0],sz[1],2])
    
    target_numpy = target.cpu().numpy()
    
    target_arr[subjid,:,:,:] = target_numpy.reshape([sz[0],sz[1],2])
    
    if False:                                                      # check sanity
        plt.imshow(magimg(spins.img))
        plt.title('original')
        plt.ion()
        plt.show()
        
        plt.imshow(magimg(reco))
        plt.title('reconstruction')
        plt.ion()
        plt.show()
        
        gfdgfdfd
        
        
    if subjid % 100 == 0:
        print(subjid)
        
        
        
np.save(os.path.join(dir_data, fn_out), target_arr)
