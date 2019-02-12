#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: aloktyus

experiment desciption:
only optimize for gradients
fix flip and relax variables 


https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2965793/


--variable flipangles assuming perfect spoiling Mxy=0, play with reordering
--RF cycling and off-resonant pulses
--PSF analysis



"""

import os, sys
import numpy as np
import scipy
import torch
import cv2
import matplotlib.pyplot as plt
from torch import optim

import core.spins
import core.scanner
import core.opt_helper

if sys.version_info[0] < 3:
    reload(core.spins)
    reload(core.scanner)
    reload(core.opt_helper)
else:
    import importlib
    importlib.reload(core.spins)
    importlib.reload(core.scanner)
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
    
def imshow(x, title=None):
    plt.imshow(x, interpolation='none')
    if title != None:
        plt.title(title)
    plt.ion()
    plt.show()     

def stop():
    sys.tracebacklimit = 0
    class ExecutionControl(Exception): pass
    raise ExecutionControl('stopped by user')
    sys.tracebacklimit = 1000

# define setup
sz = np.array([16,16])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 2                                        # number of events F/R/P
NSpins = 2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
#dt = 0.0001                         # time interval between actions (seconds)

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]


#############################################################################
## Init spin system and the scanner ::: #####################################

    
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu)
spins.set_system()

scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu)
scanner.get_ramps()
scanner.set_adc_mask()

# allow for relaxation after last readout event
scanner.adc_mask[-1] = 0

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
grad_moms[T-sz[0]:,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
grad_moms[T-sz[0]:,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])

grad_moms = setdevice(grad_moms)

# event timing vector 
event_time = torch.from_numpy(1e-2*np.zeros((scanner.T,scanner.NRep,1))).float()
event_time[0,:,0] = 1e-1
event_time[-1,:,0] = 1e2
event_time = setdevice(event_time)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms)

#############################################################################
## Forward process ::: ######################################################
    
scanner.init_signal()
spins.set_initial_magnetization(NRep=1)

# always flip 90deg on first action (test)
if False:                                 
    flips_base = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
    scanner.custom_flip(0,flips_base,spins)
    scanner.custom_relax(spins,dt=0.06)                # relax till ADC (sec)
    
# scanner forward process loop
for r in range(NRep):                                   # for all repetitions
    for t in range(T):                                      # for all actions
    
        # flip/relax/dephase only if adc is closed
        if scanner.adc_mask[t] == 0:
            scanner.flip(t,r,spins)
                  
            delay = torch.abs(event_time[t,r]) + 1e-6
            scanner.set_relaxation_tensor(spins,delay)
            scanner.set_freeprecession_tensor(spins,delay)
            scanner.relax_and_dephase(spins)
            
        scanner.set_grad_op(t)
        scanner.grad_precess(r,spins)
        scanner.read_signal(t,r,spins)
        

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

if False:                                                       # check sanity
    imshow(magimg(spins.img), 'original')
    imshow(magimg(reco), 'reconstruction')
    
    stop()
    
    
# %% ###     OPTIMIZE ######################################################@
#############################################################################    
    
def phi_FRP_model(opt_params,aux_params):
    
    flips,grads,event_time = opt_params
    use_periodic_grad_moms_cap = aux_params
    
    scanner.init_signal()
    spins.set_initial_magnetization(NRep=1)
    
    # always flip 90deg on first action (test)
    if False:                                 
        flips_base = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
        scanner.custom_flip(0,flips_base,spins)
        scanner.custom_relax(spins,dt=0.06)            # relax till ADC (sec)
        
    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor(flips)
    
    # gradients
    grad_moms = torch.cumsum(grads,0)
    
    if use_periodic_grad_moms_cap:
      fmax = torch.ones([1,1,2]).float().cuda(0)
      fmax[0,0,0] = sz[0]/2
      fmax[0,0,1] = sz[1]/2

      grad_moms = torch.sin(grad_moms)*fmax
          
    scanner.set_gradient_precession_tensor(grad_moms)
          
    scanner.init_gradient_tensor_holder()
          
    for r in range(NRep):                                   # for all repetitions
        for t in range(T):
            
            if scanner.adc_mask[t] == 0:
                scanner.flip(t,r,spins)
                delay = torch.abs(event_time[t,r]) + 1e-6
                scanner.set_relaxation_tensor(spins,delay)
                scanner.set_freeprecession_tensor(spins,delay)
                scanner.relax_and_dephase(spins)
    
            scanner.set_grad_op(t)
            scanner.grad_precess(r,spins)
            scanner.read_signal(t,r,spins)        
        
    scanner.init_reco()
    
    for t in range(T-1,-1,-1):
        if scanner.adc_mask[t] > 0:
            scanner.set_grad_adj_op(t)
            scanner.do_grad_adj_reco(t,spins)
            
    loss = (scanner.reco - target)
    phi = torch.sum((1.0/NVox)*torch.abs(loss.squeeze())**2)
    
    ereco = scanner.reco.detach().cpu().numpy().reshape([sz[0],sz[1],2])
    error = e(target.cpu().numpy().ravel(),ereco.ravel())     
    
    return (phi,scanner.reco, error)
    

def init_variables():
    g = np.random.rand(T,NRep,2) - 0.5

    grads = torch.from_numpy(g).float()
    grads = setdevice(grads)
    grads.requires_grad = True
    
    grads = setdevice(grads)
    
    flips = torch.ones((T,NRep), dtype=torch.float32) * 90 * np.pi/180
    flips = torch.zeros((T,NRep), dtype=torch.float32) * 90 * np.pi/180
    #flips[0,:] = 90*np.pi/180
    flips = setdevice(flips)
    flips.requires_grad = True
    
    flips = setdevice(flips)
    
    event_time = torch.from_numpy(np.zeros((scanner.T,scanner.NRep,1))).float()
    event_time = setdevice(event_time)
    event_time.requires_grad = True
    
    return [flips, grads, event_time]
    

    
# %% # OPTIMIZATION land

opt = core.opt_helper.OPT_helper(scanner,spins,None,1)

opt.use_periodic_grad_moms_cap = 1                 # do not sample above Nyquist flag
opt.learning_rate = 0.02                                         # ADAM step size

# fast track
# opt.training_iter = 10; opt.training_iter_restarts = 5

print('<seq> now')
opt.opti_mode = 'seq'

opt.set_opt_param_idx([0,1,2])
opt.set_handles(init_variables, phi_FRP_model)

opt.train_model_with_restarts(nmb_rnd_restart=15, training_iter=10)
#opt.train_model_with_restarts(nmb_rnd_restart=2, training_iter=2)
opt.train_model(training_iter=100)
#opt.train_model(training_iter=10)

target_numpy = target.cpu().numpy().reshape([sz[0],sz[1],2])
#event_time = torch.abs(event_time)  # need to be positive

_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)
reco = reco.detach().cpu().numpy().reshape([sz[0],sz[1],2])

imshow(magimg(target_numpy), 'target')
imshow(magimg(reco), 'reconstruction')


stop()





