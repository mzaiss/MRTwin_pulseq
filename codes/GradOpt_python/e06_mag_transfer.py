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
__allow for magnetization transfer over repetitions__


"""

import os, sys
import numpy as np
import scipy
import torch
import cv2
import matplotlib.pyplot as plt
from torch import optim

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
grad_moms[T-sz[0]:,:,0] = torch.linspace(-sz[0]/2,sz[0]/2-1,sz[0]).view(sz[0],1).repeat([1,NRep])
grad_moms[T-sz[0]:,:,1] = torch.linspace(-sz[1]/2,sz[1]/2-1,NRep).repeat([sz[0],1])

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
                  
            delay = torch.abs(event_time[t,r] + 1e-6)
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
    plt.imshow(magimg(spins.img))
    plt.title('original')
    plt.ion()
    plt.show()
    
    plt.imshow(magimg(reco))
    plt.title('reconstruction')
    plt.ion()
    plt.show()
    
    gfdgfdfd


# %% ###     OPTIMIZE ######################################################@
#############################################################################

def phi_FRP_model(flips,grads,event_time,args):
    
    scanner,spins,target,use_tanh_grad_moms_cap = args
    
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
    
    if use_tanh_grad_moms_cap:
      boost_fct = 1
        
      fmax = sz / 2
      for i in [0,1]:
          grad_moms[:,:,i] = boost_fct*fmax[i]*torch.tanh(grad_moms[:,:,i])
          
    scanner.set_gradient_precession_tensor(grad_moms)
          
    scanner.init_gradient_tensor_holder()
          
    for r in range(NRep):                                   # for all repetitions
        for t in range(T):
            
            if scanner.adc_mask[t] == 0:
                scanner.flip(t,r,spins)
                delay = torch.abs(event_time[t,r] + 1e-6)
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
    
    return (phi,scanner.reco)

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
    
    return flips, grads, event_time
    

target = target.detach()
target_numpy = target.cpu().numpy().reshape([sz[0],sz[1],2])

def train_model(doRestart=False, best_vars=None,nmb_inner_iter=None):
    
    # init gradients and flip events
    if doRestart:
        nmb_outer_iter = nmb_rnd_restart
        nmb_inner_iter = training_iter_restarts
        
        best_error = 200
        best_vars = []        
    else:
        nmb_outer_iter = 1
        if nmb_inner_iter is None:
            nmb_inner_iter = training_iter
        
        flips,grads,event_time = best_vars
        flips.requires_grad = True
        grads.requires_grad = True
        event_time.requires_grad = True
        
        
    def weak_closure():
        optimizer.zero_grad()
        loss,_ = phi_FRP_model(flips, grads, event_time, args)
        loss.backward()
        
        return loss
    
    for outer_iter in range(nmb_outer_iter):
        
        if doRestart:
            flips,grads,event_time = init_variables()
    
        optimizer = optim.LBFGS([flips,grads,event_time], lr=learning_rate, max_iter=1,history_size=200)
        
        for inner_iter in range(nmb_inner_iter):
            optimizer.step(weak_closure)
            
            _,reco = phi_FRP_model(flips, grads, event_time, args)
            reco = reco.detach().cpu().numpy().reshape([sz[0],sz[1],2])
            error = e(target_numpy.ravel(),reco.ravel())
            
            if doRestart:
                if error < best_error:
                    best_error = error
                    best_vars = (flips.detach().clone(),grads.detach().clone(),event_time.detach().clone())
                    print("recon error = %f" %error)
            else:
                print("recon error = %f" %error)
        
    if doRestart:
        return best_vars
    else:
        return reco,flips,grads,torch.abs(event_time)
    

use_tanh_grad_moms_cap = 1                 # do not sample above Nyquist flag
learning_rate = 0.1                                         # LBFGS step size
training_iter = 2500
nmb_rnd_restart = 15
training_iter_restarts = 10

args = (scanner,spins,target,use_tanh_grad_moms_cap)

best_vars = train_model(True)
    
reco,flips,grads,event_time = train_model(False,best_vars)

plt.imshow(magimg(spins.img))
plt.title('original')
plt.ion()
plt.show()

plt.imshow(magimg(reco))
plt.title('reconstruction')

gfdgfdg


# %% ###     SAVE ALL ######################################################@
#############################################################################

host_dir = "../../data/trained_models"
output_dir = "t00_magtrans_early"

if not os.path.exists(os.path.join(host_dir,output_dir)):
    os.makedirs(os.path.join(host_dir,output_dir))
  
    sz = np.array([16,16])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 2                                        # number of events F/R/P
NSpins = 2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
    
param_dict = dict()
param_dict['sz'] = sz
param_dict['NRep'] = NRep
param_dict['T'] = T
param_dict['NSpins'] = NSpins
param_dict['NCoils'] = NCoils

scipy.io.savemat(os.path.join(host_dir,output_dir,"param_dict.mat"), param_dict)

spins_dict = dict()
spins_dict['PD'] = spins.PD
spins_dict['T1'] = spins.T1          
spins_dict['T2'] = spins.T2
spins_dict['dB0'] = spins.dB0
          
scipy.io.savemat(os.path.join(host_dir,output_dir,"spins_dict.mat"), spins_dict)
          
scanner_dict = dict()
scanner_dict['adc_mask'] = scanner.adc_mask.detach().cpu().numpy()  
scanner_dict['B1'] = scanner.B1.detach().cpu().numpy()
scanner_dict['flips'] = flips.detach().cpu().numpy()
scanner_dict['grads'] = grads.detach().cpu().numpy()
scanner_dict['event_time'] = event_time.detach().cpu().numpy()
scanner_dict['reco'] = reco
          
scipy.io.savemat(os.path.join(host_dir,output_dir,"scanner_dict.mat"), scanner_dict)




