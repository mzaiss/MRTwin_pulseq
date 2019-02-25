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

use_gpu = 0

# NRMSE error function
def e(gt,x):
    return 100*np.linalg.norm((gt-x).ravel())/np.linalg.norm(gt.ravel())

# get magnitude image
def magimg(x):
  return np.sqrt(np.sum(np.abs(x)**2,2))
  
# torch to numpy
def tonumpy(x):
    return x.detach().cpu().numpy()

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
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    plt.show()     

def stop():
    sys.tracebacklimit = 0
    class ExecutionControl(Exception): pass
    raise ExecutionControl('stopped by user')
    sys.tracebacklimit = 1000

# define setup
sz = np.array([32,1])                                           # image size
NRep = 1                                          # number of repetitions
T = sz[0] + 3                                        # number of events F/R/P
NSpins = 2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
#dt = 0.0001                         # time interval between actions (seconds)

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]


#############################################################################
## Init spin system and the scanner ::: #####################################

    
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu)

numerical_phantom = np.load('../../data/brainphantom_2D.npy')
numerical_phantom = cv2.resize(numerical_phantom, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
numerical_phantom = np.expand_dims(numerical_phantom[:,16,:],1)
numerical_phantom[numerical_phantom < 0] = 0

#numerical_phantom = np.zeros((sz[0],sz[1],3))
#numerical_phantom[10,:,:]=2
#numerical_phantom[23,:,:]=1
#numerical_phantom[24,:,:]=1.5
#numerical_phantom[25,:,:]=1
#numerical_phantom[30,:,:]=0.1
#numerical_phantom[28,:,:]=0.3
#numerical_phantom[29,:,:]=0.6
#numerical_phantom[30,:,:]=0.3
#numerical_phantom[31,:,:]=0.1
#numerical_phantom[:,:,2]*=0.1  # T2=100ms
#
##numerical_phantom = cv2.resize(numerical_phantom, dsize=(sz[0], sz[1]), interpolation=cv2.INTER_CUBIC)
#numerical_phantom[numerical_phantom < 0] = 0

spins.set_system(numerical_phantom)

cutoff = 1e-12
spins.T1[spins.T1<cutoff] = cutoff
spins.T2[spins.T2<cutoff] = cutoff


scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu)
scanner.get_ramps()
scanner.set_adc_mask()

# allow for relaxation after last readout event
scanner.adc_mask[:scanner.T-scanner.sz[0]-1] = 0
scanner.adc_mask[-1] = 0

scanner.init_coil_sensitivities()

# init tensors
flips = torch.ones((T,NRep), dtype=torch.float32) * 0 * np.pi/180
flips[0,:] = 90*np.pi/180  # GRE preparation part 1 : 90 degree excitation
     
flips = setdevice(flips)
     
scanner.init_flip_tensor_holder()
scanner.set_flip_tensor(flips)

# gradient-driver precession
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

# Cartesian encoding
grad_moms[T-sz[0]-1:-1,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
grad_moms[T-sz[0]-1:-1,:,1] = torch.zeros((1,1)).repeat([sz[0],1])

# imshow(grad_moms[T-sz[0]-1:-1,:,0].cpu())
# imshow(grad_moms[T-sz[0]-1:-1,:,1].cpu())

grad_moms = setdevice(grad_moms)

# event timing vector 
event_time = torch.from_numpy(1e-2*np.zeros((scanner.T,scanner.NRep,1))).float()
event_time[1,:,0] = 1e-2  
event_time[-1,:,0] = 1e2
event_time = setdevice(event_time)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms)

#############################################################################
## Forward process ::: ######################################################
scanner.forward(spins, event_time)

#############################################################################
## Inverse pass, reconstruct image with adjoint operator ::: ################
# WARNING: so far adjoint is pure gradient-precession based
scanner.adjoint(spins)

# try to fit this
target = scanner.reco.clone()
   
reco = scanner.reco.cpu().numpy().reshape([sz[0],sz[1],2])

if False:                                                        # check sanity
    imshow(tonumpy(spins.PD).reshape([sz[0],sz[1]]), 'input array: PD')
    imshow(magimg(reco), 'target image to fit')
    
    stop()
    
    

# %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
    
def phi_FRP_model(opt_params,aux_params):
    
    flips,grads,event_time,adc_mask = opt_params
    use_periodic_grad_moms_cap,_ = aux_params
    
    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor(flips)
    
    # gradients
    grad_moms = torch.cumsum(grads,0)
    
    # dont optimize y  grads
    grad_moms[:,:,1] = 0    
    
    if use_periodic_grad_moms_cap:
        fmax = torch.ones([1,1,2]).float()
        fmax = setdevice(fmax)
        fmax[0,0,0] = sz[0]/2
        fmax[0,0,1] = sz[1]/2

        grad_moms = torch.sin(grad_moms)*fmax

    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms)
    
    scanner.adc_mask = adc_mask
   
    scanner.forward(spins, event_time)
    
    #### now one full forward model is calculated and the acquired signals are stored in self.signal
    #### now we can create the reco for this data. Gadjoint was already created during the forward process ( inj set_gradient_precession_tensor) see self.G and self.G_adj
    scanner.adjoint(spins)
            
    loss = (scanner.reco - target)
    phi = torch.sum((1.0/NVox)*torch.abs(loss.squeeze())**2)
    
    ereco = tonumpy(scanner.reco).reshape([sz[0],sz[1],2])
    error = e(target.cpu().numpy().ravel(),ereco.ravel())     
    
    return (phi,scanner.reco, error)
    

def init_variables():
    g = np.random.rand(T,NRep,2) - 0.5

    grads = torch.from_numpy(g).float()
    
#    grad_moms[T-sz[0]-1:-1,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
#    grad_moms[T-sz[0]-1:-1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])
#    
#    padder = torch.zeros((1,scanner.NRep,2),dtype=torch.float32)
#    padder = scanner.setdevice(padder)
#    temp = torch.cat((padder,grad_moms),0)
#    grads = temp[1:,:,:] - temp[:-1,:,:]   
    
    grads = setdevice(grads)
    grads.requires_grad = True
    
    
    #flips = torch.ones((T,NRep), dtype=torch.float32) * 90 * np.pi/180
    flips = torch.zeros((T,NRep), dtype=torch.float32) * 90 * np.pi/180
    
    #flips[0,:] = 90*np.pi/180
    flips = setdevice(flips)
    flips.requires_grad = True
    

    #event_time = torch.from_numpy(np.zeros((scanner.T,scanner.NRep,1))).float()
    event_time = torch.from_numpy(0.1*np.random.rand(scanner.T,scanner.NRep,1)).float()

    #event_time[0,:,0] = 1e-1
    #event_time[-1,:,0] = 1e2
    
    event_time = setdevice(event_time)
    event_time.requires_grad = True
    
    adc_mask = torch.ones((T,1)).float()*0.1
    adc_mask = torch.ones((T,1)).float()*1
    adc_mask[:scanner.T-scanner.sz[0]-1] = 0
    adc_mask[-1] = 0

    adc_mask = setdevice(adc_mask)
    adc_mask.requires_grad = True     
    
    return [flips, grads, event_time, adc_mask]
    

    
# %% # OPTIMIZATION land

opt = core.opt_helper.OPT_helper(scanner,spins,None,1)
opt.set_target(reco)

opt.use_periodic_grad_moms_cap = 1           # do not sample above Nyquist flag
opt.learning_rate = 0.01                                        # ADAM step size

# fast track
# opt.training_iter = 10; opt.training_iter_restarts = 5

print('<seq> now (with 10 iterations and several random initializations)')
opt.opti_mode = 'seq'

target_numpy = target.cpu().numpy().reshape([sz[0],sz[1],2])
imshow(magimg(target_numpy), 'target')
opt.set_opt_param_idx([0,1,2])
opt.custom_learning_rate = [0.05,0.01,0.05,0.1]

opt.set_handles(init_variables, phi_FRP_model)

opt.train_model_with_restarts(nmb_rnd_restart=5, training_iter=10)
#opt.train_model_with_restarts(nmb_rnd_restart=2, training_iter=2)

#stop()

print('<seq> now (100 iterations with best initialization')

opt.train_model(training_iter=1000, do_vis_image=True)
#opt.train_model(training_iter=10)


#event_time = torch.abs(event_time)  # need to be positive

#opt.scanner_opt_params = init_variables()
#

_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)
reco = tonumpy(reco).reshape([sz[0],sz[1],2])

imshow(magimg(target_numpy), 'target')
imshow(magimg(reco), 'reconstruction')

stop()





