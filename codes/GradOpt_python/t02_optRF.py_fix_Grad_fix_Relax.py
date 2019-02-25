#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: aloktyus

variable flipangles assuming perfect spoiling Mxy=0, play with reordering


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
    
class ExecutionControl(Exception): pass; 
raise ExecutionControl('Script out of sync with spins/scanner classes')

use_gpu = 1
batch_size = 4     # number of images used at one optimization gradient step

if use_gpu:
    torch.cuda.empty_cache()

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
T = sz[0] + 3                                        # number of events F/R/P
NSpins = 1                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
#dt = 0.0001                         # time interval between actions (seconds)

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]


#############################################################################
## Init spin system and the scanner ::: #####################################

dir_data = '/agbs/cpr/mr_motion/RIMphase/data/'
fn_data_tensor = 'T1w_10subjpack_16x16_cmplx.npy'                      # inputs
fn_tgt_tensor = "T1w_10subjpack_16x16_tgt_cmplx.npy"                  # targets

max_data = batch_size

# load and normalize
data_tensor_numpy_cmplx = np.load(os.path.join(dir_data, fn_data_tensor))
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx / np.max(data_tensor_numpy_cmplx)
ssz = data_tensor_numpy_cmplx.shape
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx.reshape([ssz[0]*ssz[1],ssz[2],ssz[3],ssz[4]])
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx[:batch_size,:,:,:]

data_tensor_numpy_cmplx = data_tensor_numpy_cmplx[:,:,:,0]

# initialize scanned object
spins = core.spins.SpinSystem_batched(sz,NVox,NSpins,batch_size,use_gpu)

spins.set_system(data_tensor_numpy_cmplx)

scanner = core.scanner.Scanner_batched_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,batch_size,use_gpu)
scanner.get_ramps()
scanner.set_adc_mask()

# allow for relaxation after last readout event
scanner.adc_mask[:scanner.T-scanner.sz[0]-1] = 0
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
grad_moms[T-sz[0]-1:-1,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
grad_moms[T-sz[0]-1:-1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])

grad_moms = setdevice(grad_moms)

# event timing vector 
event_time = torch.from_numpy(1e-2*np.zeros((scanner.T,scanner.NRep,1))).float()
event_time[0,:,0] = 1e-3
event_time[-1,:,0] = 1e2
event_time = setdevice(event_time)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms)

#############################################################################
## Forward process ::: ######################################################

spins.set_system(data_tensor_numpy_cmplx)
    
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
    
        scanner.flip(t,r,spins)
              
        delay = torch.abs(event_time[t,r]) + 1e-6
        scanner.set_relaxation_tensor(spins,delay)
        scanner.set_freeprecession_tensor(spins,delay)
        scanner.relax_and_dephase(spins)
            
        scanner.grad_precess(t,r,spins)
        scanner.read_signal(t,r,spins)
        

# init reconstructed image
scanner.init_reco()

#############################################################################
## Inverse pass, reconstruct image with adjoint operator ::: ################
# WARNING: so far adjoint is pure gradient-precession based

for t in range(T-1,-1,-1):
    if scanner.adc_mask[t] > 0:
        scanner.do_grad_adj_reco(t,spins)

    
# try to fit this
target = scanner.reco.clone()
   
reco = scanner.reco.cpu().numpy().reshape([batch_size,sz[0],sz[1],2])

img_id = 0

if False:                                                       # check sanity
    imshow(magimg(spins.images[img_id,:,:,:]), 'original')
    imshow(magimg(reco[img_id,:,:,:]), 'reconstruction')
    
    stop()
    
    
# %% ###     OPTIMIZE ######################################################@
#############################################################################    
    
def phi_FRP_model(opt_params,aux_params):
    
    flips,grads,event_time,sigmul = opt_params
    use_periodic_grad_moms_cap = aux_params
    
    spins.set_system(images=data_tensor_numpy_cmplx)
    
    scanner.init_signal()
    
    spins.set_initial_magnetization(NRep=1)
    
    # always flip 90deg on first action (test)
    if False:                                 
        flips_base = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
        scanner.custom_flip(0,flips_base,spins)
        scanner.custom_relax(spins,dt=0.06)            # relax till ADC (sec)
        

    # only allow for flip at the beginning of repetition        
    flip_mask = torch.zeros((scanner.T, scanner.NRep)).float()        
    flip_mask[0,:] = 1
    flip_mask = setdevice(flip_mask)
    flips = flips * flip_mask
        
    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor(flips)
    
    # gradients
    grad_moms = torch.cumsum(grads,0)
    
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms)
    
    spoiler = torch.zeros((spins.NSpins, 1, spins.NVox,4,1)).float()
    spoiler[:,:,:,2:,:] = 1                # preserve longitudinal component
    spoiler = setdevice(spoiler)
    
          
    for r in range(NRep):                                   # for all repetitions
        for t in range(T):
            
            scanner.flip(t,r,spins)
            delay = torch.abs(event_time[t,r]) + 1e-6
            scanner.set_relaxation_tensor(spins,delay)
            scanner.set_freeprecession_tensor(spins,delay)
            scanner.relax_and_dephase(spins)
    
            scanner.grad_precess(t,r,spins)
            scanner.read_signal(t,r,spins) 
            
        # destroy transverse component
        spins.M = spins.M * spoiler
        
    scanner.init_reco()
    
    scanner.signal = scanner.signal * sigmul
    
    for t in range(T-1,-1,-1):
        if scanner.adc_mask[t] > 0:
            scanner.do_grad_adj_reco(t,spins)

    target_numpy = target.reshape([batch_size,NVox,2]).detach().cpu().numpy()
            
    loss = (scanner.reco - target)
    phi = torch.sum((1.0/NVox)*torch.abs(loss.squeeze())**2)
    
    ereco = scanner.reco.detach().cpu().numpy().reshape([batch_size,sz[0],sz[1],2])
    error = e(target_numpy.ravel(),ereco.ravel())    
    
    return (phi,scanner.reco, error)
    

def init_variables():
    g = np.random.rand(T,NRep,2) - 0.5

    grads = torch.from_numpy(g).float()
    
    
    grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 
    grad_moms = setdevice(grad_moms)

    grad_moms[T-sz[0]-1:-1,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
    grad_moms[T-sz[0]-1:-1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])
    
#    grad_moms[T-sz[0]-1:-1,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
#    grad_moms[T-sz[0]-1:-1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])
#    grad_moms = scanner.setdevice(grad_moms)
#    
    #grad_moms[T-sz[0]-1:-1,:,1] = torch_from_numpy(np.array([]))
    
    # centric ordering
#    grad_moms[:,:,1] = 0
#    for i in range(1,int(sz[1]/2)+1):
#        grad_moms[:,i*2-1,1] = (-i)
#        if i < sz[1]/2:
#            grad_moms[:,i*2,1] = i
            
    #grad_moms[:,:,1] = torch.flip(grad_moms[:,:,1], [1])
            
    
    padder = torch.zeros((1,scanner.NRep,2),dtype=torch.float32)
    padder = scanner.setdevice(padder)
    temp = torch.cat((padder,grad_moms),0)
    grads = temp[1:,:,:] - temp[:-1,:,:]   
    
    grads = setdevice(grads)
    grads.requires_grad = True
    
    
    flips = torch.zeros((T,NRep), dtype=torch.float32) * 90 * np.pi/180
    flips = torch.rand((T,NRep), dtype=torch.float32) * 0.1
    
    flips = setdevice(flips)
    flips.requires_grad = True
    
    flips = setdevice(flips)
    
    event_time = torch.from_numpy(np.zeros((scanner.T,scanner.NRep,1))).float()

    event_time[0,:,0] = 1e-3
    #event_time[-1,:,0] = 1e2
    
    event_time = setdevice(event_time)
    event_time.requires_grad = True
    
    #adc_mask = torch.ones((T,1)).float()*1.0
    #adc_mask = torch.ones((T,1)).float()*1
    #adc_mask[:scanner.T-scanner.sz[0]-1] = 0
    #adc_mask[-1] = 0

    #adc_mask = setdevice(adc_mask)
    #adc_mask.requires_grad = True     

    # global signal scaler
    sigmul = torch.ones((1,1)).float()*1.0
    sigmul = setdevice(sigmul)
    sigmul.requires_grad = True 
    
    return [flips, grads, event_time, sigmul]
    

    
# %% # OPTIMIZATION land
    
opt = core.opt_helper.OPT_helper(scanner,spins,None,data_tensor_numpy_cmplx.shape[0])

opt.use_periodic_grad_moms_cap = 1           # do not sample above Nyquist flag
opt.learning_rate = 0.01                                        # ADAM step size

# fast track
# opt.training_iter = 10; opt.training_iter_restarts = 5

print('<seq> now')
opt.opti_mode = 'seq'

opt.set_opt_param_idx([0,3])
opt.custom_learning_rate = [0.1, 0.1]
#opt.custom_learning_rate = [0.02, 0.02]

opt.set_handles(init_variables, phi_FRP_model)

#opt.train_model_with_restarts(nmb_rnd_restart=1, training_iter=1)

opt.scanner_opt_params = opt.init_variables()
opt.train_model(training_iter=100, show_par=True)

_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)
reco = reco.detach().cpu().numpy()

target_numpy = target.reshape([batch_size,NVox,2]).detach().cpu().numpy()
reco = reco.reshape([batch_size,sz[0],sz[1],2])
error = e(target_numpy.ravel(),reco.ravel())
print(error)

img_id = 0

imshow(magimg(target_numpy[img_id,:,:].reshape([sz[0],sz[1],2])), 'target')
imshow(magimg(reco[img_id,:,:]), 'reconstruction')

flip_angles = opt.scanner_opt_params[0].detach().cpu().numpy()*180/np.pi
flip_angles = np.round(100*flip_angles[0,:])/100

print(flip_angles)







