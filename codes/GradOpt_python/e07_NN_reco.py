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

def stop():
    sys.tracebacklimit = 0
    class ExecutionControl(Exception): pass
    raise ExecutionControl('stopped by user')
    sys.tracebacklimit = 1000


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
grad_moms[T-sz[0]:,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
grad_moms[T-sz[0]:,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])

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

# sanity  check
#tgt_tensor_numpy_cmplx[0,:,:,:] = target.cpu().numpy().reshape([16,16,2])

if False:                                                      # check sanity
    plt.imshow(magimg(spins.images[img_id,:,:,:]), interpolation='none')
    plt.title('original')
    plt.ion()
    plt.show()
    
    plt.imshow(magimg(reco[img_id,:,:,:]), interpolation='none')
    plt.title('reconstruction')
    plt.ion()
    plt.show()
    
    stop()


# %% ###     OPTIMIZE ######################################################@
#############################################################################
        
def phi_FRP_model(opt_params,aux_params):
    
    flips,grads,event_time = opt_params
    use_tanh_grad_moms_cap,opti_mode = aux_params

    spins.set_system(images=data_tensor_numpy_cmplx[opt.subjidx,:,:,:])
    
    scanner.init_signal()
    
    target = torch.from_numpy(tgt_tensor_numpy_cmplx[opt.subjidx,:,:,:].reshape([batch_size,NVox,2])).float().cuda()
    
    spins.set_initial_magnetization(scanner.NRep)
    
    # always flip 90deg on first action (test)
    if False:                                 
        flips_base = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
        scanner.custom_flip_allRep(0,flips_base,spins)
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
          
    grad_moms = grads
    scanner.set_gradient_precession_tensor(grad_moms)
          
    scanner.init_gradient_tensor_holder()
          
    for t in range(T):
        
        if scanner.adc_mask[t] == 0:
            scanner.flip_allRep(t,spins)
            delay = torch.abs(event_time[t] + 1e-6)
            scanner.set_relaxation_tensor(spins,delay)
            scanner.set_freeprecession_tensor(spins,delay)
            scanner.relax_and_dephase(spins)

        scanner.set_grad_op(t)
        scanner.grad_precess_allRep(spins)
        scanner.read_signal_allRep(t,spins)        
        
    scanner.init_reco()
    
    for t in range(T-1,-1,-1):
        if scanner.adc_mask[t] > 0:
            scanner.set_grad_adj_op(t)
            scanner.do_grad_adj_reco(t,spins)
    
    if opti_mode == 'nn' or opti_mode == 'seqnn':
        scanner.reco = scanner.reco
        scanner.reco = NN(scanner.reco)
            
    loss = (scanner.reco - target)
    phi = torch.sum((1.0/NVox)*torch.abs(loss.squeeze())**2)
    
    target_numpy = tgt_tensor_numpy_cmplx[opt.subjidx,:,:,:].reshape([batch_size,NVox,2])
    ereco = scanner.reco.detach().cpu().numpy().reshape([batch_size,sz[0],sz[1],2])
    error = e(target_numpy.ravel(),ereco.ravel())    

    return (phi,scanner.reco,error)

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
    
    event_time = torch.from_numpy(np.zeros((scanner.T,1))).float()
    event_time = setdevice(event_time)
    event_time.requires_grad = True
    
    return [flips, grads, event_time]
    

    
# %% # OPTIMIZATION land

# set number of convolution neurons
nmb_conv_neurons_list = [2,32,32,32,2]

# initialize reconstruction module
NN = core.nnreco.RecoConvNet_basic(spins.sz, nmb_conv_neurons_list).cuda()
    
opt = core.opt_helper.OPT_helper(scanner,spins,NN,tgt_tensor_numpy_cmplx.shape[0])

opt.use_tanh_grad_moms_cap = 1                 # do not sample above Nyquist flag
opt.learning_rate = 0.02                                         # ADAM step size

# fast track
# opt.training_iter = 10; opt.training_iter_restarts = 5

print('<seq> now')
opt.opti_mode = 'seq'

opt.set_handles(init_variables, phi_FRP_model)

opt.train_model_with_restarts(nmb_rnd_restart=15, training_iter=10)
#opt.train_model_with_restarts(nmb_rnd_restart=2, training_iter=2)
opt.train_model(training_iter=100)
#opt.train_model(training_iter=10)

if True:
    print('<nn> now')
    opt.opti_mode = 'nn'
    opt.train_model(training_iter=100)
    #opt.train_model(training_iter=10)

if True:
    print('<seqnn> now')
    opt.opti_mode = 'seqnn'
    opt.train_model(training_iter=1500)
    #opt.train_model(training_iter=10)


target_numpy = tgt_tensor_numpy_cmplx[opt.subjidx,:,:,:].reshape([batch_size,NVox,2])
#event_time = torch.abs(event_time)  # need to be positive

# %% # show results

opt.new_batch()
_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)
reco = reco.detach().cpu().numpy().reshape([batch_size,sz[0],sz[1],2])

img_id = 0

plt.imshow(magimg(target_numpy[img_id,:,:].reshape([sz[0],sz[1],2])), interpolation='none')
plt.title('target')
plt.ion()
plt.show()

plt.imshow(magimg(reco[img_id,:,:]), interpolation='none')
plt.title('reconstruction')

target_numpy = tgt_tensor_numpy_cmplx[opt.subjidx,:,:,:].reshape([batch_size,NVox,2])
reco = reco.reshape([batch_size,sz[0],sz[1],2])
error = e(target_numpy.ravel(),reco.ravel())
print(error)





