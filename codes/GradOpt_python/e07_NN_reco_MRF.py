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
    
class ExecutionControl(Exception): pass; 
raise ExecutionControl('Script out of sync with spins/scanner classes')

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


batch_size = 32     # number of images used at one optimization gradient step

# define setup
sz = np.array([16,16])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 3                                        # number of events F/R/P
NSpins = 1                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
#dt = 0.0001                        # time interval between actions (seconds)

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]

#############################################################################
## Init spin system and the scanner ::: #####################################

dir_data = '../../data/'
fn_data_tensor = 'num_brain_slice16x16.npy'                            # inputs

# load and normalize
data_tensor_numpy_cmplx = np.load(os.path.join(dir_data, fn_data_tensor)).astype(np.float32)
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx / np.max(data_tensor_numpy_cmplx)
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx + 1e-5
ssz = data_tensor_numpy_cmplx.shape

# initialize scanned object
spins = core.spins.SpinSystem_batched(sz,NVox,NSpins,batch_size,use_gpu)

batch_idx = np.random.choice(batch_size,batch_size,replace=False)
spins.set_system(data_tensor_numpy_cmplx[batch_idx,:,:,0])
spins.T1 = setdevice(torch.from_numpy(data_tensor_numpy_cmplx[batch_idx,:,:,1]).view([batch_size,NVox]))
spins.T2 = setdevice(torch.from_numpy(data_tensor_numpy_cmplx[batch_idx,:,:,2]).view([batch_size,NVox]))

scanner = core.scanner.Scanner_batched_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,batch_size,use_gpu)
scanner.get_ramps()

scanner.set_adc_mask()
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
        

scanner.init_reco()

#############################################################################
## Inverse pass, reconstruct image with adjoint operator ::: ################
# WARNING: so far adjoint is pure gradient-precession based

for t in range(T-1,-1,-1):
    if scanner.adc_mask[t] > 0:
        scanner.do_grad_adj_reco(t,spins)

    
# try to fit this
target = scanner.reco.clone()
  
reco = scanner.reco.detach().cpu().numpy().reshape([batch_size,sz[0],sz[1],2])
#tgt_tensor_numpy_cmplx = reco

img_id = 0

# sanity  check
#tgt_tensor_numpy_cmplx[0,:,:,:] = target.cpu().numpy().reshape([16,16,2])

if False:                                                      # check sanity
    imshow(spins.images[img_id,:,:], 'original')
    imshow(magimg(reco[img_id,:,:,:]), 'reconstruction')
    
    stop()
    

# %% ###     OPTIMIZE ######################################################@
#############################################################################
        
def phi_FRP_model(opt_params,aux_params):
    
    flips,grads,event_time, adc_mask = opt_params
    use_periodic_grad_moms_cap,opti_mode = aux_params

    spins.set_system(data_tensor_numpy_cmplx[opt.subjidx,:,:,0])
    spins.T1 = setdevice(torch.from_numpy(data_tensor_numpy_cmplx[opt.subjidx,:,:,1]).view([batch_size,NVox]))
    spins.T2 = setdevice(torch.from_numpy(data_tensor_numpy_cmplx[opt.subjidx,:,:,2]).view([batch_size,NVox]))    
    
    target = torch.from_numpy(data_tensor_numpy_cmplx[opt.subjidx,:,:,:].reshape([batch_size,NVox,3])).float().cuda()
    target = target / torch.max(target,dim=1)[0].unsqueeze(1)
    #target[torch.isnan(target)] = 0
    
    # gradients
    #grad_moms = torch.cumsum(grads,0)
    
    if use_periodic_grad_moms_cap:
      fmax = torch.ones([1,1,2]).float().cuda(0)
      fmax[0,0,0] = sz[0]/2
      fmax[0,0,1] = sz[1]/2

      grad_moms = torch.sin(grad_moms)*fmax
          
    grad_moms = grads
    scanner.adc_mask = adc_mask
    
    scanner.init_gradient_tensor_holder()
    scanner.set_gradient_precession_tensor(grad_moms)
    
    #############################################################################
    ## Forward process ::: ######################################################
        
    spins.set_initial_magnetization(NRep=1)
    scanner.init_signal()
    
    for r in range(NRep):                                   # for all repetitions
        for t in range(T):
            
            scanner.flip(t,r,spins)
                  
            delay = torch.abs(event_time[t,r] + 1e-6)
            scanner.set_relaxation_tensor(spins,delay)
            scanner.set_freeprecession_tensor(spins,delay)
            scanner.relax_and_dephase(spins)
                
            scanner.grad_precess(t,r,spins)
            scanner.read_signal(t,r,spins)    
        
    scanner.init_reco()
    
    echo_stack = torch.zeros((NRep,NVox)).float()
    echo_stack = setdevice(echo_stack)
    
    adc_mask = scanner.adc_mask.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4)
    s = scanner.signal * adc_mask
    s = torch.sum(s,1)
    
    r = torch.matmul(scanner.G_adj.permute([1,2,3,0,4]).contiguous().view([1,scanner.NRep,scanner.NVox,4,scanner.T*4]), s.permute([0,2,1,3,4]).contiguous().view([scanner.batch_size,scanner.NRep,1,scanner.T*4,1]))
    
    echo_stack = r[:,:,:,:2,0].permute([0,2,1,3]).contiguous().view([scanner.batch_size,NVox,scanner.NRep*2])
    
#    for t in range(T-1,-1,-1):
#        if scanner.adc_mask[t] > 0:
#            scanner.do_grad_adj_reco(t,spins)
#            
    scanner.reco = NN(echo_stack)
    
#    if opti_mode == 'nn' or opti_mode == 'seqnn':
#        scanner.reco = NN(scanner.reco)
        
    loss = (scanner.reco - target)
    phi = torch.sum((1.0/NVox)*torch.abs(loss.squeeze())**2)
    
    target_numpy = data_tensor_numpy_cmplx[opt.subjidx,:,:,:].reshape([batch_size,NVox,3])
    target_numpy = target_numpy / np.max(target_numpy,1)[0].reshape([1,1,3])
    
    ereco = scanner.reco.detach().cpu().numpy().reshape([batch_size,sz[0],sz[1],3])
    error = e(target_numpy.ravel(),ereco.ravel())    

    return (phi,scanner.reco,error)

def init_variables():
    g = (np.random.rand(T,NRep,2) - 0.5)*0.1

    #grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 
    #grad_moms = setdevice(grad_moms)

#    grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 
#    grad_moms[T-sz[0]:,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
#    grad_moms[T-sz[0]:,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])
    
#    padder = torch.zeros((1,scanner.NRep,2),dtype=torch.float32)
#    padder = scanner.setdevice(padder)
#    temp = torch.cat((padder,grad_moms),0)
#    grads = temp[1:,:,:] - temp[:-1,:,:]   
   
    #grad_moms = setdevice(grad_moms)
    grads = torch.from_numpy(g).float()
   
    grads = setdevice(grads)
    grads.requires_grad = True
    
    flips = torch.ones((T,NRep), dtype=torch.float32) * 90 * np.pi/180
    flips = torch.zeros((T,NRep), dtype=torch.float32) * 90 * np.pi/180
    #flips[0,:] = 90*np.pi/180
    
    flips = setdevice(flips)
    flips.requires_grad = True
    
    event_time = torch.from_numpy(1e-2*np.random.rand(scanner.T,scanner.NRep,1)).float()
#    event_time[0,:,0] = 1e-1
#    event_time[-1,:,0] = 1e2
    
    event_time = setdevice(event_time)
    event_time.requires_grad = True
    
    adc_mask = torch.ones((T,1)).float()*0.1
#    adc_mask = torch.ones((T,1)).float()*1
#    adc_mask[:scanner.T-scanner.sz[0]-1] = 0
#    adc_mask[-1] = 0

    adc_mask = setdevice(adc_mask)
    adc_mask.requires_grad = True
    
    return [flips, grads, event_time, adc_mask]
    

    
# %% # OPTIMIZATION land

# set number of convolution neurons
nmb_conv_neurons_list = [2*NRep,32,32,32,3]

# initialize reconstruction module
NN = core.nnreco.RecoConvNet_basic(spins.sz, nmb_conv_neurons_list).cuda()
    
opt = core.opt_helper.OPT_helper(scanner,spins,NN,data_tensor_numpy_cmplx.shape[0])

opt.use_periodic_grad_moms_cap = 0           # do not sample above Nyquist flag
opt.learning_rate = 0.05                                       # ADAM step size

# fast track
# opt.training_iter = 10; opt.training_iter_restarts = 5

print('<seqnn> now')
opt.opti_mode = 'seqnn'

opt.set_opt_param_idx([0,1,2,3])

opt.custom_learning_rate = [0.1, 0.05, 0.1, 0.1]
opt.set_handles(init_variables, phi_FRP_model)

#opt.train_model_with_restarts(nmb_rnd_restart=15, training_iter=10)
#opt.train_model_with_restarts(nmb_rnd_restart=2, training_iter=2)

opt.scanner_opt_params = opt.init_variables()
opt.train_model(training_iter=10000)
#opt.train_model(training_iter=10)

if False:
    print('<nn> now')
    opt.opti_mode = 'nn'
    opt.train_model(training_iter=100)
    #opt.train_model(training_iter=10)

if False:
    print('<seqnn> now')
    opt.opti_mode = 'seqnn'
    opt.train_model(training_iter=1500)
    #opt.train_model(training_iter=10)


#event_time = torch.abs(event_time)  # need to be positive

# %% # show results

target_numpy = data_tensor_numpy_cmplx[opt.subjidx,:,:,:].reshape([batch_size,NVox,3])

opt.aux_params = [opt.use_periodic_grad_moms_cap, opt.opti_mode]

opt.new_batch()
_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)
reco = reco.detach().cpu().numpy().reshape([batch_size,sz[0],sz[1],2])

img_id = 0

imshow(magimg(target_numpy[img_id,:,:].reshape([sz[0],sz[1],2])), 'target')
imshow(magimg(reco[img_id,:,:]), 'reconstruction')

target_numpy = data_tensor_numpy_cmplx[opt.subjidx,:,:,:].reshape([batch_size,NVox,3])
reco = reco.reshape([batch_size,sz[0],sz[1],2])
error = e(target_numpy.ravel(),reco.ravel())
print(error)





