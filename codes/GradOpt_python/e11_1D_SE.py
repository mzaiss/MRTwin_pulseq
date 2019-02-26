#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: mzaiss

experiment desciption:
optimize for flip and gradient events and also for time delays between those
assume irregular event grid where flip and gradient events are interleaved with
relaxation and free pression events subject to free variable (dt) that specifies
the duration of each relax/precess event
__allow for magnetization transfer over repetitions__

1D imaging: 


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
import core.target_seq_holder

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
    
# torch to numpy
def tonumpy(x):
    return x.detach().cpu().numpy()

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
NRep = 2                                          # number of repetitions
T = sz[0] + 3                                        # number of events F/R/P
NSpins = 1                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
#dt = 0.0001                         # time interval between actions (seconds)

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]


#############################################################################
## Init spin system and the scanner ::: #####################################

# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu)

numerical_phantom = np.load('../../data/brainphantom_2D.npy')
numerical_phantom = cv2.resize(numerical_phantom, dsize=(sz[0],sz[0]), interpolation=cv2.INTER_CUBIC)
#numerical_phantom = cv2.resize(numerical_phantom, dsize=(sz[0], sz[1]), interpolation=cv2.INTER_CUBIC)
numerical_phantom[numerical_phantom < 0] = 0
row=22
h1=numerical_phantom[:,:,0].copy()
h1[:,row]*=2
plt.imshow(h1[:,:])
plt.show()
numerical_phantom=numerical_phantom[:,row,:].reshape(sz[0],1,3).copy()

plt.plot(numerical_phantom[:,:,0], label='PD')
plt.plot(numerical_phantom[:,:,1], label='T1')
plt.plot(numerical_phantom[:,:,2], label='T2')
plt.show()

numerical_phantom[numerical_phantom < 0] = 0
spins.set_system(numerical_phantom)

cutoff = 1e-12
spins.T1[spins.T1<cutoff] = cutoff
spins.T2[spins.T2<cutoff] = cutoff
# end initialize scanned object

scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu)
scanner.get_ramps()
scanner.set_adc_mask()

# allow for relaxation after last readout event
scanner.adc_mask[:scanner.T-scanner.sz[0]-1] = 0
scanner.adc_mask[-1] = 0

scanner.init_coil_sensitivities()

# init tensors
flips = torch.ones((T,NRep), dtype=torch.float32) * 0 * np.pi/180
flips[0,:] = 90*np.pi/180  # SE preparation part 1 : 90 degree excitation
flips[1,:] = 180*np.pi/180  # SE preparation part 2 : 180 degree refocus
     
flips = setdevice(flips)
     
scanner.init_flip_tensor_holder()
scanner.set_flip_tensor(flips)

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

grad_moms[T-sz[0]-1:-1,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
if NRep == 1:
    grad_moms[T-sz[0]-1:-1,:,1] = torch.zeros((1,1)).repeat([sz[0],1])
else:
    grad_moms[T-sz[0]-1:-1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])
    
grad_moms[-1,:,:] = grad_moms[-2,:,:]

# dont optimize y  grads
grad_moms[:,:,1] = 0

    
#imshow(grad_moms[T-sz[0]-1:-1,:,0].cpu())
#imshow(grad_moms[T-sz[0]-1:-1,:,1].cpu())

grad_moms = setdevice(grad_moms)

# event timing vector 
event_time = torch.from_numpy(1e-2*np.ones((scanner.T,scanner.NRep,1))).float()
event_time[0,:,0] = 1e-1  
event_time = setdevice(event_time)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms)

#############################################################################
## Forward process ::: ######################################################
    
# forward/adjoint pass
scanner.forward(spins, event_time)
scanner.adjoint(spins)

# try to fit this
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder()
targetSeq.target_image = target
targetSeq.sz = sz
targetSeq.flips = flips
targetSeq.grad_moms = grad_moms
targetSeq.event_time = event_time
targetSeq.adc_mask = scanner.adc_mask

if False: # check sanity: is target what you expect and is sequence what you expect
    targetSeq.print_status(True, reco=None)
    stop()
    
    
# %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
    
def phi_FRP_model(opt_params,aux_params):
    
    flips,grads,event_time,adc_mask = opt_params
    use_periodic_grad_moms_cap,_ = aux_params
    
    flip_mask = torch.zeros((scanner.T, scanner.NRep)).float()        
    flip_mask[:2,:] = 1
    flip_mask = setdevice(flip_mask)
    flips = flips * flip_mask    
    
    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor(flips)
    
    # gradients
    #grad_moms = torch.cumsum(grads,0)
    grad_moms = grads*1
    
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
    
    #scanner.adc_mask = adc_mask
          
    # forward/adjoint pass
    scanner.forward(spins, event_time)
    scanner.adjoint(spins)

            
    loss = (scanner.reco - targetSeq.target_image)
    #phi = torch.sum((1.0/NVox)*torch.abs(loss.squeeze())**2)
    phi = torch.sum(loss.squeeze()**2/NVox)
    
    ereco = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(targetSeq.target_image).ravel(),ereco.ravel())     
    
    return (phi,scanner.reco, error)
    

def init_variables():
    
    use_gtruth_grads = True    # if this is changed also use_periodic_grad_moms_cap must be changed
    if use_gtruth_grads:
        grad_moms = targetSeq.grad_moms.clone()
        
#        padder = torch.zeros((1,scanner.NRep,2),dtype=torch.float32)
#        padder = scanner.setdevice(padder)
#        temp = torch.cat((padder,grad_moms),0)
#        grads = temp[1:,:,:] - temp[:-1,:,:]   
    else:
        g = (np.random.rand(T,NRep,2) - 0.5)*2*np.pi
        
        grad_moms = torch.from_numpy(g).float()
        grad_moms[:,:,1] = 0        
        grad_moms = setdevice(grad_moms)
    
    grad_moms.requires_grad = True
    
    flips = targetSeq.flips.clone()
    flips.requires_grad = True
   
    event_time = targetSeq.event_time.clone()
    event_time = torch.from_numpy(1e-3*np.random.rand(scanner.T,scanner.NRep,1)).float()
    event_time = setdevice(event_time)
    event_time.requires_grad = True

    adc_mask = targetSeq.adc_mask.clone()
    adc_mask.requires_grad = True     
    
    return [flips, grad_moms, event_time, adc_mask]
    

    
# %% # OPTIMIZATION land

opt = core.opt_helper.OPT_helper(scanner,spins,None,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))

opt.use_periodic_grad_moms_cap = 0           # do not sample above Nyquist flag
opt.learning_rate = 0.01                                        # ADAM step size
opt.optimzer_type = 'Adam'

# fast track
# opt.training_iter = 10; opt.training_iter_restarts = 5

print('<seq> now (with 10 iterations and several random initializations)')
opt.opti_mode = 'seq'

#target_numpy = tonumpy(target).reshape([sz[0],sz[1],2])
#imshow(magimg(target_numpy), 'target')

opt.set_opt_param_idx([2])
#opt.custom_learning_rate = [0.1,0.01,0.1,0.1]
opt.custom_learning_rate = [0.01,0.01,0.01,0.01]

opt.set_handles(init_variables, phi_FRP_model)
opt.scanner_opt_params = opt.init_variables()


opt.train_model_with_restarts(nmb_rnd_restart=15, training_iter=10,do_vis_image=True)
#opt.train_model_with_restarts(nmb_rnd_restart=1, training_iter=1)

#stop()

print('<seq> now (100 iterations with best initialization')
#opt.scanner_opt_params = opt.init_variables()
opt.train_model(training_iter=200, do_vis_image=True)
#opt.train_model(training_iter=10)


#event_time = torch.abs(event_time)  # need to be positive
#opt.scanner_opt_params = init_variables()

_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)
#reco = tonumpy(reco).reshape([sz[0],sz[1],2])

# plot
targetSeq.print_status(True, reco=None)
opt.print_status(True, reco)

print("e: %f, total flipangle is %f °, total scan time is %f s," % (error, np.abs(tonumpy(opt.scanner_opt_params[0].permute([1,0]))).sum()*180/np.pi, tonumpy(torch.abs(opt.scanner_opt_params[2])[:,:,0].permute([1,0])).sum() ))


stop()

