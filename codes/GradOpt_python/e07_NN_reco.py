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

import core.spins
import core.scanner

if sys.version_info[0] < 3:
    reload(core.spins)
    reload(core.scanner)

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

dir_data = '/agbs/cpr/mr_motion/RIMphase/data/'
fn_data_tensor = 'T1w_10subjpack_16x16_cmplx.npy'
fn_tgt_tensor = "T1w_10subjpack_16x16_tgt_cmplx.npy"

data_tensor_numpy_cmplx = np.load(os.path.join(dir_data, fn_data_tensor))
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx / np.max(data_tensor_numpy_cmplx)

tgt_tensor_numpy_cmplx = np.load(os.path.join(dir_data, fn_tgt_tensor))

ssz = data_tensor_numpy_cmplx.shape
data_tensor_numpy_cmplx = data_tensor_numpy_cmplx.reshape([ssz[0]*ssz[1],ssz[2],ssz[3],ssz[4]])

subjid = 160

batch_size = 24

# initialize scanned object
spins = core.spins.SpinSystem_batched(sz,NVox,NSpins,batch_size,use_gpu)

batch_idx = np.random.choice(data_tensor_numpy_cmplx.shape[0],batch_size,replace=False)
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

img_id = 1

if False:                                                      # check sanity
    plt.imshow(magimg(spins.images[img_id,:,:,:]))
    plt.title('original')
    plt.ion()
    plt.show()
    
    plt.imshow(magimg(reco[img_id,:,:,:]))
    plt.title('reconstruction')
    plt.ion()
    plt.show()
    
    gfdgfdfd


# %% ###     OPTIMIZE ######################################################@
#############################################################################


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.lspec = [2,64,64,64,2]
        self.conv_layers = []
        self.bn_layers = []
        
        for l_idx in range(len(self.lspec)-1):
            layer = nn.Conv2d(self.lspec[l_idx], self.lspec[l_idx+1], 3, padding=1)
            self.conv_layers.append(layer)
            
            weight = torch.zeros([self.lspec[l_idx+1],self.lspec[l_idx],3,3]).cuda()
            
            weight[0,0,1,1] = 1
                  
            #layer.weight.data = weight
            layer = layer.cuda()
            
            bnlayer = nn.BatchNorm2d(self.lspec[l_idx+1])
            self.bn_layers.append(bnlayer)
            
            bnlayer = bnlayer.cuda()
            
            
        self.conv0 = self.conv_layers[0]
        self.conv1 = self.conv_layers[1]
        self.conv2 = self.conv_layers[2]
        self.conv3 = self.conv_layers[3]
        
        self.conv0_bn = self.bn_layers[0]
        self.conv1_bn = self.bn_layers[1]
        self.conv2_bn = self.bn_layers[2]
        self.conv3_bn = self.bn_layers[3]
        

    def forward(self, x):
        x = x.permute([0,2,1])
        x = x.view([batch_size,2,sz[0],sz[1]])
        
        normfact = 600.0
        
        x = x / normfact
        
        for l_idx in range(len(self.lspec)-1):
            
            x = self.conv_layers[l_idx](x)
            
            if l_idx < len(self.lspec) - 2:
                x = self.bn_layers[l_idx](x)
                x = torch.relu(x)
        
        x = x * normfact
        
        x = x.view([batch_size,2,sz[0]*sz[1]])
        x = x.permute([0,2,1])
        
        return x
    
    
class CTRL():
    def __init__(self):
        self.globepoch = 0
        self.subjidx = 0
        
    def new(self):
        self.globepoch += 1
        
        #if self.globepoch % 1:
        self.subjidx = np.random.choice(data_tensor_numpy_cmplx.shape[0], batch_size)
        #self.subjidx = np.random.choice(12, batch_size)
        #self.subjidx = np.random.randint(12)
        #self.subjidx = 0

        
ctrl = CTRL()
    
    
NN = ConvNet().cuda()
#NN.requires_grad = True

def phi_FRP_model(flips,grads,event_time,args):
    
    scanner,spins,target,use_tanh_grad_moms_cap,opti_mode = args

    ctrl.new()
    spins.set_system(images=data_tensor_numpy_cmplx[ctrl.subjidx,:,:,:])
    
    scanner.init_signal()
    
    target = torch.from_numpy(tgt_tensor_numpy_cmplx[ctrl.subjidx,:,:,:].reshape([batch_size,NVox,2])).float().cuda()
    
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
    
    normfact = 1.0
    
    if opti_mode == 'nn' or opti_mode == 'seqnn':
        scanner.reco = scanner.reco/normfact
        scanner.reco = NN(scanner.reco)
            
    loss = normfact*(scanner.reco - target/normfact)
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
    
    event_time = torch.from_numpy(np.zeros((scanner.T,1))).float()
    event_time = setdevice(event_time)
    event_time.requires_grad = True
    
    return flips, grads, event_time
    

#target = target.detach()
#target_numpy = target.cpu().numpy().reshape([sz[0],sz[1],2])

def train_model(doRestart=False, best_vars=None):
    
    # init gradients and flip events
    if doRestart:
        nmb_outer_iter = nmb_rnd_restart
        nmb_inner_iter = training_iter_restarts
        
        best_error = 200
        best_vars = []        
    else:
        nmb_outer_iter = 1
        nmb_inner_iter = training_iter
        
        flips,grads,event_time = best_vars
        flips.requires_grad = True
        grads.requires_grad = True
        event_time.requires_grad = True
        
        
    def weak_closure():
        optimizer.zero_grad()
        loss,_ = phi_FRP_model(flips, grads, event_time, args)
        #print(loss.item())
        loss.backward()
        
        return loss
    
    for outer_iter in range(nmb_outer_iter):
        
        if doRestart:
            flips,grads,event_time = init_variables()
            
        if args[4] == 'seq':
            optimizer = optim.LBFGS([flips,grads,event_time], lr=learning_rate, max_iter=1,history_size=200)
        elif args[4] == 'nn':
            optimizer = optim.LBFGS(list(NN.parameters()), lr=learning_rate, max_iter=1,history_size=200)
        elif args[4] == 'seqnn':
            optimizer = optim.LBFGS(list(NN.parameters())+list([flips,grads,event_time]), lr=learning_rate, max_iter=1,history_size=200)
        
        for inner_iter in range(nmb_inner_iter):
            optimizer.step(weak_closure)
            
            target_numpy = tgt_tensor_numpy_cmplx[ctrl.subjidx,:,:,:].reshape([batch_size,NVox,2])
            
            _,reco = phi_FRP_model(flips, grads, event_time, args)
            reco = reco.detach().cpu().numpy().reshape([batch_size,sz[0],sz[1],2])
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
        return reco,flips,grads,event_time
    
# %% #
    

use_tanh_grad_moms_cap = 1                 # do not sample above Nyquist flag
learning_rate = 0.1                                         # LBFGS step size
training_iter = 150
nmb_rnd_restart = 15
training_iter_restarts = 10



learning_rate = 1.0

opti_mode = 'seq'
args = (scanner,spins,None,use_tanh_grad_moms_cap,opti_mode)
best_vars = train_model(True)

hfhgfgfhgf

reco,flips,grads,event_time = train_model(False,best_vars)

if True:
    print('<nn> now')
    best_vars = flips,grads,event_time
    training_iter = 50
    opti_mode = 'nn'
    args = (scanner,spins,None,use_tanh_grad_moms_cap,opti_mode)
    reco,flips,grads,event_time = train_model(False,best_vars)

if True:
    #learning_rate = 0.05                                         # LBFGS step size
    print('<seqnn> now')
    best_vars = flips,grads,event_time
    training_iter = 150
    opti_mode = 'seqnn'
    args = (scanner,spins,None,use_tanh_grad_moms_cap,opti_mode)
    reco,flips,grads,event_time = train_model(False,best_vars)


target_numpy = tgt_tensor_numpy_cmplx[ctrl.subjidx,:,:,:].reshape([batch_size,NVox,2])
event_time = torch.abs(event_time)

plt.imshow(magimg(target_numpy[0,:,:].reshape([sz[0],sz[1],2])))
plt.title('target')
plt.ion()
plt.show()

plt.imshow(magimg(reco[0,:,:]))
plt.title('reconstruction')











