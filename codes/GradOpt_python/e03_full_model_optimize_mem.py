#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: aloktyus


TODO:
  for loop over NSpins
  for loop over NRep ..

"""

import os, sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch import optim

use_gpu = 1

# NRMSE error function
def e(gt,x):
    return 100*np.linalg.norm((gt-x).ravel())/np.linalg.norm(gt.ravel())

# get magnitude image
def magimg(x):
  return np.sqrt(np.sum(np.abs(x)**2,2))

def get_ramps(sz):
    rampX = np.pi*np.linspace(-1,1,sz[0] + 1)
    rampX = -np.expand_dims(rampX[:-1],1)
    rampX = np.tile(rampX, (1, sz[1]))
    
    rampX = torch.from_numpy(rampX).float()
    rampX = rampX.view([1,1,Nvox])    
    
    # set gradient spatial forms
    rampY = np.pi*np.linspace(-1,1,sz[1] + 1)
    rampY = -np.expand_dims(rampY[:-1],0)
    rampY = np.tile(rampY, (sz[0], 1))
    
    rampY = torch.from_numpy(rampY).float()
    rampY = rampY.view([1,1,Nvox])    
    
    rampX = setdevice(rampX)
    rampY = setdevice(rampY)
    
    return (rampX, rampY)

def init_flip_tensor_holder(T,NRep):
    F = torch.zeros((T,NRep,1,4,4), dtype=torch.float32)
    
    F[:,:,0,3,3] = 1
    F[:,:,0,1,1] = 1
     
    F = setdevice(F)
     
    return F
     
def set_flip_tensor(F,flips,T,NRep):
    
    flips_cos = torch.cos(flips)
    flips_sin = torch.sin(flips)
    
    F[:,:,0,0,0] = flips_cos
    F[:,:,0,0,2] = flips_sin
    F[:,:,0,2,0] = -flips_sin
    F[:,:,0,2,2] = flips_cos 
     

def get_relaxation_tensor(T1,T2,dt,Nvox):
    R = torch.zeros((Nvox,4,4), dtype=torch.float32) 
    
    T2_r = torch.exp(-dt/T2)
    T1_r = torch.exp(-dt/T1)
    
    R[:,3,3] = 1
    
    R[:,0,0] = T2_r
    R[:,1,1] = T2_r
    R[:,2,2] = T1_r
    R[:,2,3] = 1 - T1_r
     
    R = R.view([1,Nvox,4,4])
    
    R = setdevice(R)

    return R

def get_freeprecession_tensor(dB0,NSpins):
    P = torch.zeros((NSpins,1,1,4,4), dtype=torch.float32)
    
    B0_nspins = dB0.view([NSpins])
    
    B0_nspins_cos = torch.cos(B0_nspins)
    B0_nspins_sin = torch.sin(B0_nspins)
     
    P[:,0,0,0,0] = B0_nspins_cos
    P[:,0,0,0,1] = -B0_nspins_sin
    P[:,0,0,1,0] = B0_nspins_sin
    P[:,0,0,1,1] = B0_nspins_cos
     
    P = setdevice(P)

    return P

def init_gradient_tensor_holder(NRep,Nvox):
    G = torch.zeros((NRep,Nvox,4,4), dtype=torch.float32)
    G[:,:,2,2] = 1
    G[:,:,3,3] = 1
     
    G = setdevice(G)
    
    return G
    
def set_grad_op(G,B0_grad_cos,B0_grad_sin,isConj):
    
    if isConj:
        sign = -1
    else:
        sign = 1
    
    G[:,:,0,0] = B0_grad_cos
    G[:,:,0,1] = -sign*B0_grad_sin
    G[:,:,1,0] = sign*B0_grad_sin
    G[:,:,1,1] = B0_grad_cos
    
def get_gradient_precession_tensor(t,grads,rampX,rampY,NRep,Nvox):
    
    B0X = torch.unsqueeze(torch.unsqueeze(grads[t,:,0],0),2) * rampX
    B0Y = torch.unsqueeze(torch.unsqueeze(grads[t,:,1],0),2) * rampY
    
    B0_grad = (B0X + B0Y).view([1,NRep,Nvox])
    
    B0_grad_cos = torch.cos(B0_grad)
    B0_grad_sin = torch.sin(B0_grad)
    
    return (B0_grad_cos, B0_grad_sin)

def get_initial_magnetization(PD,NSpins,NRep,Nvox):
    M0 = torch.zeros((NSpins,NRep,Nvox,4), dtype=torch.float32)
    M0[:,:,:,2:] = 1
    M0[:,:,:,2] = M0[:,:,:,2] * PD.view([Nvox])    # weight by proton density
    
    M0 = setdevice(M0)
    
    return M0

def setdevice(x):
    if use_gpu:
        x = x.cuda(0)
        
    return x

# define setup

sz = np.array([48,48])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0]                                            # number of events F/R/P
NSpins = 2
dt = 0.0001                         # time interval between actions (seconds)

m = np.load('../../data/phantom.npy')
m = cv2.resize(m, dsize=(sz[0], sz[1]), interpolation=cv2.INTER_CUBIC)
m = m / np.max(m)

Nvox = sz[0]*sz[1]

# set relaxations (unit - seconds) and proton density
PD = torch.from_numpy(magimg(m).reshape([Nvox])).float()
T1 = torch.ones(Nvox, dtype=torch.float32)*1e6
T2 = torch.ones(Nvox, dtype=torch.float32)*2
T2[0:Nvox/2] = 0.09

# set NSpins offresonance
dB0 = torch.from_numpy(0*(1e-3*np.pi/180)*np.arange(0,NSpins).reshape([NSpins])).float()

# set linear gradient ramps
rampX, rampY = get_ramps(sz)

# set ADC mask
adc_mask = torch.from_numpy(np.ones((T,1))).float()
#adc_mask[:1] = 0

adc_mask = setdevice(adc_mask)

# init tensors
flips = torch.ones((T,NRep), dtype=torch.float32) * 0 * np.pi/180
flips[0,:] = 90*np.pi/180
F = init_flip_tensor_holder(T,NRep)
set_flip_tensor(F,flips,T,NRep)

R = get_relaxation_tensor(T1,T2,dt,Nvox)

P = get_freeprecession_tensor(dB0,NSpins)

# gradient-driver precession
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

grad_moms[:,:,0] = torch.linspace(-sz[0]/2,sz[0]/2-1,T).view(T,1).repeat([1,NRep])
grad_moms[:,:,1] = torch.linspace(-sz[1]/2,sz[1]/2-1,NRep).repeat([T,1])

padder = torch.zeros((1,NRep,2),dtype=torch.float32)
temp = torch.cat((padder,grad_moms),0)
grads = temp[1:,:,:] - temp[:-1,:,:]

grads = setdevice(grads)
grad_moms = setdevice(grad_moms)

G = init_gradient_tensor_holder(NRep,Nvox)

## Forward model :::
    
# init signal holder
signal = torch.zeros((T,NRep,4,1), dtype=torch.float32) 
signal[:,:,3,0] = 1
      
signal = setdevice(signal)

# initialize magnetization vector
M_init = get_initial_magnetization(PD,NSpins,NRep,Nvox)
M = M_init.clone().view([NSpins,NRep,Nvox,4,1])

if False:                                          # fixed flip position case
    # beginning of repetition flip
    flips_init = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
    F1 = init_flip_tensor_holder(1,NRep)
    set_flip_tensor(F1,flips_init,1,NRep)
    M = torch.matmul(F1[0,:,:,:],M)
    
    # relax till ADC
    till_ADC = 0.06                                                     # seconds
    R1 = get_relaxation_tensor(T1,T2,0.06,Nvox)
    M = torch.matmul(R1,M)

for t in range(T):
    
    M = torch.matmul(F[t,:,:,:],M)                                     # Flip
    M = torch.matmul(R,M)                                             # Relax
    
    B0_grad_cos, B0_grad_sin = get_gradient_precession_tensor(t,grads,rampX,rampY,NRep,Nvox)
    
    # gradient-driver precession
    set_grad_op(G,B0_grad_cos[0,:,:],B0_grad_sin[0,:,:],False)
    
    M = torch.matmul(G,M)
    
    # free precession
    M = torch.matmul(P,M)
    
    # ADC -- read sig
    signal[t,:,:,0] = torch.sum(M,[0,2,4]) * adc_mask[t]
          
# init reconstructed image
reco = torch.zeros((Nvox,2), dtype = torch.float32)
reco = setdevice(reco)

for t in range(T-1,-1,-1):
    
    # get kumulative gradients
    B0_grad_cos, B0_grad_sin = get_gradient_precession_tensor(t,grad_moms,rampX,rampY,NRep,Nvox)    
    
    set_grad_op(G,B0_grad_cos[0,:,:],B0_grad_sin[0,:,:],True)
    
    s = signal[t,:,:,:] * adc_mask[t]
    r = torch.matmul(G.permute([1,0,2,3]),s)
    reco = reco + torch.sum(r[:,:,:2,0],1)
    
# try to fit this
target = reco.clone()
   
reco = reco.cpu().numpy().reshape([sz[0],sz[1],2])

plt.imshow(magimg(m))
plt.title('original')
plt.ion()
plt.show()

plt.imshow(magimg(reco))
plt.title('reconstruction')
plt.ion()
plt.show()


# %% optimize

def phi_FRP_model(grads,args):
    
    target,M_init,use_tanh_grad_moms_cap = args
    
    M = M_init.clone()
    M = M.view([NSpins,NRep,Nvox,4,1])
    
    if False:                                      # fixed flip position case
        # initial flip/relax
        M = torch.matmul(F1[0,:,:,:],M)
        M = torch.matmul(R1,M)    
    
    # gradients
    grad_moms = torch.cumsum(grads,0)
    
    if use_tanh_grad_moms_cap:
      boost_fct = 1
        
      fmax = sz / 2
      for i in [0,1]:
          grad_moms[:,:,i] = boost_fct*fmax[i]*torch.tanh(grad_moms[:,:,i])
          
    padder = torch.zeros((1,NRep,2),dtype=torch.float32)
    padder = setdevice(padder)
    temp = torch.cat((padder,grad_moms),0)
    grads_cap = temp[1:,:,:] - temp[:-1,:,:]          
          
    # init signal 
    signal = torch.zeros((T,NRep,4,1), dtype=torch.float32) 
    signal[:,:,3,0] = 1
          
    signal = setdevice(signal)
          
    G = init_gradient_tensor_holder(NRep,Nvox)
          
    for t in range(T):
        
        M = torch.matmul(F[t,:,:,:],M)   # Flip
        M = torch.matmul(R,M)      # Relax
        
        B0_grad_cos, B0_grad_sin = get_gradient_precession_tensor(t,grads_cap,rampX,rampY,NRep,Nvox)
        
        # gradient-driver precession
        set_grad_op(G,B0_grad_cos[0,:,:],B0_grad_sin[0,:,:],False)

        M = torch.matmul(G,M)
        M = torch.matmul(P,M)     # free precession
         
        # ADC -- read sig
        signal[t,:,:,0] = torch.sum(M,[0,2,4]) * adc_mask[t]
      
    reco = torch.zeros((Nvox,2), dtype = torch.float32)
    reco = setdevice(reco)
    
    for t in range(T-1,-1,-1):
        
        B0_grad_cos, B0_grad_sin = get_gradient_precession_tensor(t,grad_moms,rampX,rampY,NRep,Nvox)
        
        set_grad_op(G,B0_grad_cos[0,:,:],B0_grad_sin[0,:,:],True)
        
        s = signal[t,:,:,:] * adc_mask[t]
        r = torch.matmul(G.permute([1,0,2,3]),s)
        reco = reco + torch.sum(r[:,:,:2,0],1)
        
    loss = (reco - target)
    phi = torch.sum((1.0/Nvox)*torch.abs(loss.squeeze())**2)
    
    return (phi,reco)

# init gradients

g = np.random.rand(T,NRep,2) - 0.5
#g = g.ravel()
grads = torch.from_numpy(g).float()
grads = setdevice(grads)
grads.requires_grad = True


target = target.detach()

use_tanh_grad_moms_cap = 1
learning_rate = 0.1

args = (target,M_init,use_tanh_grad_moms_cap)

optimizer = optim.LBFGS([grads], lr=learning_rate, max_iter=1,history_size=200)

def weak_closure():
    optimizer.zero_grad()
    loss,_ = phi_FRP_model(grads, args)
    loss.backward()
    
    return loss

target_numpy = target.cpu().numpy().reshape([sz[0],sz[1],2])

training_iter = 80
for i in range(training_iter):
    optimizer.step(weak_closure)
    
    _,reco = phi_FRP_model(grads, args)
    
    reco = reco.detach().cpu().numpy().reshape([sz[0],sz[1],2])
    
    print("error =%f" %e(target_numpy.ravel(),reco.ravel()))
    

phi,reco = phi_FRP_model(grads, args)

reco = reco.detach().cpu().numpy().reshape([sz[0],sz[1],2])

plt.imshow(magimg(reco))
plt.title('reconstruction')











