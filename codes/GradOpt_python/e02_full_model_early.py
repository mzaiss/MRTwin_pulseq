#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: aloktyus
"""

import os, sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch import optim

# NRMSE error function
def e(gt,x):
    return 100*np.linalg.norm((gt-x).ravel())/np.linalg.norm(gt.ravel())

# %% full model early spec

NRep = 10                                             # number of repetitions
T = 8                                                # number of events F/R/P
sz = np.array([4,6])                                             # image size
NSpins = 12

Nvox = sz[0]*sz[1]

# set relaxations and proton densities
PD = torch.rand(Nvox, dtype=torch.float32)
T1 = torch.rand(Nvox, dtype=torch.float32)
T2 = torch.rand(Nvox, dtype=torch.float32)

# set NSpins offresonance
dB0 = torch.rand(NSpins, dtype=torch.float32)

# set gradient spatial forms
rampX = np.pi*np.linspace(-1,1,sz[0] + 1)
rampX = np.expand_dims(rampX[:-1],1)
rampX = np.tile(rampX, (1, sz[1]))

# set gradient spatial forms
rampY = np.pi*np.linspace(-1,1,sz[1] + 1)
rampY = np.expand_dims(rampY[:-1],0)
rampY = np.tile(rampY, (sz[0], 1))

# set ADC mask
adc_mask = np.ones((T,1) )
adc_mask[:1] = 0

M0 = torch.zeros((Nvox,4), dtype=torch.float32)
M0[:,2:] = 1
  
F = torch.zeros((T,NRep,1,4,4), dtype=torch.float32)                         # FLIP pperator
R = torch.zeros((Nvox,4,4), dtype=torch.float32)                             # RELAXATION operator
P = torch.zeros((T,NSpins,NRep,Nvox,4,4), dtype=torch.float32)               # PRECESSION operator: split into free/nonfree parts

# init tensors

# flips

F[:,:,0,3,3] = 1
F[:,:,0,1,1] = 1

flips = torch.zeros((T,NRep), dtype=torch.float32)

flips_cos = torch.cos(flips)
flips_sin = torch.sin(flips)

F[:,:,0,0,0] = flips_cos
F[:,:,0,0,2] = flips_sin
F[:,:,0,2,0] = -flips_sin
F[:,:,0,2,2] = flips_cos
 
# relaxations
dt = 1
T2_r = torch.exp(-dt/T2)
T1_r = torch.exp(-dt/T1)

R[:,3,3] = 1

R[:,0,0] = T2_r
R[:,1,1] = T2_r
R[:,2,2] = T1_r
R[:,2,3] = 1 - T1_r
 
R = R.view([1,Nvox,4,4])

# precession
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

rampX = torch.from_numpy(rampX).float()
rampX = rampX.view([1,1,Nvox])

rampY = torch.from_numpy(rampY).float()
rampY = rampX.view([1,1,Nvox])

B0X = torch.unsqueeze(grad_moms[:,:,0],2) * rampX
B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * rampY

B0_grad = (B0X + B0Y).view([T,1,NRep,Nvox])

B0_nspins = dB0.view([1,NSpins,1,1])

B0 = B0_grad + B0_nspins

B0_cos = torch.cos(B0)
B0_sin = torch.sin(B0)

P[:,:,:,:,2,2] = 1
P[:,:,:,:,3,3] = 1
 
P[:,:,:,:,0,0] = B0_cos
P[:,:,:,:,0,1] = -B0_sin
P[:,:,:,:,1,0] = B0_sin
P[:,:,:,:,1,1] = B0_cos

# Forward model  
signal = torch.zeros((T,NRep,2), dtype=torch.float32) 

# weight by proton density
M_init = M0*PD.view([Nvox,1])

M = M_init.clone()
M = M.view([Nvox,4,1])

for t in range(T):
    M = torch.matmul(F[t,:,:,:],M)
    M = torch.matmul(R,M)
    M = torch.matmul(P[t,:,:,:,:,:],M)

# %% easy sim

def magimg(x):
  return np.sqrt(np.sum(np.abs(x)**2,2))

sz = np.array([16,16])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0]                                            # number of events F/R/P
NSpins = 10

m = np.load('../../data/phantom.npy')
m = cv2.resize(m, dsize=(sz[0], sz[1]), interpolation=cv2.INTER_CUBIC)
m = m / np.max(m)

Nvox = sz[0]*sz[1]

# set relaxations and proton densities
PD = torch.from_numpy(magimg(m).reshape([Nvox])).float()
T1 = torch.ones(Nvox, dtype=torch.float32)*1e6
T2 = torch.ones(Nvox, dtype=torch.float32)*2                        # seconds
T2[0:Nvox/2] = 0.09

# set NSpins offresonance
#dB0 = torch.zeros(NSpins, dtype=torch.float32)
dB0 = torch.from_numpy(0*(1e-3*np.pi/180)*np.arange(0,NSpins).reshape([NSpins])).float()

# set gradient spatial forms
rampX = np.pi*np.linspace(-1,1,sz[0] + 1)
rampX = np.expand_dims(rampX[:-1],1)
rampX = np.tile(rampX, (1, sz[1]))

# set gradient spatial forms
rampY = np.pi*np.linspace(-1,1,sz[1] + 1)
rampY = np.expand_dims(rampY[:-1],0)
rampY = np.tile(rampY, (sz[0], 1))

# set ADC mask
adc_mask = np.ones((T,1) )
#adc_mask[:1] = 0

F = torch.zeros((T,NRep,1,4,4), dtype=torch.float32)                         # FLIP pperator
R = torch.zeros((Nvox,4,4), dtype=torch.float32)                             # RELAXATION operator
G = torch.zeros((NRep,Nvox,4,4), dtype=torch.float32)                        # GRADIENT PRECESSION operator
P = torch.zeros((NSpins,1,1,4,4), dtype=torch.float32)                       # FREE PRECESSION operator: due to off-resonance component

# init tensors

# flips
F[:,:,0,3,3] = 1
F[:,:,0,1,1] = 1

flips = torch.ones((T,NRep), dtype=torch.float32) * 0 * np.pi/180

flips_cos = torch.cos(flips)
flips_sin = torch.sin(flips)

F[:,:,0,0,0] = flips_cos
F[:,:,0,0,2] = flips_sin
F[:,:,0,2,0] = -flips_sin
F[:,:,0,2,2] = flips_cos
 
# relaxations
dt = 0.0001                                                         # seconds
T2_r = torch.exp(-dt/T2)
T1_r = torch.exp(-dt/T1)

R[:,3,3] = 1

R[:,0,0] = T2_r
R[:,1,1] = T2_r
R[:,2,2] = T1_r
R[:,2,3] = 1 - T1_r
 
R = R.view([1,Nvox,4,4])

# precession

# non-free
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

grad_moms[:,:,0] = torch.linspace(-sz[0]/2,sz[0]/2-1,T).view(T,1).repeat([1,NRep])
grad_moms[:,:,1] = torch.linspace(-sz[1]/2,sz[1]/2-1,NRep).repeat([T,1])

temp = torch.cat((torch.zeros((1,NRep,2),dtype=torch.float32),grad_moms),0)
grads = temp[1:,:,:] - temp[:-1,:,:]

rampX_t = torch.from_numpy(rampX).float()
rampX_t = rampX_t.view([1,1,Nvox])

rampY_t = torch.from_numpy(rampY).float()
rampY_t = rampY_t.view([1,1,Nvox])

B0X = torch.unsqueeze(grads[:,:,0],2) * rampX_t
B0Y = torch.unsqueeze(grads[:,:,1],2) * rampY_t

B0_grad = (B0X + B0Y).view([T,NRep,Nvox])

B0_grad_cos = torch.cos(B0_grad)
B0_grad_sin = torch.sin(B0_grad)

G[:,:,2,2] = 1
G[:,:,3,3] = 1
 
# free precession
B0_nspins = dB0.view([NSpins])

B0_nspins_cos = torch.cos(B0_nspins)
B0_nspins_sin = torch.sin(B0_nspins)
 
P[:,0,0,0,0] = B0_nspins_cos
P[:,0,0,0,1] = -B0_nspins_sin
P[:,0,0,1,0] = B0_nspins_sin
P[:,0,0,1,1] = B0_nspins_cos

# Forward model  
signal = torch.zeros((T,NRep,4,1), dtype=torch.float32) 
signal[:,:,3,0] = 1

M0 = torch.zeros((NSpins,NRep,Nvox,4), dtype=torch.float32)
M0[:,:,:,2:] = 1
M0[:,:,:,2] = M0[:,:,:,2] * PD.view([Nvox])

# weight by proton density
M_init = M0

M = M_init.clone()
M = M.view([NSpins,NRep,Nvox,4,1])

# beginning of repetition flip
FF = torch.zeros((1,NRep,1,4,4), dtype=torch.float32)
FF[:,:,0,3,3] = 1
FF[:,:,0,1,1] = 1

flips = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
flips_cos = torch.cos(flips)
flips_sin = torch.sin(flips)
FF[:,:,0,0,0] = flips_cos
FF[:,:,0,0,2] = flips_sin
FF[:,:,0,2,0] = -flips_sin
FF[:,:,0,2,2] = flips_cos

M = torch.matmul(FF[0,:,:,:],M)

# relax till ADC

RR = torch.zeros((Nvox,4,4), dtype=torch.float32)
fdt = 0.06                                                          # seconds
T2_r = torch.exp(-fdt/T2)
T1_r = torch.exp(-fdt/T1)
RR[:,3,3] = 1
RR[:,0,0] = T2_r
RR[:,1,1] = T2_r
RR[:,2,2] = T1_r
RR[:,2,3] = 1 - T1_r
RR = RR.view([1,Nvox,4,4])

M = torch.matmul(RR,M)

# flip
F[1:,:,0,0,0] = 1
F[1:,:,0,0,2] = 0
F[1:,:,0,2,0] = 0
F[1:,:,0,2,2] = 1

for t in range(T):
    
    # Flip
    M = torch.matmul(F[t,:,:,:],M)
    # Relax
    M = torch.matmul(R,M)
    
    # gradient-driver precession
    G[:,:,0,0] = B0_grad_cos[t,:,:]
    G[:,:,0,1] = -B0_grad_sin[t,:,:]
    G[:,:,1,0] = B0_grad_sin[t,:,:]
    G[:,:,1,1] = B0_grad_cos[t,:,:]
    
    M = torch.matmul(G,M)
    
    # free precession
    M = torch.matmul(P,M)
    
    # ADC -- read sig
    s = torch.sum(M,[0,2,4])
    signal[t,:,:,0] = s
          
B0X = torch.unsqueeze(grad_moms[:,:,0],2) * rampX_t
B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * rampY_t
B0_grad = (B0X + B0Y).view([T,NRep,Nvox])

B0_grad = B0_grad.permute([0,2,1])

B0_grad_cos = torch.cos(B0_grad)
B0_grad_sin = torch.sin(B0_grad)

G = torch.zeros((Nvox,NRep,4,4), dtype=torch.float32)

reco = torch.zeros((Nvox,2), dtype = torch.float32)

for t in range(T-1,-1,-1):
    
    G[:,:,0,0] = B0_grad_cos[t,:,:]
    G[:,:,0,1] = B0_grad_sin[t,:,:]
    G[:,:,1,0] = -B0_grad_sin[t,:,:]
    G[:,:,1,1] = B0_grad_cos[t,:,:]    
    
    r = torch.matmul(G,signal[t,:,:,:])
    reco = reco + torch.sum(r[:,:,:2,0],1)
    
target = reco.clone()
   
reco = reco.cpu()
reco = reco.numpy().reshape([sz[0],sz[1],2])

plt.imshow(magimg(m))
plt.title('original')
plt.ion()
plt.show()

plt.imshow(magimg(reco))
plt.title('reconstruction')
plt.ion()
plt.show()

#hgfhgf


# %% optimize

def phi_FRP_model(grads,args):
    
    target,M_init,use_tanh_grad_moms_cap = args
    
    M = M_init.clone()
    M = M.view([NSpins,NRep,Nvox,4,1])
    
    
    
    
    # beginning of repetition flip
    FF = torch.zeros((1,NRep,1,4,4), dtype=torch.float32)
    FF[:,:,0,3,3] = 1
    FF[:,:,0,1,1] = 1
    
    flips = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
    flips_cos = torch.cos(flips)
    flips_sin = torch.sin(flips)
    FF[:,:,0,0,0] = flips_cos
    FF[:,:,0,0,2] = flips_sin
    FF[:,:,0,2,0] = -flips_sin
    FF[:,:,0,2,2] = flips_cos
    
    M = torch.matmul(FF[0,:,:,:],M)
    
    # relax till ADC
    
    RR = torch.zeros((Nvox,4,4), dtype=torch.float32)
    fdt = 0.06                                                          # seconds
    T2_r = torch.exp(-fdt/T2)
    T1_r = torch.exp(-fdt/T1)
    RR[:,3,3] = 1
    RR[:,0,0] = T2_r
    RR[:,1,1] = T2_r
    RR[:,2,2] = T1_r
    RR[:,2,3] = 1 - T1_r
    RR = RR.view([1,Nvox,4,4])
    
    M = torch.matmul(RR,M)    
    
    
    
    
    
    # gradients
    rampX_t = torch.from_numpy(rampX).float()
    rampX_t = rampX_t.view([1,1,Nvox])
    
    rampY_t = torch.from_numpy(rampY).float()
    rampY_t = rampY_t.view([1,1,Nvox])
    
    grad_moms = torch.cumsum(grads,0)
    
    if use_tanh_grad_moms_cap:
      boost_fct = 1
        
      fmax = sz / 2
      for i in [0,1]:
          grad_moms[:,:,i] = boost_fct*fmax[i]*torch.tanh(grad_moms[:,:,i])
          
    temp = torch.cat((torch.zeros((1,NRep,2),dtype=torch.float32),grad_moms),0)
    grads_cap = temp[1:,:,:] - temp[:-1,:,:]
    
    B0X = torch.unsqueeze(grads_cap[:,:,0],2) * rampX_t
    B0Y = torch.unsqueeze(grads_cap[:,:,1],2) * rampY_t
    
    B0_grad = (B0X + B0Y).view([T,NRep,Nvox])
    
    B0_grad_cos = torch.cos(B0_grad)
    B0_grad_sin = torch.sin(B0_grad)    
    
    G = torch.zeros((NRep,Nvox,4,4), dtype=torch.float32)
    G[:,:,2,2] = 1
    G[:,:,3,3] = 1   
     
    # init signal 
    signal = torch.zeros((T,NRep,4,1), dtype=torch.float32) 
    signal[:,:,3,0] = 1
    
    for t in range(T):
        
        # Flip
        M = torch.matmul(F[t,:,:,:],M)
        # Relax
        M = torch.matmul(R,M)
        
        # gradient-driver precession
        G[:,:,0,0] = B0_grad_cos[t,:,:]
        G[:,:,0,1] = -B0_grad_sin[t,:,:]
        G[:,:,1,0] = B0_grad_sin[t,:,:]
        G[:,:,1,1] = B0_grad_cos[t,:,:]
        
        M = torch.matmul(G,M)
        
        # free precession
        M = torch.matmul(P,M)
        
        # ADC -- read sig
        s = torch.sum(M,[0,2,4])
        signal[t,:,:,0] = s
              
    B0X = torch.unsqueeze(grad_moms[:,:,0],2) * rampX_t
    B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * rampY_t
    B0_grad = (B0X + B0Y).view([T,NRep,Nvox])
    
    B0_grad = B0_grad.permute([0,2,1])
    
    B0_grad_cos = torch.cos(B0_grad)
    B0_grad_sin = torch.sin(B0_grad)
    
    #G = torch.zeros((Nvox,NRep,4,4), dtype=torch.float32)
    G = G.permute([1,0,2,3])
    G[:,:,2,2] = 1
    G[:,:,3,3] = 1    
    
    reco = torch.zeros((Nvox,2), dtype = torch.float32)
    
    for t in range(T-1,-1,-1):
        
        G[:,:,0,0] = B0_grad_cos[t,:,:]
        G[:,:,0,1] = B0_grad_sin[t,:,:]
        G[:,:,1,0] = -B0_grad_sin[t,:,:]
        G[:,:,1,1] = B0_grad_cos[t,:,:]    
        
        r = torch.matmul(G,signal[t,:,:,:])
        reco = reco + torch.sum(r[:,:,:2,0],1)
        
    loss = (reco - target)
    phi = torch.sum((1.0/Nvox)*torch.abs(loss.squeeze())**2)
    
    return (phi,reco)

M0 = torch.zeros((NSpins,NRep,Nvox,4), dtype=torch.float32)
M0[:,:,:,2:] = 1
M0[:,:,:,2] = M0[:,:,:,2] * PD.view([Nvox])

# weight by proton density
M_init = M0.clone()

# init gradients

g = np.random.rand(T,NRep,2) - 0.5
#g = g.ravel()
grads = torch.from_numpy(g).float()
grads.requires_grad = True

target_numpy = target.detach()
target_numpy = target_numpy.cpu()
target_numpy = target_numpy.numpy().reshape([sz[0],sz[1],2])

use_tanh_grad_moms_cap = 1

args = (target,M_init,use_tanh_grad_moms_cap)

optimizer = optim.LBFGS([grads], max_iter=1,history_size=200)

def weak_closure():
    optimizer.zero_grad()
    loss,_ = phi_FRP_model(grads, args)
    
    loss.backward()
    
    return loss

training_iter = 50
for i in range(training_iter):
    optimizer.step(weak_closure)
    
    _,reco = phi_FRP_model(grads, args)
    
    reco = reco.detach()
    reco = reco.cpu()
    reco = reco.numpy().reshape([sz[0],sz[1],2])
    
    print("error =%f" %e(target_numpy.ravel(),reco.ravel()))
    

phi,reco = phi_FRP_model(grads, args)

reco = reco.detach()
reco = reco.cpu()
reco = reco.numpy().reshape([sz[0],sz[1],2])

plt.imshow(magimg(reco))
plt.title('reconstruction')











