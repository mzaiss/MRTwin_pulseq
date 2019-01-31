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

# %% full model early spec

NRep = 10                                             # number of repetitions
T = 8                                                # number of events F/R/P
sz = [4,6]                                                       # image size
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

sz = [16,16]                                                     # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0]                                            # number of events F/R/P
NSpins = 2

m = np.load('../../data/phantom.npy')
m = cv2.resize(m, dsize=(sz[0], sz[1]), interpolation=cv2.INTER_CUBIC)
m = m / np.max(m)

Nvox = sz[0]*sz[1]

# set relaxations and proton densities
PD = torch.from_numpy(magimg(m).reshape([Nvox])).float()
T1 = torch.ones(Nvox, dtype=torch.float32)*1e6
T2 = torch.ones(Nvox, dtype=torch.float32)*1e6

# set NSpins offresonance
dB0 = torch.zeros(NSpins, dtype=torch.float32)

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
P = torch.zeros((NSpins,Nvox,4,4), dtype=torch.float32)                      # FREE PRECESSION operator: due to off-resonance component

# init tensors

# flips
F[:,:,0,3,3] = 1
F[:,:,0,1,1] = 1

flips = torch.ones((T,NRep), dtype=torch.float32) * 90 * np.pi/180

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
 
# free
B0_nspins = dB0.view([NSpins,1])

B0_nspins_cos = torch.cos(B0_nspins)
B0_nspins_sin = torch.sin(B0_nspins)
 
P[:,:,0,0] = B0_nspins_cos
P[:,:,0,1] = -B0_nspins_sin
P[:,:,1,0] = B0_nspins_sin
P[:,:,1,1] = B0_nspins_cos

# Forward model  
signal = torch.zeros((T,1,NRep,4,1), dtype=torch.float32) 
signal[:,0,:,3,0] = 1

M0 = torch.zeros((Nvox,4), dtype=torch.float32)
M0[:,2:] = 1
M0[:,2] = M0[:,2] * PD.view([Nvox])

# weight by proton density
M_init = M0

M = M_init.clone()
M = M.view([Nvox,4,1])

# flip
M = torch.matmul(F[0,:,:,:],M)

for t in range(T):
    
    G[:,:,0,0] = B0_cos[t,:,:]
    G[:,:,0,1] = -B0_sin[t,:,:]
    G[:,:,1,0] = B0_sin[t,:,:]
    G[:,:,1,1] = B0_cos[t,:,:]
    
    M = torch.matmul(G,M)
    
    s = torch.sum(M,[0,2,4])
    signal[t,0,:,:,0] = s
    
B0X = torch.unsqueeze(grad_moms[:,:,0],2) * rampX_t
B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * rampY_t
B0_grad = (B0X + B0Y).view([T,NRep,Nvox])

B0 = B0_grad + B0_nspins
B0_cos = torch.cos(B0)
B0_sin = torch.sin(B0)

P[:,:,0,0] = B0_cos
P[:,:,0,1] = -B0_sin
P[:,:,1,0] = B0_sin
P[:,:,1,1] = B0_cos
          
P = P.permute([0,1,3,2,5,4])
P = P.view([T,Nvox,NRep,4,4])

reco = torch.zeros((Nvox,2), dtype = torch.float32)

for t in range(T-1,-1,-1):
    r = torch.matmul(P[t,:,:,:,:],signal[t,:,:,:,:])
    reco = reco + torch.sum(r[:,:,:2,0],1)
    
   
reco = reco.cpu()
reco = reco.numpy().reshape([sz[0],sz[1],2])

plt.imshow(magimg(m))
plt.title('original')
plt.ion()
plt.show()

plt.imshow(magimg(reco))
plt.title('reconstruction')

