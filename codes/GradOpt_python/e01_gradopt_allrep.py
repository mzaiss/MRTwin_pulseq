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

use_gpu = 0

# forward Fourier transform
def fftfull(x):                                                 # forward FFT
    return np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(x)))

def ifftfull(x):                                                # inverse FFT
    return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(x)))

# NRMSE error function
def e(gt,x):
    return 100*np.linalg.norm((gt-x).ravel())/np.linalg.norm(gt.ravel())

# Check numerical vs. analytical derivatives : optimize all reps simultaneously
T = 4                                     # number of time points in readout                                                                                                  
sz = np.array([4, 6])                                   # image size (Nx Ny)          
NRep = 8

# set gradient spatial forms
rampX = np.pi*np.linspace(-1,1,sz[0] + 1)
rampX = np.expand_dims(rampX[:-1],1)
rampX = np.tile(rampX, (1, sz[1]))

# set gradient spatial forms
rampY = np.pi*np.linspace(-1,1,sz[1] + 1)
rampY = np.expand_dims(rampY[:-1],0)
rampY = np.tile(rampY, (sz[0], 1))

adc_mask = np.ones((T,1) )
adc_mask[:1] = 0

# initialize gradients (X/Y directions)
g = np.random.rand(NRep,T,2)
g = g.ravel()

# initialize complex-valued magnetization image
m = np.random.rand(sz[0],sz[1]) + 1j*np.random.rand(sz[0],sz[1])

use_tanh_grad_moms_cap = 1                                                   # otherwise put L2 penalty on the grad_moms controlled by lbd
lbd = 0*1e-2

def phi_grad_allrep_readout2d_compact_torch(g,args):
    
    m,rampX,rampY,adc_mask,sz,NRep,lbd,use_tanh_grad_moms_cap,use_gpu = args
    
    T = adc_mask.size()[1]
    N = np.prod(sz)             # Fourier transform normalization coefficient
    g = torch.reshape(g,[NRep,T,2])
    
    E = torch.zeros((NRep,T,2,N,2), dtype = torch.float32)
    if use_gpu:
        E = E.cuda()
    
    phi = 0
    
    grad_moms = torch.cumsum(g,1)
    
    if use_tanh_grad_moms_cap:
      boost_fct = 1
        
      fmax = sz / 2
      for i in [0,1]:
          grad_moms[:,:,i] = boost_fct*fmax[i]*torch.tanh(grad_moms[:,:,i])  # soft threshold
          
      #grad_moms = torch.clamp(grad_moms,-fmax[0],fmax[0])
          
          # this part makes no sense, we cant exceed fmax anyway
          # still, would be good to deal with smooth tanh part
          
          #for rep in range(NRep):
          #    grad_moms[rep,torch.abs(grad_moms[rep,:,i]) > fmax[i],i] = torch.sign(grad_moms[rep,torch.abs(grad_moms[rep,:,i]) > fmax[i],i])*fmax[i]  # hard threshold, this part is nondifferentiable        
              
    B0X = torch.unsqueeze(grad_moms[:,:,0],2) * rampX
    B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * rampY
                         
    B0 = B0X + B0Y
    
    E_cos = torch.cos(B0)
    E_sin = torch.sin(B0)
  
    # encoding operator (ignore relaxation)
    E[:,:,0,:,0] = E_cos
    E[:,:,0,:,1] = -E_sin
    E[:,:,1,:,0] = E_sin
    E[:,:,1,:,1] = E_cos
     
    E = E * adc_mask
    
    fwd = torch.matmul(E.view(NRep,T,2,N*2),m.view(N*2))
    inv = torch.matmul(E.permute([3,4,0,1,2]).view(N,2,NRep*T*2),fwd.view(NRep*T*2,1)).squeeze()
    inv = inv / N
    
    reco = inv.clone()
    
    loss = (inv - m)
    phi = torch.sum(torch.abs(loss.squeeze())**2)
    
    return (phi,reco)

g_t = torch.from_numpy(g).float()
g_t.requires_grad = True
m_t = torch.from_numpy(np.stack((np.real(m), np.imag(m)),2)).float()
rampX_t = torch.from_numpy(rampX).float()
rampY_t = torch.from_numpy(rampY).float()
adc_mask_t = torch.from_numpy(adc_mask).float()

# vectorize ramps and the image
rampX_t = torch.unsqueeze(torch.unsqueeze(rampX_t.flatten(),0),0)
rampY_t = torch.unsqueeze(torch.unsqueeze(rampY_t.flatten(),0),0)
m_t = m_t.view([np.prod(sz),2])
adc_mask_t = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(adc_mask_t,0),2),3)

args = (m_t,rampX_t,rampY_t,adc_mask_t,sz,NRep,lbd,use_tanh_grad_moms_cap,use_gpu)
phi,_ = phi_grad_allrep_readout2d_compact_torch(g_t, args)

phi.backward()

dg_ana = (g_t.grad).numpy()

print("torch phi =%f" %phi)

## easy sim, Cartesian grid

sz = np.array([16, 16])                                  # image size (Nx Ny)          
T = sz[0]                                  # number of time points in readout                                                                                                  
NRep = sz[1]

# set gradient spatial forms
rampX = np.pi*np.linspace(-1,1,sz[0] + 1)
rampX = -np.expand_dims(rampX[:-1],1)
rampX = np.tile(rampX, (1, sz[1]))

# set gradient spatial forms
rampY = np.pi*np.linspace(-1,1,sz[1] + 1)
rampY = -np.expand_dims(rampY[:-1],0)
rampY = np.tile(rampY, (sz[0], 1))

adc_mask = np.ones((T,1) )
#adc_mask[:1] = 0

# initialize gradients (X/Y directions)
g = np.random.rand(NRep,T,2)
g = g.ravel()

# initialize complex-valued magnetization image
m = np.load('../../data/phantom.npy')

m = cv2.resize(m, dsize=(sz[0], sz[1]), interpolation=cv2.INTER_CUBIC)

g_t = torch.from_numpy(g).float()
g_t.requires_grad = True
m_t = torch.from_numpy(m).float()
rampX_t = torch.from_numpy(rampX).float()
rampY_t = torch.from_numpy(rampY).float()
adc_mask_t = torch.from_numpy(adc_mask).float()

# vectorize ramps and the image
rampX_t = torch.unsqueeze(torch.unsqueeze(rampX_t.flatten(),0),0)
rampY_t = torch.unsqueeze(torch.unsqueeze(rampY_t.flatten(),0),0)
m_t = m_t.view([np.prod(sz),2])
adc_mask_t = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(adc_mask_t,0),2),3)

use_tanh_grad_moms_cap = 1                                                   # otherwise put L2 penalty on the grad_moms controlled by lbd
lbd = 0*1e-2

grad_moms = torch.zeros(NRep,T,2, dtype=torch.float32)
                       
grad_moms[:,:,0] = torch.linspace((-sz[0]/2),(sz[0]/2-1),int(T)).repeat([NRep,1])
grad_moms[:,:,1] = torch.linspace(-sz[1]/2,sz[1]/2-1,int(NRep)).view(NRep,1).repeat([1,T])
    
temp = torch.cat((torch.zeros((NRep,1,2),dtype=torch.float32),grad_moms),1)
g_t = temp[:,1:,:] - temp[:,:-1,:]
    
use_tanh_grad_moms_cap = 0

lbd = 0

use_gpu = 1

if use_gpu:
    g_t = g_t.cuda()
    m_t = m_t.cuda()
    rampX_t = rampX_t.cuda()
    rampY_t = rampY_t.cuda()
    adc_mask_t = adc_mask_t.cuda()

args = (m_t,rampX_t,rampY_t,adc_mask_t,sz,NRep,lbd,use_tanh_grad_moms_cap,use_gpu)
out,reco = phi_grad_allrep_readout2d_compact_torch(g_t,args)

reco = reco.cpu()
reco = reco.numpy().reshape([sz[0],sz[1],2])

def img(x):
    return np.sqrt(np.sum(np.abs(x)**2,2))

plt.imshow(img(m))
plt.title('original')
plt.ion()
plt.show()

plt.imshow(img(reco))
plt.title('reconstruction')


# %% easy optimize

g = np.random.rand(NRep,T,2) - 0.5
g = g.ravel()

lbd = 0
use_tanh_grad_moms_cap = 1

args = (m_t,rampX_t,rampY_t,adc_mask_t,sz,NRep,lbd,use_tanh_grad_moms_cap,use_gpu)

g_t = torch.from_numpy(g).float()

if use_gpu:
    g_t = g_t.cuda()

g_t.requires_grad = True

optimizer = optim.LBFGS([g_t], max_iter=1,history_size=200)

def weak_closure():
    optimizer.zero_grad()
    loss,_ = phi_grad_allrep_readout2d_compact_torch(g_t, args)
    
    loss.backward()
    
    return loss

training_iter = 200
for i in range(training_iter):
    optimizer.step(weak_closure)
    
    _,reco = phi_grad_allrep_readout2d_compact_torch(g_t, args)
    
    reco = reco.detach()
    reco = reco.cpu()
    reco = reco.numpy().reshape([sz[0],sz[1],2])
    
    print("error =%f" %e(m.ravel(),reco.ravel()))


_,reco = phi_grad_allrep_readout2d_compact_torch(g_t, args)

reco = reco.detach()
reco = reco.cpu()
reco = reco.numpy().reshape([sz[0],sz[1],2])

plt.imshow(img(reco))
plt.title('reconstruction')
plt.ion()
plt.show()









