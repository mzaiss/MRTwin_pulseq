#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: aloktyus
"""

import os, sys
import numpy as np
import torch


# forward Fourier transform
def fftfull(x):                                                 # forward FFT
    return np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(x)))

def ifftfull(x):                                                # inverse FFT
    return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(x)))

# NRMSE error function
def e(gt,x):
    return 100*np.linalg.norm((gt-x).ravel())/np.linalg.norm(gt.ravel())

def phi_grad_allrep_readout2d(g,args):
    
    m,rampX,rampY,adc_mask,sz,NRep,lbd,use_tanh_grad_moms_cap = args
    
    T = adc_mask.size

    g = np.reshape(g,[NRep,T,2])
    N = np.prod(sz)             # Fourier transform normalization coefficient
    
    # L2 norm for regularization
    def l2(z):
        return z*np.conj(z)
    
    def dl2(z):
        return 2*z
    
    def dtanh(z):
        return 1 - np.tanh(z)**2
    
    # vectorize ramps and the image
    rampX = np.expand_dims(rampX.ravel(),0)
    rampY = np.expand_dims(rampY.ravel(),0)
    m = np.expand_dims(m.ravel(),1)
    
    phi = 0
    tdg = np.zeros((NRep,2*T))
    
    prediction = 0
    
    grads = np.zeros((NRep,T,2))
    E_all_rep = np.zeros((NRep,T,N), dtype = np.complex128)
    
    for rep in range(NRep):
    
      # integrate over time to get grad_moms from the gradients
      grad_moms = np.cumsum(g[rep,:,:].squeeze(),0)
    
      # prewinder (to be relaxed in future)
      #g[:,0] = g[:,0] - T/2 - 1
    
      if use_tanh_grad_moms_cap:
        fmax = sz / 2                     # cap the grad_moms to [-1..1]*sz/2
    
        for i in [0,1]:
          grad_moms[:,i] = fmax[i]*np.tanh(grad_moms[:,i])      # soft threshold
          grad_moms[np.abs(grad_moms[:,i]) > fmax[i],i] = np.sign(grad_moms[np.abs(grad_moms[:,i]) > fmax[i],i])*fmax[i]  # hard threshold, this part is nondifferentiable
        
      grads[rep,:,:] = np.diff(np.concatenate((np.expand_dims(np.array([0,0]),0),grad_moms),0),1,0)            # actual gradient forms are the derivative of grad_moms (inverse cumsum)
    
      # compute the B0 by adding gradients in X/Y after multiplying them respective ramps
      B0X = np.expand_dims(grad_moms[:,0],1) * rampX 
      B0Y = np.expand_dims(grad_moms[:,1],1) * rampY
    
      B0 = B0X + B0Y
    
      # encoding operator (ignore relaxation)
      E = np.exp(1j*B0)
      
      E = E * adc_mask
      
      E_all_rep[rep,:,:] = E
      
      # compute loss
      E = np.matrix(E)
      prediction = prediction + np.array(E.H*(E*m)) / N
      
      phi = phi + lbd*np.sum(l2(grad_moms.ravel()))
      
    
    loss = (prediction - m)
    phi = phi + np.sum(l2(loss.ravel()))
    
    cmx = np.matrix(np.conj(dl2(loss)) * m.T / N)
    
    for rep in range(NRep):
      
      # integrate over time to get grad_moms from the gradients
      grad_moms = np.cumsum(g[rep,:,:].squeeze(),0)
    
      save_grad_moms = grad_moms.copy()          # needed for tanh derivative
    
      if use_tanh_grad_moms_cap:
        fmax = sz / 2                     # cap the grad_moms to [-1..1]*sz/2
    
        for i in [0,1]:
          grad_moms[:,i] = fmax[i]*np.tanh(grad_moms[:,i])      # soft threshold
          grad_moms[np.abs(grad_moms[:,i]) > fmax[i],i] = np.sign(grad_moms[np.abs(grad_moms[:,i]) > fmax[i],i])*fmax[i]  # hard threshold, this part is nondifferentiable
      
    
      # compute the B0 by adding gradients in X/Y after multiplying them respective ramps
      B0X = np.expand_dims(grad_moms[:,0],1) * rampX 
      B0Y = np.expand_dims(grad_moms[:,1],1) * rampY
    
      B0 = B0X + B0Y
    
      # encoding operator (ignore relaxation)
      E = np.exp(1j*B0)
      
      E = E * adc_mask
      E = np.matrix(E)
    
      # compute derivative with respect to temporal gradient waveforms
      dgXY = np.multiply((np.conj(E) * cmx + np.conj(E * cmx.T)),E)
      dgXY = np.array(dgXY)
      
      dg = np.zeros(grad_moms.shape, dtype = np.complex128)
      
      dg[:,0] = np.sum(1j*dgXY*rampX,1)
      dg[:,1] = np.sum(1j*dgXY*rampY,1)
      
      if use_tanh_grad_moms_cap:
        for i in [0,1]:
          dg[:,i] = fmax[i]*dtanh(save_grad_moms[:,i]) * dg[:,i]
        
      
      dg = np.cumsum(dg[::-1,:],0)[::-1,:]
      dg = np.real(dg)
    
      # regularization part derivatives
      rega = dl2(grad_moms)
      if use_tanh_grad_moms_cap:
        for i in [0,1]:
          rega[:,i] = fmax[i]*dtanh(save_grad_moms[:,i]) * rega[:,i]
          
      rega = lbd*np.cumsum(rega[::-1,:],0)[::-1,:]
    
      tdg[rep,:] = (dg + rega).ravel()
    
    tdg = tdg.ravel()
             
    return (phi,tdg)
    #return (phi,tdg,prediction,E_all_rep,grad_moms,grads)


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

# pack the parameters for the gradient function
args = (m,rampX,rampY,adc_mask,sz,NRep,lbd,use_tanh_grad_moms_cap)
(phi,dg_ana) = phi_grad_allrep_readout2d(g, args)                            # compute loss and analytical derivatives

print("python phi =%f" %np.real(phi))

# compute numerical derivatives
h = 1e-4
dg = np.zeros(g.shape)
dphi_h = np.zeros(g.shape)
for i in range(g.size):
  dg[i] = 1 
  (cphi, _) = phi_grad_allrep_readout2d(g+h*dg,args)
  cphi -= phi
  dphi_h[i] = cphi
  dg[i] = 0
  
  

dx_num = dphi_h/h 
dphi_h = None
dx = None

print("deriv-err=%1.3f " %(e(dx_num,dg_ana)))

## Torch land

def phi_grad_allrep_readout2d_torch(g,args):
    
    m,rampX,rampY,adc_mask,sz,NRep,lbd,use_tanh_grad_moms_cap = args
    
    T = adc_mask.size()[0]

    g = torch.reshape(g,[NRep,T,2])
    N = np.prod(sz)             # Fourier transform normalization coefficient
    
    phi = 0
    
    E = torch.zeros((T,2,N,2), dtype = torch.float64)
    
    prediction = torch.zeros((N,2), dtype = torch.float64)
    
    grads = torch.zeros((NRep,T,2), dtype = torch.float64)
    E_all_rep = torch.zeros((NRep,T,N,2), dtype = torch.float64)
    
    for rep in range(NRep):
    
      # integrate over time to get grad_moms from the gradients
      grad_moms = torch.cumsum(g[rep,:,:].squeeze(),0)
    
      # prewinder (to be relaxed in future)
      #g[:,0] = g[:,0] - T/2 - 1
    
      if use_tanh_grad_moms_cap:
        fmax = sz / 2                     # cap the grad_moms to [-1..1]*sz/2
    
        for i in [0,1]:
          grad_moms[:,i] = fmax[i]*torch.tanh(grad_moms[:,i])      # soft threshold
          grad_moms[torch.abs(grad_moms[:,i]) > fmax[i],i] = torch.sign(grad_moms[torch.abs(grad_moms[:,i]) > fmax[i],i])*fmax[i]  # hard threshold, this part is nondifferentiable
        
        
      temp = torch.cat((torch.unsqueeze(torch.zeros((2),dtype=torch.float64),0),grad_moms),0)
      grads[rep,:,:] = temp[1:,:] - temp[:-1,:]                              # actual gradient forms are the derivative of grad_moms (inverse cumsum)
    
      # compute the B0 by adding gradients in X/Y after multiplying them respective ramps
      B0X = torch.unsqueeze(grad_moms[:,0],1) * rampX 
      B0Y = torch.unsqueeze(grad_moms[:,1],1) * rampY
    
      B0 = B0X + B0Y
      
      E_cos = torch.cos(B0)
      E_sin = torch.sin(B0)
      
      # encoding operator (ignore relaxation)
      E[:,0,:,0] = E_cos
      E[:,0,:,1] = -E_sin
      E[:,1,:,0] = E_sin
      E[:,1,:,1] = E_cos
    
      E = E * torch.unsqueeze(torch.unsqueeze(adc_mask,2),3)
      
      fwd = torch.matmul(E.view(T,2,N*2),m.view(N*2))
      inv = torch.matmul(E.permute([2,3,0,1]).view(N,2,T*2),fwd.view(T*2,1)).squeeze()

      prediction = prediction + inv / N
      #prediction = prediction + np.array(E.H*(E*m)) / N
      
    loss = (prediction - m)
    phi = phi + torch.sum(torch.abs(loss.squeeze())**2)
    
    return (phi)

def phi_grad_allrep_readout2d_compact_torch(g,args):
    
    m,rampX,rampY,adc_mask,sz,NRep,lbd,use_tanh_grad_moms_cap = args
    
    T = adc_mask.size()[1]
    N = np.prod(sz)             # Fourier transform normalization coefficient
    g = torch.reshape(g,[NRep,T,2])
    E = torch.zeros((NRep,T,2,N,2), dtype = torch.float64)
    
    phi = 0
    
    grad_moms = torch.cumsum(g,1)
    
    if use_tanh_grad_moms_cap:
      fmax = sz / 2
      for i in [0,1]:
          grad_moms[:,:,i] = fmax[i]*torch.tanh(grad_moms[:,:,i])      # soft threshold
          for rep in range(NRep):
              grad_moms[rep,torch.abs(grad_moms[rep,:,i]) > fmax[i],i] = torch.sign(grad_moms[rep,torch.abs(grad_moms[rep,:,i]) > fmax[i],i])*fmax[i]  # hard threshold, this part is nondifferentiable        
              
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
    
    loss = (inv - m)
    phi = torch.sum(torch.abs(loss.squeeze())**2)
    
    return (phi)

g_t = torch.from_numpy(g)
g_t.requires_grad = True
m_t = torch.from_numpy(np.stack((np.real(m), np.imag(m)),2))
rampX_t = torch.from_numpy(rampX)
rampY_t = torch.from_numpy(rampY)
adc_mask_t = torch.from_numpy(adc_mask)

# vectorize ramps and the image
rampX_t = torch.unsqueeze(torch.unsqueeze(rampX_t.flatten(),0),0)
rampY_t = torch.unsqueeze(torch.unsqueeze(rampY_t.flatten(),0),0)
m_t = m_t.view([np.prod(sz),2])
adc_mask_t = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(adc_mask_t,0),2),3)

args = (m_t,rampX_t,rampY_t,adc_mask_t,sz,NRep,lbd,use_tanh_grad_moms_cap)
(phi) = phi_grad_allrep_readout2d_compact_torch(g_t, args)

phi.backward()

dg_ana = (g_t.grad).numpy()

print("deriv-err=%1.3f " %(e(dx_num,dg_ana)))

print("torch phi =%f" %phi)









