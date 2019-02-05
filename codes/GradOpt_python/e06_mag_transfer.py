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

# WHAT we measure
class SpinSystem():
    
    def __init__(self,sz,NVox,NSpins):
        
        self.sz = sz                                             # image size
        self.NVox = sz[0]*sz[1]                                 # voxel count
        self.NSpins = NSpins              # number of spin sims in each voxel
        
        self.PD = None                        # proton density tensor (NVox,)
        self.T1 = None                          # T1 relaxation times (NVox,)
        self.T2 = None                          # T2 relaxation times (NVox,)
        self.dB0 = None                        # spin off-resonance (NSpins,)
        
        self.M0 = None          # initial magnetization state (NSpins,NVox,4)
        self.M = None            # curent magnetization state (NSpins,NVox,4)
        
        # aux
        self.img = None
        self.R2 = None
    
    def set_system(self):
        
        # load image
        m = np.load('../../data/phantom.npy')
        m = cv2.resize(m, dsize=(self.sz[0], self.sz[1]), interpolation=cv2.INTER_CUBIC)
        m = m / np.max(m)
        self.img = m.copy()
        
        # set relaxations (unit - seconds) and proton density
        PD = torch.from_numpy(magimg(m).reshape([self.NVox])).float()
        T1 = torch.ones(self.NVox, dtype=torch.float32)*4
        T2 = torch.ones(self.NVox, dtype=torch.float32)*2
        T2[0:NVox/2] = 0.09
        
        # set NSpins offresonance (from R2)
        factor = (0*1e0*np.pi/180) / self.NSpins
        dB0 = torch.from_numpy(factor*np.arange(0,self.NSpins).reshape([self.NSpins])).float()
        
        self.T1 = setdevice(T1)
        self.T2 = setdevice(T2)
        self.PD = setdevice(PD)
        self.dB0 = setdevice(dB0)
        
    def set_initial_magnetization(self):
        M0 = torch.zeros((self.NSpins,self.NVox,4), dtype=torch.float32)
        
        M0 = setdevice(M0)
        
        # set initial longitudinal magnetization value
        M0[:,:,2:] = 1
        M0[:,:,2:] = M0[:,:,2:] * self.PD.view([self.NVox,1])    # weight by proton density
        
        M = M0.clone().view([self.NSpins,self.NVox,4,1])
        
        self.M0 = M0
        
        self.M = setdevice(M)
        

# HOW we measure
class Scanner():
    
    def __init__(self,sz,NVox,NSpins,NRep,T,NCoils,noise_std):
        
        self.sz = sz                                             # image size
        self.NVox = sz[0]*sz[1]                                 # voxel count
        self.NSpins = NSpins              # number of spin sims in each voxel
        self.NRep = NRep                              # number of repetitions
        self.T = T                       # number of "actions" with a readout
        self.NCoils = NCoils                # number of receive coil elements
        self.noise_std = noise_std              # additive Gaussian noise std
        
        self.adc_mask = None         # ADC signal acquisition event mask (T,)
        self.rampX = None        # spatial encoding linear gradient ramp (sz)
        self.rampY = None
        self.F = None                              # flip tensor (T,NRep,4,4)
        self.R = None                          # relaxation tensor (NVox,4,4)
        self.P = None                   # free precession tensor (NSpins,4,4)
        self.G = None            # gradient precession tensor (NRep,NVox,4,4)
        self.G_adj = None         # adjoint gradient operator (NRep,NVox,4,4)

        self.B0_grad_cos = None  # accum phase due to gradients (T,NRep,NVox)
        self.B0_grad_sin = None
        self.B0_grad_adj_cos = None  # adjoint grad phase accum (T,NRep,NVox)
        self.B0_grad_adj_sin = None
        
        self.B1 = None          # coil sensitivity profiles (NCoils,NVox,2,2)

        self.signal = None                # measured signal (NCoils,T,NRep,4)
        self.reco =  None                       # reconstructed image (NVox,) 
        
    def set_adc_mask(self):
        adc_mask = torch.from_numpy(np.ones((self.T,1))).float()
        adc_mask[:T-self.sz[0]] = 0
        adc_mask[-1] = 0
        
        self.adc_mask = setdevice(adc_mask)

    def get_ramps(self):
        
        use_nonlinear_grads = False                       # very experimental
        
        baserampX = np.linspace(-1,1,self.sz[0] + 1)
        baserampY = np.linspace(-1,1,self.sz[1] + 1)
        
        if use_nonlinear_grads:
            baserampX = np.abs(baserampX)**1.2 * np.sign(baserampX)
            baserampY = np.abs(baserampY)**1.2 * np.sign(baserampY)
        
        rampX = np.pi*baserampX
        rampX = -np.expand_dims(rampX[:-1],1)
        rampX = np.tile(rampX, (1, self.sz[1]))
        
        rampX = torch.from_numpy(rampX).float()
        rampX = rampX.view([1,1,self.NVox])    
        
        # set gradient spatial forms
        rampY = np.pi*baserampY
        rampY = -np.expand_dims(rampY[:-1],0)
        rampY = np.tile(rampY, (self.sz[0], 1))
        
        rampY = torch.from_numpy(rampY).float()
        rampY = rampY.view([1,1,self.NVox])    
        
        self.rampX = setdevice(rampX)
        self.rampY = setdevice(rampY)
        
    def init_coil_sensitivities(self):
        # handle complex mul as matrix mul
        B1 = torch.zeros((self.NCoils,self.NVox,2,2), dtype=torch.float32)
        B1[:,:,0,0] = 1
        B1[:,:,1,1] = 1
        
        self.B1 = setdevice(B1)
        
    def init_flip_tensor_holder(self):
        F = torch.zeros((self.T,self.NRep,1,4,4), dtype=torch.float32)
        
        F[:,:,0,3,3] = 1
        F[:,:,0,1,1] = 1
         
        self.F = setdevice(F)
         
    def set_flip_tensor(self,flips):
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        self.F[:,:,0,0,0] = flips_cos
        self.F[:,:,0,0,2] = flips_sin
        self.F[:,:,0,2,0] = -flips_sin
        self.F[:,:,0,2,2] = flips_cos 
         
    def set_relaxation_tensor(self,spins,dt):
        R = torch.zeros((self.NVox,4,4), dtype=torch.float32) 
        
        R = setdevice(R)
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,3,3] = 1
        
        R[:,0,0] = T2_r
        R[:,1,1] = T2_r
        R[:,2,2] = T1_r
        R[:,2,3] = 1 - T1_r
         
        R = R.view([1,self.NVox,4,4])
        
        self.R = R
        
    def set_freeprecession_tensor(self,spins,dt):
        P = torch.zeros((self.NSpins,1,4,4), dtype=torch.float32)
        
        P = setdevice(P)
        
        B0_nspins = spins.dB0.view([self.NSpins])
        
        B0_nspins_cos = torch.cos(B0_nspins*dt)
        B0_nspins_sin = torch.sin(B0_nspins*dt)
         
        P[:,0,0,0] = B0_nspins_cos
        P[:,0,0,1] = -B0_nspins_sin
        P[:,0,1,0] = B0_nspins_sin
        P[:,0,1,1] = B0_nspins_cos
         
        P[:,:,2,2] = 1
        P[:,:,3,3] = 1         
         
        self.P = P
         
    
    def init_gradient_tensor_holder(self):
        G = torch.zeros((self.NRep,self.NVox,4,4), dtype=torch.float32)
        G[:,:,2,2] = 1
        G[:,:,3,3] = 1
         
        G_adj = torch.zeros((self.NRep,self.NVox,4,4), dtype=torch.float32)
        G_adj[:,:,2,2] = 1
        G_adj[:,:,3,3] = 1
         
        self.G = setdevice(G)
        self.G_adj = setdevice(G_adj)
        
    def set_grad_op(self,t):
        
        self.G[:,:,0,0] = self.B0_grad_cos[t,:,:]
        self.G[:,:,0,1] = -self.B0_grad_sin[t,:,:]
        self.G[:,:,1,0] = self.B0_grad_sin[t,:,:]
        self.G[:,:,1,1] = self.B0_grad_cos[t,:,:]
        
    def set_grad_adj_op(self,t):
        
        self.G_adj[:,:,0,0] = self.B0_grad_adj_cos[t,:,:]
        self.G_adj[:,:,0,1] = self.B0_grad_adj_sin[t,:,:]
        self.G_adj[:,:,1,0] = -self.B0_grad_adj_sin[t,:,:]
        self.G_adj[:,:,1,1] = self.B0_grad_adj_cos[t,:,:]        
        
    def set_gradient_precession_tensor(self,grad_moms):
        
        padder = torch.zeros((1,NRep,2),dtype=torch.float32)
        padder = setdevice(padder)
        temp = torch.cat((padder,grad_moms),0)
        grads = temp[1:,:,:] - temp[:-1,:,:]        
        
        B0X = torch.unsqueeze(grads[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grads[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_cos = torch.cos(B0_grad)
        self.B0_grad_sin = torch.sin(B0_grad)
        
        # for backward pass
        B0X = torch.unsqueeze(grad_moms[:,:,0],2) * self.rampX
        B0Y = torch.unsqueeze(grad_moms[:,:,1],2) * self.rampY
        
        B0_grad = (B0X + B0Y).view([self.T,self.NRep,self.NVox])
        
        self.B0_grad_adj_cos = torch.cos(B0_grad)
        self.B0_grad_adj_sin = torch.sin(B0_grad)        
        
    def flip(self,t,r,spins):
        
        spins.M = torch.matmul(self.F[t,r,:,:,:],spins.M)
        
    def relax(self,spins):
        spins.M = torch.matmul(self.R,spins.M)
        
    def relax_and_dephase(self,spins):
        
        spins.M = torch.matmul(self.R,spins.M)
        spins.M = torch.matmul(self.P,spins.M)
        
    def grad_precess(self,r,spins):
        
        spins.M = torch.matmul(self.G[r,:,:,:],spins.M)
        
    def init_signal(self):
        signal = torch.zeros((self.NCoils,self.T,self.NRep,4,1), dtype=torch.float32) 
        signal[:,:,:,2:,0] = 1                                 # aux dim zero
              
        self.signal = setdevice(signal)
        
    def init_reco(self):
        reco = torch.zeros((self.NVox,2), dtype = torch.float32)
        
        self.reco = setdevice(reco)
        
    def read_signal(self,t,r,spins):
        
        if self.adc_mask[t] > 0:
            sig = torch.sum(spins.M[:,:,:2,0],[0])
            sig = torch.matmul(self.B1,sig.unsqueeze(0).unsqueeze(3))
            
            self.signal[:,t,r,:2] = (torch.sum(sig,[1]) * self.adc_mask[t])
            
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.signal[:,t,r,:,0].shape).float()
                noise[:,2:] = 0
                noise = setdevice(noise)
                self.signal[:,t,r,:,0] = self.signal[:,t,r,:,0] + noise

    # reconstruct image readout by readout            
    def do_grad_adj_reco(self,t,spins):
        
        s = self.signal[:,t,:,:,:] * self.adc_mask[t]
        # for now we ignore parallel imaging options here (do naive sum sig over coil)
        s = torch.sum(s, 0)                                                  
        r = torch.matmul(self.G_adj.permute([1,0,2,3]), s)
        self.reco = self.reco + torch.sum(r[:,:,:2,0],1)
        
    ## extra func land        
    # aux flexible operators for sandboxing things
    def custom_flip(self,t,spins,flips):
        
        F = torch.zeros((self.T,self.NRep,1,4,4), dtype=torch.float32)
        
        F[:,:,0,3,3] = 1
        F[:,:,0,1,1] = 1
         
        F = setdevice(F)
        
        flips = setdevice(flips)
        
        flips_cos = torch.cos(flips)
        flips_sin = torch.sin(flips)
        
        F[:,:,0,0,0] = flips_cos
        F[:,:,0,0,2] = flips_sin
        F[:,:,0,2,0] = -flips_sin
        F[:,:,0,2,2] = flips_cos         
        
        spins.M = torch.matmul(F[t,:,:,:],spins.M)
        
    def custom_relax(self,spins,dt=None):
        
        R = torch.zeros((self.NVox,4,4), dtype=torch.float32) 
        
        T2_r = torch.exp(-dt/spins.T2)
        T1_r = torch.exp(-dt/spins.T1)
        
        R[:,3,3] = 1
        
        R[:,0,0] = T2_r
        R[:,1,1] = T2_r
        R[:,2,2] = T1_r
        R[:,2,3] = 1 - T1_r
         
        R = R.view([1,self.NVox,4,4])
        
        R = setdevice(R)
        
        spins.M = torch.matmul(R,spins.M)        
    
#############################################################################
## Init spin system and the scanner ::: #####################################

    
# initialize scanned object
spins = SpinSystem(sz,NVox,NSpins)
spins.set_system()

scanner = Scanner(sz,NVox,NSpins,NRep,T,NCoils,noise_std)
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
event_time = torch.from_numpy(1e-2*np.zeros((scanner.T,scanner.NRep,1))).float()
event_time[0,:,0] = 1e-1
event_time[-1,:,0] = 1e2
event_time = setdevice(event_time)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms)

#############################################################################
## Forward process ::: ######################################################
    
scanner.init_signal()
spins.set_initial_magnetization()

# always flip 90deg on first action (test)
if False:                                 
    flips_base = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
    scanner.custom_flip(0,flips_base,spins)
    scanner.custom_relax(spins,dt=0.06)                # relax till ADC (sec)
    
# scanner forward process loop
for r in range(NRep):                                   # for all repetitions
    for t in range(T):                                      # for all actions
    
        # flip/relax/dephase only if adc is closed
        if scanner.adc_mask[t] == 0:
            scanner.flip(t,r,spins)
                  
            delay = torch.abs(event_time[t,r] + 1e-6)
            scanner.set_relaxation_tensor(spins,delay)
            scanner.set_freeprecession_tensor(spins,delay)
            scanner.relax_and_dephase(spins)
            
        scanner.set_grad_op(t)
        scanner.grad_precess(r,spins)
        scanner.read_signal(t,r,spins)
        

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
   
reco = scanner.reco.cpu().numpy().reshape([sz[0],sz[1],2])

if False:                                                       # check sanity
    plt.imshow(magimg(spins.img))
    plt.title('original')
    plt.ion()
    plt.show()
    
    plt.imshow(magimg(reco))
    plt.title('reconstruction')
    plt.ion()
    plt.show()
    
    gfdgfdfd


# %% ###     OPTIMIZE ######################################################@
#############################################################################

def phi_FRP_model(flips,grads,event_time,args):
    
    scanner,spins,target,use_tanh_grad_moms_cap = args
    
    scanner.init_signal()
    spins.set_initial_magnetization()
    
    # always flip 90deg on first action (test)
    if False:                                 
        flips_base = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
        scanner.custom_flip(0,flips_base,spins)
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
          
    for r in range(NRep):                                   # for all repetitions
        for t in range(T):
            
            if scanner.adc_mask[t] == 0:
                scanner.flip(t,r,spins)
                delay = torch.abs(event_time[t,r] + 1e-6)
                scanner.set_relaxation_tensor(spins,delay)
                scanner.set_freeprecession_tensor(spins,delay)
                scanner.relax_and_dephase(spins)
    
            scanner.set_grad_op(t)
            scanner.grad_precess(r,spins)
            scanner.read_signal(t,r,spins)        
        
    scanner.init_reco()
    
    for t in range(T-1,-1,-1):
        if scanner.adc_mask[t] > 0:
            scanner.set_grad_adj_op(t)
            scanner.do_grad_adj_reco(t,spins)
            
    loss = (scanner.reco - target)
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
    
    event_time = torch.from_numpy(np.zeros((scanner.T,scanner.NRep,1))).float()
    event_time = setdevice(event_time)
    event_time.requires_grad = True
    
    return flips, grads, event_time
    

target = target.detach()
target_numpy = target.cpu().numpy().reshape([sz[0],sz[1],2])

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
        loss.backward()
        
        return loss
    
    for outer_iter in range(nmb_outer_iter):
        
        if doRestart:
            flips,grads,event_time = init_variables()
    
        optimizer = optim.LBFGS([flips,grads,event_time], lr=learning_rate, max_iter=1,history_size=200)
        
        for inner_iter in range(nmb_inner_iter):
            optimizer.step(weak_closure)
            
            _,reco = phi_FRP_model(flips, grads, event_time, args)
            reco = reco.detach().cpu().numpy().reshape([sz[0],sz[1],2])
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
        return reco
    

use_tanh_grad_moms_cap = 1                 # do not sample above Nyquist flag
learning_rate = 0.1                                         # LBFGS step size
training_iter = 2000
nmb_rnd_restart = 15
training_iter_restarts = 10

args = (scanner,spins,target,use_tanh_grad_moms_cap)

best_vars = train_model(True)
    
reco = train_model(False,best_vars)

plt.imshow(magimg(spins.img))
plt.title('original')
plt.ion()
plt.show()

plt.imshow(magimg(reco))
plt.title('reconstruction')











