#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: aloktyus

variable flipangles assuming perfect spoiling Mxy=0, play with reordering


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
if core.opt_helper.get_cuda_mem_GB() > 3:
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
    
def dbg():
    import pdb; pdb.set_trace()    
    
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

# define setup
sz = np.array([2,2])                                           # image size
NRep = 200                                          # number of repetitions
T = sz[0] + 3                                        # number of events F/R/P
NSpins = 1                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
#dt = 0.0001                         # time interval between actions (seconds)

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]


#############################################################################
## Init spin system and the scanner ::: #####################################


numerical_phantom = np.load('../../data/brainphantom_2D.npy')
numerical_phantom = cv2.resize(numerical_phantom, dsize=(sz[0],sz[0]), interpolation=cv2.INTER_CUBIC)
numerical_phantom[numerical_phantom < 0] = 0
    
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu)
spins.set_system(numerical_phantom)

# uniform PD
spins.PD[:] = 1

scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu)
scanner.get_ramps()
scanner.set_adc_mask()

# allow for relaxation after last readout event
scanner.adc_mask[:scanner.T-scanner.sz[0]-1] = 0
scanner.adc_mask[-1] = 0

scanner.init_coil_sensitivities()

# init tensors
flips = torch.ones((T,NRep), dtype=torch.float32) * 0 * np.pi/180
#flips[0,:] = 90*np.pi/180

T1 = 1 # = spins.T1[0] ---- seconds
TR = 0.05   

spins.T1[:] = 1

E1 = torch.exp(-TR/spins.T1[0])
flips[0,:] = torch.acos(E1)

flips = setdevice(flips)
     
scanner.init_flip_tensor_holder()
scanner.set_flip_tensor(flips)

# gradient-driver precession
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

# FID
#grad_moms[T-sz[0]-1:-1,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
#grad_moms[T-sz[0]-1:-1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])

grad_moms = setdevice(grad_moms)

# event timing vector 
event_time = torch.from_numpy(np.zeros((scanner.T,scanner.NRep,1))).float()
event_time[0,:,0] = 1e-3
event_time[-1,:,0] = TR
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
    
    
spoiler = torch.zeros((spins.NSpins, 1, spins.NVox,4,1)).float()
spoiler[:,:,:,2:,:] = 1                # preserve longitudinal component
spoiler = setdevice(spoiler)


z_comp = np.zeros((NRep,1))
t_comp = np.zeros((NRep,1))

ss_at_ernst = (1-E1)/(1-E1**2)
spins.M[:,:,:,2,:] = (spins.M0[:,:,:,2] * ss_at_ernst).unsqueeze(3)
    
# scanner forward process loop
for r in range(NRep):                                   # for all repetitions

    #spins.M[:,:,:,2,:] = (spins.M0[:,:,:,2] * ss_at_ernst).unsqueeze(3)

    for t in range(T):                                      # for all actions
    
        scanner.flip(t,r,spins)
        
        if t == 0:
            t_comp[r] =  torch.sqrt(torch.sum(spins.M[0,0,0,:2]**2)).detach().cpu()
            z_comp[r] = spins.M[0,0,0,2].detach().cpu()            
              
        #delay = torch.abs(event_time[t,r] + 1e-8)
        delay = torch.abs(event_time[t,r])
        scanner.set_relaxation_tensor(spins,delay)
        scanner.set_freeprecession_tensor(spins,delay)
        scanner.relax_and_dephase(spins)
            
        scanner.grad_precess(t,r,spins)
        scanner.read_signal(t,r,spins)
        
        

        
    spins.M = spins.M * spoiler
        

# init reconstructed image
scanner.init_reco()

#############################################################################
## Inverse pass, reconstruct image with adjoint operator ::: ################
# WARNING: so far adjoint is pure gradient-precession based

for t in range(T-1,-1,-1):
    if scanner.adc_mask[t] > 0:
        scanner.do_grad_adj_reco(t,spins)

    
# try to fit this
#target = scanner.reco.clone()
target = scanner.signal[:,:,:,:2,0].clone()
   
reco = scanner.reco.cpu().numpy().reshape([sz[0],sz[1],2])

if False:                                                       # check sanity
    imshow(magimg(spins.img), 'original')
    imshow(magimg(reco), 'reconstruction')
    
    stop()
    
    
plt.close("all")    
    
    
if False:
    fig_orig = plt.figure()  # create a figure object
    ax = fig_orig.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax.plot(z_comp)
    ax.set_ylabel('z component -- forward sim')
    
    fig_opt = plt.figure()  # create a figure object
    ax_opt = fig_opt.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax_opt.set_ylabel('z component -- optimization')
    
    fig_t = plt.figure()  # create a figure object
    ax_t = fig_t.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax_t.plot(t_comp)
    ax_t.set_ylabel('transverse component -- forward sim')
    
    fig_t = plt.figure()  # create a figure object
    ax_t_opt = fig_t.add_subplot(1, 1, 1)  # create an axes object in the figure
    ax_t_opt.set_ylabel('transversez component -- optimization')

z_comp_sim = z_comp.copy()
t_comp_sim = t_comp.copy()

# %% ###     OPTIMIZE ######################################################@
#############################################################################    
    
total_iter = 660
z_comp_opt = np.zeros((total_iter,NRep,1))
t_comp_opt = np.zeros((total_iter,NRep,1))
flip_angles_comp = np.zeros((total_iter,NRep,1))

    
def phi_FRP_model(opt_params,aux_params):
    
    global z_comp_opt    
    global t_comp_opt
    global flip_angles_comp    
    
    flips,grads,event_time,sigmul = opt_params
    
    scanner.init_signal()
    spins.set_initial_magnetization()
    
    # always flip 90deg on first action (test)
    if False:                                 
        flips_base = torch.ones((1,NRep), dtype=torch.float32) * 90 * np.pi/180
        scanner.custom_flip(0,flips_base,spins)
        scanner.custom_relax(spins,dt=0.06)            # relax till ADC (sec)
        

    # only allow for flip at the beginning of repetition        
    flip_mask = torch.zeros((scanner.T, scanner.NRep)).float()        
    flip_mask[0,:] = 1
    flip_mask = setdevice(flip_mask)
    flips = flips * flip_mask
        
    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor(flips)
    
    # gradients
    grad_moms = torch.cumsum(grads,0)
    
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms)
    
    spoiler = torch.zeros((spins.NSpins, 1, spins.NVox,4,1)).float()
    spoiler[:,:,:,2:,:] = 1                # preserve longitudinal component
    spoiler = setdevice(spoiler)
    
    #spins.M[:,:,:,2,:] = (spins.M0[:,:,:,2] * ss_at_ernst).unsqueeze(3)
          
    for r in range(NRep):                                   # for all repetitions
        #spins.M[:,:,:,2,:] = (spins.M0[:,:,:,2] * ss_at_ernst).unsqueeze(3)    
    
        for t in range(T):
            scanner.flip(t,r,spins)
            
            if t == 0:
                z_comp_opt[opt.globepoch,r] = spins.M[0,0,0,2].detach().cpu()
                t_comp_opt[opt.globepoch,r] = torch.sqrt(torch.sum(spins.M[0,0,0,:2]**2)).detach().cpu()
                
            
            delay = torch.abs(event_time[t,r]) + 1e-6
            scanner.set_relaxation_tensor(spins,delay)
            scanner.set_freeprecession_tensor(spins,delay)
            scanner.relax_and_dephase(spins)
    
            scanner.grad_precess(t,r,spins)
            scanner.read_signal(t,r,spins) 
            
        # destroy transverse component
        spins.M = spins.M * spoiler
        
        #z_comp[r] = spins.M[0,0,0,2].detach().cpu()
        
    flip_angles_est = opt.scanner_opt_params[0].detach().cpu().numpy()*180/np.pi
    flip_angles_comp[opt.globepoch,:,0] = np.round(flip_angles_est[0,:]*100)/100        
        
    scanner.init_reco()
    
    scanner.signal = scanner.signal * sigmul
    
    for t in range(T-1,-1,-1):
        if scanner.adc_mask[t] > 0:
            scanner.do_grad_adj_reco(t,spins)
            
    signal = scanner.signal[:,:,:,:2,0]            
            
            
    loss = (signal - target)
    phi = torch.sum((1.0/NVox)*torch.abs(loss.squeeze())**2)
    
    #ereco = scanner.reco.detach().cpu().numpy().reshape([sz[0],sz[1],2])
    ereco = scanner.signal[:,:,:,:2,0].detach().cpu().numpy()
    error = e(target.cpu().numpy().ravel(),ereco.ravel())     

    #ax_opt.plot(z_comp)
    #ax_t_opt.plot(t_comp)
    
    #plt.pause(0.01)

    
    return (phi,scanner.reco, error)
    

def init_variables():
    g = np.random.rand(T,NRep,2) - 0.5

    grads = torch.from_numpy(g).float()
    
    grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 
    grad_moms = scanner.setdevice(grad_moms)

    # FID    
    #grad_moms[T-sz[0]-1:-1,:,0] = torch.linspace(-int(sz[0]/2),int(sz[0]/2)-1,int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
    #grad_moms[T-sz[0]-1:-1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])


    padder = torch.zeros((1,scanner.NRep,2),dtype=torch.float32)
    padder = scanner.setdevice(padder)
    temp = torch.cat((padder,grad_moms),0)
    grads = temp[1:,:,:] - temp[:-1,:,:]   
    
    grads = setdevice(grads)
    grads.requires_grad = True
    
    flips = torch.ones((T,NRep), dtype=torch.float32) * np.pi* 45 / 180.0
    
    flips = setdevice(flips)
    flips.requires_grad = True
    
    flips = setdevice(flips)
    
    event_time = torch.from_numpy(np.zeros((scanner.T,scanner.NRep,1))).float()

    event_time[0,:,0] = 1e-3
    event_time[-1,:,0] = TR
    
    event_time = setdevice(event_time)
    event_time.requires_grad = True
    
    # global signal scaler
    sigmul = torch.ones((1,1)).float()*1.0
    sigmul = setdevice(sigmul)
    sigmul.requires_grad = True 
    
    return [flips, grads, event_time, sigmul]
    
   
# %% # OPTIMIZATION land
    
opt = core.opt_helper.OPT_helper(scanner,spins,None,1)

opt.use_periodic_grad_moms_cap = 1           # do not sample above Nyquist flag
opt.learning_rate = 0.01                                        # ADAM step size

print('<seq> now')
opt.opti_mode = 'seq'

opt.set_opt_param_idx([0])
opt.custom_learning_rate = [0.05]

opt.set_handles(init_variables, phi_FRP_model)


opt.scanner_opt_params = opt.init_variables()

opt.custom_learning_rate = [0.05]
opt.train_model(training_iter=15, show_par=False)

opt.custom_learning_rate = [0.05]
opt.train_model(training_iter=50, show_par=False)


#target_numpy = target.cpu().numpy().reshape([sz[0],sz[1],2])
#
#_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)
#reco = reco.detach().cpu().numpy().reshape([sz[0],sz[1],2])
#
#imshow(magimg(target_numpy), 'target')
#imshow(magimg(reco), 'reconstruction')

flip_angles = opt.scanner_opt_params[0].detach().cpu().numpy()*180/np.pi
flip_angles = np.round(flip_angles[0,:])

print(flip_angles)

plt.plot(z_comp)

stop()



# %% # PLOT land

z_comp_opt = z_comp_opt[1:,:,0]
t_comp_opt = t_comp_opt[1:,:,0]
flip_angles_comp = flip_angles_comp[1:,:,0]

#z_comp_sim = z_comp.copy()
#t_comp_sim = t_comp.copy()

fig_orig = plt.figure()  # create a figure object
ax = fig_orig.add_subplot(1, 1, 1)  # create an axes object in the figure
ax.plot(z_comp_sim)
ax.set_ylabel('z component -- forward sim')

fig_orig = plt.figure()  # create a figure object
ax = fig_orig.add_subplot(1, 1, 1)  # create an axes object in the figure
ax.plot(t_comp_sim)
ax.set_ylabel('t component -- forward sim')

fig_orig = plt.figure()  # create a figure object
ax = fig_orig.add_subplot(1, 1, 1)  # create an axes object in the figure
ax.set_ylabel('z component -- optimization evolution')

for i in range(z_comp_opt.shape[0]):
    ax.plot(z_comp_opt[i,:])
    
    #plt.pause(0.5)

fig_orig = plt.figure()  # create a figure object
ax = fig_orig.add_subplot(1, 1, 1)  # create an axes object in the figure
ax.set_ylabel('z component -- optimization evolution')

for i in range(z_comp_opt.shape[0]):
    ax.plot(z_comp_opt[i,:])
    
fig_orig = plt.figure()  # create a figure object
ax = fig_orig.add_subplot(1, 1, 1)  # create an axes object in the figure
ax.set_ylabel('t component -- optimization evolution')

for i in range(t_comp_opt.shape[0]):
    ax.plot(t_comp_opt[i,:])
    
fig_orig = plt.figure()  # create a figure object
ax = fig_orig.add_subplot(1, 1, 1)  # create an axes object in the figure
ax.set_ylabel('flip angles -- optimization evolution')

for i in range(flip_angles_comp.shape[0]):
    ax.plot(flip_angles_comp[i,:])
    
    
np.save('longitudinal_comp_ernst_sim.npy', z_comp_sim)
np.save('transverse_comp_ernst_sim.npy', t_comp_sim)

np.save('longitudinal_comp_optimization.npy', z_comp_opt)
np.save('transverse_comp_optimization.npy', t_comp_opt)
np.save('flip_angle_changes_optimization.npy', flip_angles_comp)

import scipy.io as sio
# Create a dictionary
adict = {}
adict['longitudinal_comp_ernst_sim'] = z_comp_sim
adict['transverse_comp_ernst_sim'] = t_comp_sim
adict['longitudinal_comp_optimization'] = z_comp_opt
adict['transverse_comp_optimization'] = t_comp_opt
adict['flip_angle_changes_optimization'] = flip_angles_comp

sio.savemat('ernst_sim.mat', adict)










