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
import scipy.io
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
    #importlib.reload(core.spins)
    #importlib.reload(core.scanner)
    #importlib.reload(core.opt_helper)    

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

def magimg_torch(x):
  return torch.sqrt(torch.sum(torch.abs(x)**2,1))

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
sz = np.array([24,24])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 3                                        # number of events F/R/P
NSpins = 64                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
#dt = 0.0001                         # time interval between actions (seconds)

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]


#############################################################################
## Init spin system and the scanner ::: #####################################

# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu)

numerical_phantom = np.load('../../data/brainphantom_2D.npy')
numerical_phantom = cv2.resize(numerical_phantom, dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
#numerical_phantom = cv2.resize(numerical_phantom, dsize=(sz[0], sz[1]), interpolation=cv2.INTER_CUBIC)
numerical_phantom[numerical_phantom < 0] = 0
numerical_phantom=np.swapaxes(numerical_phantom,0,1)

# inhomogeneity (in Hz)
B0inhomo = np.zeros((sz[0],sz[1],1)).astype(np.float32)
B0inhomo = (np.random.rand(sz[0],sz[1]) - 0.5)
B0inhomo = ndimage.filters.gaussian_filter(B0inhomo,1)        # smooth it a bit
B0inhomo = B0inhomo*1500 / np.max(B0inhomo)
B0inhomo = np.expand_dims(B0inhomo,2)
numerical_phantom = np.concatenate((numerical_phantom,B0inhomo),2)

spins.set_system(numerical_phantom)

cutoff = 1e-12
spins.T1[spins.T1<cutoff] = cutoff
spins.T2[spins.T2<cutoff] = cutoff
# end initialize scanned object
spins.T1*=1
spins.T2*=1
imshow(numerical_phantom[:,:,0], title="PD")

#begin nspins with R*
R2 = 30.0
omega = np.linspace(0+1e-5,1-1e-5,NSpins) - 0.5
omega = np.expand_dims(omega[:],1).repeat(NVox, axis=1)

#omega = np.random.rand(NSpins,NVox) - 0.5
omega*=0.9
#omega = np.expand_dims(omega[:,0],1).repeat(NVox, axis=1)

omega = R2 * np.tan ( np.pi  * omega)

#omega = np.random.rand(NSpins,NVox) * 100
if NSpins==1:
    omega[:,:]=0

spins.omega = torch.from_numpy(omega.reshape([NSpins,NVox])).float()
#spins.omega[torch.abs(spins.omega) > 1e3] = 0
spins.omega = setdevice(spins.omega)
#end nspins with R*


scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu)
scanner.get_ramps()
scanner.set_adc_mask()

# allow for relaxation after last readout event
scanner.adc_mask[:scanner.T-scanner.sz[0]-1] = 0
#scanner.adc_mask[:3] = 0
scanner.adc_mask[-1] = 0

scanner.init_coil_sensitivities()

flips = torch.zeros((T,NRep,2), dtype=torch.float32)
flips[0,:,0] = 90*np.pi/180  # GRE preparation part 1 : 90 degree excitation 

flips = setdevice(flips)

scanner.init_flip_tensor_holder()
scanner.set_flipXY_tensor(flips)


# gradient-driver precession
# Cartesian encoding
if False:
    grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 
    
    # xgradmom
    grad_moms[T-sz[0]-1:-1,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
    # ygradmom
    if NRep == 1:
        grad_moms[T-sz[0]-1:-1,:,1] = torch.zeros((1,1)).repeat([sz[0],1])
    else:
        grad_moms[1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))
        grad_moms[-1,:,1] = -torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))
            
    grad_moms[1,0,0] = -torch.ones((1,1))*sz[0]/2  # RARE: rewinder after 90 degree half length, half gradmom


grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

# Cartesian encoding
grad_moms[1,:,0] = -sz[0]/2
grad_moms[T-sz[0]-1:-1,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep])
#grad_moms[T-sz[0]-1:-1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep)).repeat([sz[0],1])
grad_moms[1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))
    
#imshow(grad_moms[T-sz[0]-1:-1,:,0].cpu())
#imshow(grad_moms[T-sz[0]-1:-1,:,1].cpu())

grad_moms = setdevice(grad_moms)

# event timing vector 
event_time = torch.from_numpy(0.2*1e-3*np.ones((scanner.T,scanner.NRep,1))).float()
event_time[:2,:,0] = 1e-2  
event_time[-1,:,0] = 1e2
event_time = setdevice(event_time)

scanner.init_gradient_tensor_holder()

#scanner.set_gradient_precession_tensor_adjhistory(grad_moms)
scanner.set_gradient_precession_tensor(grad_moms,refocusing=False,wrap_k=False)

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

if True: # check sanity: is target what you expect and is sequence what you expect
    targetSeq.print_status(True, reco=None)
    
    #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:4]), label='x')
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(tonumpy(scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(scanner.T+1)*scanner.NRep]) )
        plt.title("ROI_def %d" % scanner.ROI_def)
        fig = plt.gcf()
        fig.set_size_inches(16, 3)
    plt.show()
    
    stop()
    
# %% # export to matlab
experiment_id='GRE90_target'
scanner_dict = dict()
scanner_dict['adc_mask'] = scanner.adc_mask.detach().cpu().numpy()
scanner_dict['B1'] = scanner.B1.detach().cpu().numpy()
scanner_dict['flips'] = flips.detach().cpu().numpy()
scanner_dict['grad_moms'] = grad_moms.detach().cpu().numpy()
scanner_dict['event_times'] = event_time.detach().cpu().numpy()
#scanner_dict['reco'] = tonumpy(reco).reshape([sz[0],sz[1],2])
scanner_dict['ROI'] = tonumpy(scanner.ROI_signal)
scanner_dict['sz'] = sz
scanner_dict['adjoint_mtx'] = tonumpy(scanner.G_adj.permute([2,3,0,1,4]))
scanner_dict['signal'] = tonumpy(scanner.signal)

path=os.path.join('./out/',experiment_id)
try:
    os.makedirs(path)
except:
    pass
scipy.io.savemat(os.path.join(path,"scanner_dict.mat"), scanner_dict)

    