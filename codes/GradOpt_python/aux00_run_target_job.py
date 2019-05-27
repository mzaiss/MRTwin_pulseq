#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
"""


import os, sys
import numpy as np
import scipy
import scipy.io
from  scipy import ndimage
import torch
import cv2
import matplotlib.pyplot as plt
from torch import optim
import core.spins
import core.scanner
import core.opt_helper
import core.target_seq_holder

use_gpu = 0
gpu_dev = 0

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
        x = x.cuda(gpu_dev)    
    return x
    
def stop():
    sys.tracebacklimit = 0
    class ExecutionControl(Exception): pass
    raise ExecutionControl('stopped by user')
    sys.tracebacklimit = 1000
    
input_path = "/is/ei/aloktyus/Desktop/pulseq_mat_py/seq190527"
input_path = 'K:\CEST_seq\pulseq_zero\sequences\seq190527'

experiment_id = "t04_tgtBSSFP_tsk_BSSFP_32_alpha_2_prep_FA45_phaseincr5"
fullpath_seq = os.path.join(input_path, experiment_id)

use_target = False
    
if use_target:
    input_array = np.load(os.path.join(fullpath_seq, "target_arr.npy"))
    jobtype = "target"
else:
    input_array = np.load(os.path.join(fullpath_seq, "lastiter_arr.npy"))
    jobtype = "lastiter"
    
input_array = input_array.item()

# define setup
sz = input_array['sz']
NRep = sz[1]
T = sz[0] + 4
NSpins = 2**2
NCoils = input_array['signal'].shape[0]
noise_std = 0*1e0                               # additive Gaussian noise std
NVox = sz[0]*sz[1]

scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev)
scanner.set_adc_mask()

scanner.adc_mask = setdevice(torch.from_numpy(input_array['adc_mask']))
scanner.B1 = setdevice(torch.from_numpy(input_array['B1']))
scanner.signal = setdevice(torch.from_numpy(input_array['signal']))
scanner.reco = setdevice(torch.from_numpy(input_array['reco']).reshape([NVox,2]))
scanner.kspace_loc = setdevice(torch.from_numpy(input_array['kloc']))
sequence_class = input_array['sequence_class']

flips = setdevice(torch.from_numpy(input_array['flips']))
event_time = setdevice(torch.from_numpy(input_array['event_times']))
grad_moms = setdevice(torch.from_numpy(input_array['grad_moms']))

scanner.init_flip_tensor_holder()
scanner.set_flipXY_tensor(flips)

# rotate ADC according to excitation phase
scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2) #GRE/FID specific

TR=torch.sum(event_time[:,1])
TE=torch.sum(event_time[:11,1])

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

###############################################################################
######### SIMULATION

# simulation adjoint
scanner.adjoint()
sim_reco_adjoint = magimg(tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2]))

# simulation generalized adjoint
scanner.generalized_adjoint()
sim_reco_generalized_adjoint = magimg(tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2]))

# simulation IFFT
scanner.do_ifft_reco()
sim_reco_ifft = magimg(tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2]))

# simulation NUFFT
scanner.do_nufft_reco()
sim_reco_nufft = magimg(tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2]))

plt.subplot(221)
plt.imshow(sim_reco_adjoint, interpolation='none')
plt.title("sim ADJOINT")
plt.subplot(222)
plt.imshow(sim_reco_generalized_adjoint, interpolation='none')
plt.title("sim GENERALIZED ADJOINT") 
plt.subplot(223)
plt.imshow(sim_reco_ifft, interpolation='none')
plt.title("sim IFFT")
plt.subplot(224)
plt.imshow(sim_reco_nufft, interpolation='none')
plt.title("sim NUFFT") 

plt.ion()
plt.show()

coil_idx = 0
adc_idx = np.where(scanner.adc_mask.cpu().numpy())[0]
sim_kspace = scanner.signal[coil_idx,adc_idx,:,:2,0]
sim_kspace = magimg(tonumpy(sim_kspace.detach()).reshape([sz[0],sz[1],2]))

###############################################################################
######### REAL

# send to scanner
scanner.send_job_to_real_system(experiment_id, basepath_seq_override=fullpath_seq, jobtype=jobtype)
scanner.get_signal_from_real_system(experiment_id, basepath_seq_override=fullpath_seq, jobtype=jobtype)

real_kspace = scanner.signal[coil_idx,adc_idx,:,:2,0]
real_kspace = magimg(tonumpy(real_kspace.detach()).reshape([sz[0],sz[1],2]))

plt.subplot(121)
plt.imshow(sim_kspace, interpolation='none')
plt.title("sim kspace pwr")
plt.subplot(122)
plt.imshow(real_kspace, interpolation='none')
plt.title("real kspace pwr") 

plt.ion()
plt.show()

    
# real adjoint
scanner.adjoint()
real_reco_adjoint = magimg(tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2]))

# real generalized adjoint
scanner.generalized_adjoint()
real_reco_generalized_adjoint = magimg(tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2]))

# real IFFT
scanner.do_ifft_reco()
real_reco_ifft = magimg(tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2]))

# real NUFFT
scanner.do_nufft_reco()
real_reco_nufft = magimg(tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2]))

plt.subplot(221)
plt.imshow(real_reco_adjoint, interpolation='none')
plt.title("real ADJOINT")
plt.subplot(222)
plt.imshow(real_reco_generalized_adjoint, interpolation='none')
plt.title("real GENERALIZED ADJOINT") 
plt.subplot(223)
plt.imshow(real_reco_ifft, interpolation='none')
plt.title("real IFFT")
plt.subplot(224)
plt.imshow(real_reco_nufft, interpolation='none')
plt.title("real NUFFT") 

plt.ion()
plt.show()

    