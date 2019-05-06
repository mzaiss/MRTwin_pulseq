#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
"""

experiment_id = 'e23_tgtGRESP_cartesian_tsk_radial_sup_onlyNN'
experiment_description = """
target - cartesian grid GRE
learn: radial  (dont optimize gradmoms, hardset them to radial) + NN reco to compensate for density
supervised learning on pixels
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
import core.nnreco
import core.target_seq_holder

use_gpu = 1
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

# define setup
sz = np.array([32,32])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 35**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements

noise_std = 0*1e0                               # additive Gaussian noise std
batch_size = 16

NVox = sz[0]*sz[1]


#############################################################################
## Init spin system ::: #####################################

# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev)

real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
real_phantom_resized = np.zeros((sz[0],sz[1],5), dtype=np.float32)
cutoff = 1e-12
for i in range(5):
    t = cv2.resize(real_phantom[:,:,i], dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
    if i == 0:
        t[t < 0] = 0
    elif i == 1 or i == 2:
        t[t < cutoff] = cutoff
        
    real_phantom_resized[:,:,i] = t

real_phantom_resized[:,:,1] *= 1 # Tweak T1
real_phantom_resized[:,:,2] *= 1 # Tweak T2
real_phantom_resized[:,:,3] *= 1 # Tweak dB0

# initialize the training database, let it be just a bunch squares (<csz> x <csz>) with random PD/T1/T2
# ignore B0 inhomogeneity:-> since non-zero PD regions are just tiny squares, the effect of B0 is just constant phase accum in respective region
csz = 12
nmb_samples = 64
spin_db_input = np.zeros((nmb_samples, sz[0], sz[1], 5), dtype=np.float32)

for i in range(nmb_samples):
    
    csz = 4 + np.random.randint(16)
    
    rvx = np.int(np.floor(np.random.rand() * (sz[0] - csz)))
    rvy = np.int(np.floor(np.random.rand() * (sz[0] - csz)))
    
    b0 = (np.random.rand() - 0.5) * 120                            # -60..60 Hz
    
    for j in range(rvx,rvx+csz):
        for k in range(rvy,rvy+csz):
            pd = 0.5 + np.random.rand()
            t2 = 0.3 + np.random.rand()
            t1 = t2 + np.random.rand()
              
            spin_db_input[i,j,k,0] = pd
            spin_db_input[i,j,k,1] = t1
            spin_db_input[i,j,k,2] = t2
            spin_db_input[i,j,k,3] = b0
            
# scatter point
if False:
    spin_db_input = np.zeros((nmb_samples, sz[0], sz[1], 5), dtype=np.float32)
    
    for i in range(nmb_samples):
        for pt in range(csz**2):
            b0 = (np.random.rand() - 0.5) * 120                            # -60..60 Hz
            pd = 0.5 + np.random.rand()
            t2 = 0.3 + np.random.rand()
            t1 = t2 + np.random.rand()
            
            j = np.random.randint(sz[0])
            k = np.random.randint(sz[1])
              
            spin_db_input[i,j,k,0] = pd
            spin_db_input[i,j,k,1] = t1
            spin_db_input[i,j,k,2] = t2
            spin_db_input[i,j,k,3] = b0
            
#spin_db_input[0,:,:,:] = real_phantom_resized
            
pd_mask_db = setdevice(torch.from_numpy((spin_db_input[:,:,:,0] > 0).astype(np.float32)))
pd_mask_db = pd_mask_db.flip([1,2]).permute([0,2,1])
            
spins.set_system(real_phantom_resized)
# end initialize scanned object

plt.subplot(141)
plt.imshow(real_phantom_resized[:,:,0], interpolation='none')
plt.title("PD")
plt.subplot(142)
plt.imshow(real_phantom_resized[:,:,1], interpolation='none')
plt.title("T1")
plt.subplot(143)
plt.imshow(real_phantom_resized[:,:,2], interpolation='none')
plt.title("T2")
plt.subplot(144)

plt.imshow(real_phantom_resized[:,:,3], interpolation='none')
plt.title("inhom")
plt.show()
print('use_gpu = ' +str(use_gpu)) 

#begin nspins with R*
R2 = 0.0
omega = np.linspace(0+1e-5,1-1e-5,NSpins) - 0.5    # cutoff might bee needed for opt.
#omega = np.random.rand(NSpins,NVox) - 0.5
omega = np.expand_dims(omega[:],1).repeat(NVox, axis=1)
omega*=0.9  # cutoff large freqs
omega = R2 * np.tan ( np.pi  * omega)

if NSpins==1:
    omega[:,:]=0
    
spins.omega = torch.from_numpy(omega.reshape([NSpins,NVox])).float()
spins.omega = setdevice(spins.omega)
#end nspins with R*


#############################################################################
## Init scanner system ::: #####################################
scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev)
scanner.set_adc_mask()

# begin sequence definition

# allow for relaxation and spoiling in the first two and last two events (after last readout event)
scanner.adc_mask[:2]  = 0
scanner.adc_mask[-2:] = 0

# RF events: flips and phases
flips = torch.zeros((T,NRep,2), dtype=torch.float32)
flips[0,:,0] = 5*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
#flips[0,:,1] = torch.rand(flips.shape[1])*90*np.pi/180

# randomize RF phases
flips[0,:,1] = torch.tensor(scanner.phase_cycler[:NRep]).float()*np.pi/180

flips = setdevice(flips)

scanner.init_flip_tensor_holder()
scanner.set_flipXY_tensor(flips)

# rotate ADC according to excitation phase
scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.2*1e-3*np.ones((scanner.T,scanner.NRep))).float()
event_time[1,:] =  2e-3
event_time[-2,:] = 2*1e-3
event_time[-1,:] = 12.6*1e-3
event_time = setdevice(event_time)

TR=torch.sum(event_time[:,1])
TE=torch.sum(event_time[:11,1])

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

grad_moms[1,:,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
grad_moms[1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))  # phase encoding in second event block
grad_moms[2:-2,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep]) # ADC open, readout, freq encoding
grad_moms[-2,:,0] = torch.ones(1)*sz[0]*2      # GRE/FID specific, SPOILER
grad_moms[-2,:,1] = -grad_moms[1,:,1]      # GRE/FID specific, SPOILER
grad_moms = setdevice(grad_moms)

# end sequence 
scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,refocusing=False,)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
# forward/adjoint pass
scanner.forward_sparse_fast(spins, event_time)
scanner.adjoint(spins)

# try to fit this2
target_phantom = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target_phantom)

if True: # check sanity: is target what you expect and is sequence what you expect
    targetSeq.print_status(True, reco=None)
    
    #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(tonumpy(scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(scanner.T)*scanner.NRep]) )
        plt.title("ROI_def %d" % scanner.ROI_def)
        fig = plt.gcf()
        fig.set_size_inches(16, 3)
    plt.show()
    
    targetSeq.export_to_matlab(experiment_id)
    
# Prepare target db: iterate over all samples in the DB
target_db = setdevice(torch.zeros((nmb_samples,NVox,2)).float())
    
for i in range(nmb_samples):
    spins.set_system(spin_db_input[i,:,:,:])
    scanner.forward_sparse_fast(spins, event_time)
    scanner.adjoint(spins)
    target_db[i,:,:] = scanner.reco.clone().squeeze() 
  
# since we optimize only NN reco part, we can save time by doing fwd pass (radial enc) on all training examples
adjoint_reco_db = setdevice(torch.zeros((nmb_samples,NVox,2)).float())
adjoint_reco_phantom = setdevice(torch.zeros((1,NVox,2)).float())

    
#############################################################################
## Optimization land ::: ####################################################

use_only_mag_asinput = False
use_PD_masking_for_loss = False
    
def init_variables():
    adc_mask = targetSeq.adc_mask.clone()
    
    flips = targetSeq.flips.clone()
    flips = setdevice(flips)
    
    flip_mask = torch.ones((scanner.T, scanner.NRep, 2)).float()     
    flips.zero_grad_mask = flip_mask
      
    event_time = targetSeq.event_time.clone()
    event_time = setdevice(event_time)
    
    event_time_mask = torch.ones((scanner.T, scanner.NRep)).float()        
    event_time.zero_grad_mask = event_time_mask
        
    grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 
    
    grad_moms[1,:,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
    grad_moms[1,:,1] = 0*torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))  # phase encoding in second event block
    grad_moms[2:-2,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep]) # ADC open, readout, freq encoding
    
    for rep in range(NRep):
        alpha = torch.tensor(rep * 180 * (1.0/(NRep-1)) * np.pi / 180)
        
        rotomat = torch.zeros((2,2)).float()
        rotomat[0,0] = torch.cos(alpha)
        rotomat[0,1] = -torch.sin(alpha)
        rotomat[1,0] = torch.sin(alpha)
        rotomat[1,1] = torch.cos(alpha)
        
        # rotate grid
        grad_moms[1,rep,:] = (torch.matmul(rotomat,grad_moms[1,rep,:].unsqueeze(1))).squeeze()
        grad_moms[2:-2,rep,:] = (torch.matmul(rotomat.unsqueeze(0),grad_moms[2:-2,rep,:].unsqueeze(2))).squeeze()
    
    grad_moms[-2,:,0] = torch.ones(1)*sz[0]*2      # GRE/FID specific, SPOILER
    grad_moms[-2,:,1] = -grad_moms[1,:,1]      # GRE/FID specific, SPOILER
    grad_moms = setdevice(grad_moms)
    
    # precompute/make forward reco/pass once
    # in this experiment we optimize just for CNN params
    
    scanner.init_flip_tensor_holder()
    scanner.set_flipXY_tensor(flips)    
    # rotate ADC according to excitation phase
    scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2)  # GRE/FID specific, this must be the excitation pulse
          
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms,refocusing=False) # GRE/FID specific, maybe adjust for higher echoes
    
    for i in range(nmb_samples):
        spins.set_system(spin_db_input[i,:,:,:])
        scanner.forward_sparse_fast(spins, event_time)
        scanner.adjoint(spins)
        adjoint_reco_db[i,:,:] = scanner.reco.clone().squeeze()
        
    spins.set_system(real_phantom_resized)
    scanner.forward_sparse_fast(spins, event_time)
    scanner.adjoint(spins)
    adjoint_reco_phantom[0,:,:] = scanner.reco.clone().squeeze()        
         
    return [adc_mask, flips, event_time, grad_moms]

def phi_FRP_model(opt_params,aux_params,do_test_onphantom=False):
      
    # loop over all samples in the batch, and do forward/backward pass for each sample
    loss_image = 0
     
    if do_test_onphantom == True:                                     # testing
        reco_input = adjoint_reco_phantom[0,:,:].reshape([1,NVox,2])
        cnn_output = CNN(reco_input)
        tgt = target_phantom
        opt.set_target(tonumpy(tgt).reshape([sz[0],sz[1],2]))      
    else:                                                            # training
        reco_input = setdevice(torch.zeros((batch_size,NVox,2)).float())
        tgt = setdevice(torch.zeros((batch_size,NVox,2)).float())
        
        all_samp_idx = np.zeros((batch_size,))
        for i in range(batch_size):
            samp_idx = np.random.choice(nmb_samples,1)
            all_samp_idx[i] = samp_idx
            reco_input[i,:,:] = adjoint_reco_db[samp_idx,:,:].reshape([NVox,2])
            tgt[i,:,:] = target_db[samp_idx,:,:].clone()
            
            if use_only_mag_asinput:
                mag = magimg_torch(reco_input[i,:,:].squeeze())
                reco_input[i,:,0] = mag
                reco_input[i,:,1] = 0
                
                mag = magimg_torch(tgt[i,:,:])
                tgt[i,:,0] = mag
                tgt[i,:,1] = 0  
                   
        cnn_output = CNN(reco_input)
            
        # only compute loss within training pixels
        if use_PD_masking_for_loss:
            pixel_mask = pd_mask_db[all_samp_idx,:,:].reshape([NVox,1])
            cnn_output *= pixel_mask
            tgt *= pixel_mask
        opt.set_target(tonumpy(tgt[0,:,:]).reshape([sz[0],sz[1],2]))
           
    loss_diff = (cnn_output - tgt)
    loss_image += torch.sum(loss_diff.squeeze()**2/(NVox))
      
    loss = loss_image
  
    ereco = tonumpy(cnn_output.detach()).reshape([cnn_output.shape[0],sz[0],sz[1],2])
    error = e(tonumpy(tgt).ravel(),ereco.ravel())
    
    print("loss_image: {}".format(loss_image))
    
    if do_test_onphantom:
        ereco = ereco[0,:,:,:]
        plt.imshow(magimg(ereco))
        plt.show()
        plt.ion()
    
    return (loss,cnn_output,error)
        
# %% # OPTIMIZATION land
    
# set number of convolution neurons (number of elements in the list - number of layers, each element of the list - number of conv neurons)
nmb_conv_neurons_list = [2,128,32,2]

# initialize reconstruction module
CNN = core.nnreco.RecoConvNet_basic(spins.sz, nmb_conv_neurons_list,3).cuda()

opt = core.opt_helper.OPT_helper(scanner,spins,CNN,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))
opt.target_seq_holder=targetSeq
opt.experiment_description = experiment_description
opt.learning_rate = 1e-2

opt.optimzer_type = 'Adam'
opt.opti_mode = 'nn'
# 
opt.set_opt_param_idx([]) # ADC, RF, time, grad

opt.set_handles(init_variables, phi_FRP_model)
opt.scanner_opt_params = opt.init_variables()

for epi in range(400):
  opt.train_model(training_iter=1000, do_vis_image=False, save_intermediary_results=False) # save_intermediary_results=1 if you want to plot them later
  _,reco,error = phi_FRP_model(opt.scanner_opt_params, None, do_test_onphantom = True)
      

_,reco,error = phi_FRP_model(opt.scanner_opt_params, None)

# plot
targetSeq.print_status(True, reco=None)
opt.print_status(True, reco)

print("e: %f, total flipangle is %f °, total scan time is %f s," % (error, np.abs(tonumpy(opt.scanner_opt_params[1].permute([1,0]))).sum()*180/np.pi, tonumpy(torch.abs(opt.scanner_opt_params[2])[:,:,0].permute([1,0])).sum() ))

stop()

# %% # try to reconstruct real measurement data
datapath = '/media/upload3t/CEST_seq/pulseq_zero/sequences/seq190430/e15_tgtGRESP_cartesian_rotated_radial/raw_data.mat'
raw_data = scipy.io.loadmat(datapath)

adc_idx = np.where(tonumpy(scanner.adc_mask))[0]
scanner.signal[0,adc_idx,:,0,0] = setdevice(torch.from_numpy(np.real(raw_data['spectrum'])))
scanner.signal[0,adc_idx,:,1,0] = setdevice(torch.from_numpy(np.imag(raw_data['spectrum'])))

scanner.adjoint(spins)
reco_input = scanner.reco.clone().reshape([1,NVox,2])

if use_only_mag_asinput:
    mag = magimg_torch(reco_input.squeeze())
    reco_input[0,:,0] = mag
    reco_input[0,:,1] = 0

cnn_output = CNN(reco_input)
#cnn_output = reco_input
ereco = tonumpy(cnn_output.detach()).reshape([sz[0],sz[1],2])
plt.imshow(magimg(ereco))

# %% # save optimized parameter history

opt.save_param_reco_history(experiment_id)
opt.export_to_matlab(experiment_id)
            




