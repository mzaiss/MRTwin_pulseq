#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
"""

experiment_id = 'p08_tgtFLAIR_RARE_tsklowSAR_scaler'
sequence_class = "RARE"
experiment_description = """
RARE, cpmg
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

use_gpu = 1
gpu_dev = 2

if sys.platform != 'linux':
    use_gpu = 0
    gpu_dev = 0
    
do_scanner_query = False

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
sz = np.array([64,64])                                           # image size
NRep = sz[1]  + 1                                       # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 4**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
import time; today_datestr = time.strftime('%y%m%d')
noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]

#############################################################################
## Init spin system ::: #####################################

# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev)

cutoff = 1e-12

real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
real_phantom = scipy.io.loadmat('../../data/numerical_brain_cropped.mat')['cropped_brain']


real_phantom_resized = np.zeros((sz[0],sz[1],5), dtype=np.float32)
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
scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev)
adc_mask = torch.from_numpy(np.ones((T,1))).float()
adc_mask[:2]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

measRepStep = NRep-1
first_meas = np.arange(1,measRepStep+1)

# RF events: flips and phases
flips = torch.zeros((T,NRep,2), dtype=torch.float32)
flips[0,0,0] = 180*np.pi/180  # FLAIR pulse

flips[0,1,0] = 90*np.pi/180  # RARE specific, RARE preparation part 1 : 90 degree excitation 
flips[0,1,1] = 90*np.pi/180  # SE preparation part 1 : 90 phase
flips[1,first_meas,0] = 180*np.pi/180  # RARE specific, RARE preparation part 2 : 180 degree excitation 

flips = setdevice(flips)

scanner.init_flip_tensor_holder()
B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([scanner.NCoils, scanner.NVox]))
B1plus[B1plus == 0] = 1    # set b1+ to one, where we dont have phantom measurements
B1plus[:] = 1
scanner.B1plus = setdevice(B1plus)    
scanner.set_flip_tensor_withB1plus(flips)

# rotate ADC according to excitation phase
scanner.set_ADC_rot_tensor(flips[0,:,1]*0) #GRE/FID specific


# event timing vector 

event_time = torch.from_numpy(0.05*1e-4*np.ones((scanner.T,scanner.NRep))).float()

# first repetition
event_time[0,0] =  2e-3
event_time[1,0] =  0.5*1e-3
event_time[-2,0] = 2*1e-3
event_time[-1,0] = 3.0*1e-3 + 3  # Thats the FLAIR IR time 


TEd= 0.5*1e-3 # increase to reduce SAR
event_time[0,2:] = 0.2*1e-3     # for TE2_180_2   delay only
event_time[-1,first_meas] = 0.8*1e-3     # for TE2_180_2   delay only
event_time[1,first_meas] =  1.7*1e-3 +TEd      # for TE2_180     180 + prewinder   
event_time[0,1] = torch.sum(event_time[1:int(sz[0]/2+2),1])     # for TE2_90      90 +  rewinder
#event_time[1:,0,0] = 0.2*1e-3
event_time[-2,first_meas] = 0.7*1e-3 +TEd                        # spoiler
event_time = setdevice(event_time)

TE2_90   = torch.sum(event_time[0,1])*1000  # time after 90 until 180
TE2_180  = torch.sum(event_time[1:int(sz[0]/2+2),1])*1000 # time after 180 til center k-space
TE2_180_2= (torch.sum(event_time[int(sz[0]/2+2):,1])+event_time[0,1])*1000 # time after center k-space til next 180
TACQ = torch.sum(event_time)*1000

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

# xgradmom
grad_moms[2:-2,first_meas,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,measRepStep]) # read
grad_moms[1,first_meas,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(measRepStep))        # yblip
grad_moms[-2,first_meas,1] = -torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(measRepStep))      # backblip
grad_moms[0,1,0] = torch.ones((1,1))*sz[0]/2+ torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom

grad_moms[1,first_meas,0] =  torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[-2,first_meas,0] =  torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom

# reverse linear reordering
#grad_moms[1,:,1] = -grad_moms[1,:,1]
#grad_moms[-2,:,1] = -grad_moms[1,:,1]     # backblip

#grad_moms[[1,-2],:,1] = torch.roll(grad_moms[[1,-2],:,1],0,dims=[1])

#     centric ordering
#grad_moms[1,:,1] = 0
#for i in range(1,int(sz[1]/2)+1):
#    grad_moms[1,i*2-1,1] = (-i)
#    if i < sz[1]/2:
#        grad_moms[1,i*2,1] = i
#grad_moms[-2,:,1] = -grad_moms[1,:,1]     # backblip


grad_moms = setdevice(grad_moms)

# end sequence 
scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=True for RARE, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    

# forward/adjoint pass
scanner.forward_fast(spins, event_time)
#scanner.adjoint()

# try to fit this
# scanner.reco = scanner.do_ifft_reco()
reco_sep = scanner.adjoint_separable()

first_scan = reco_sep[first_meas,:,:].sum(0)
first_scan_kspace = tonumpy(scanner.signal[0,2:-2,first_meas,:2,0])
first_scan_kspace_mag = magimg(first_scan_kspace)

scanner.reco = first_scan

# try to fit this
# scanner.reco = scanner.do_ifft_reco()
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target)
if True: # check sanity: is target what you expect and is sequence what you expect
    #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')

    if True:
        # print results
        ax1=plt.subplot(231)
        ax=plt.imshow(magimg(tonumpy(first_scan).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('first scan')
        plt.ion()
        
        # print results
        ax1=plt.subplot(234)
        ax=plt.imshow(first_scan_kspace_mag, interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('first scan kspace')
        plt.ion()
        
        fig.set_size_inches(18, 7)
        
        plt.show()
    #stop()
        
    # %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
        
def init_variables():
    adc_mask = targetSeq.adc_mask.clone()
    
    flips = targetSeq.flips.clone()
#    flips[0,:,:]=flips[0,:,:]
    flips = setdevice(flips)
    
    flip_mask = torch.ones((scanner.T, scanner.NRep, 2)).float()     
    #flip_mask[0,:,:] = 0
    flip_mask[2:,:,:] = 0
    flip_mask = setdevice(flip_mask)
    flips.zero_grad_mask = flip_mask
      
    event_time = targetSeq.event_time.clone()
    #event_time = torch.from_numpy(1e-7*np.random.rand(scanner.T,scanner.NRep)).float()
    #event_time*=0.5
    #event_time[:,0] = 0.4*1e-3  
    #event_time[-1,:] = 0.012 # target is fully relaxed GRE (FA5), task is FLASH with TR>=12ms
    event_time = setdevice(event_time)
    
    event_time_mask = torch.ones((scanner.T, scanner.NRep)).float()        
    event_time_mask[2:-2,:] = 0
    event_time_mask = setdevice(event_time_mask)
    event_time.zero_grad_mask = event_time_mask
        
    grad_moms = targetSeq.grad_moms.clone()

    grad_moms_mask = torch.zeros((scanner.T, scanner.NRep, 2)).float()        
    grad_moms_mask[1,:,:] = 1
    grad_moms_mask[-2,:,:] = 1
    grad_moms_mask = setdevice(grad_moms_mask)
    grad_moms.zero_grad_mask = grad_moms_mask
    
    #grad_moms[1,:,0] = grad_moms[1,:,0]*0    # remove rewinder gradients
    #grad_moms[1,:,1] = -grad_moms[1,:,1]*0      # GRE/FID specific, SPOILER
    
    #grad_moms[-2,:,0] = torch.ones(1)*sz[0]*0      # remove spoiler gradients
    #grad_moms[-2,:,1] = -grad_moms[1,:,1]*0      # GRE/FID specific, SPOILER
        
    scale_param = setdevice(torch.ones(2).float())
        
    return [adc_mask, flips, event_time, grad_moms, scale_param]
    
def reparameterize(opt_params):

    return opt_params

def phi_FRP_model(opt_params,aux_params):
    
    adc_mask,flips,event_time,grad_moms,scale_param = reparameterize(opt_params)

    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor_withB1plus(flips)
    # rotate ADC according to excitation phase
    scanner.set_ADC_rot_tensor(-flips[0,:,1]*0)  # GRE/FID specific, this must be the excitation pulse
          
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms,sequence_class) # RARE specific, maybe adjust for higher echoes
         
    # forward/adjoint pass
    scanner.forward_fast(spins, event_time)
    reco_sep = scanner.adjoint_separable()
    scanner.reco = scale_param.view([1,2])*reco_sep[first_meas,:,:].sum(0)

    lbd = 0.4*1e1         # switch on of SAR cost
    #loss_image = (scanner.reco - targetSeq.target_image)
    loss_image = (scanner.reco - targetSeq.target_image)
    #loss_image = (magimg_torch(scanner.reco) - magimg_torch(targetSeq.target_image))   # only magnitude optimization
    loss_image = torch.sum(loss_image.squeeze()**2/NVox)
    loss_sar = torch.sum(flips[:,:,0]**2)

    lbd_kspace = 0.3*1e1
    loss_kspace=0.0
#    k = torch.cumsum(grad_moms, 0)
#    k = k*torch.roll(scanner.adc_mask, -1).view([T,1,1])
#    k = k.flatten()
#    mask = (torch.abs(k) > sz[0]/2).float()
#    k = k * mask
#    loss_kspace = torch.sum(k**2) / (NRep*torch.sum(scanner.adc_mask))
    
    loss = loss_image + lbd*loss_sar + lbd_kspace*loss_kspace
    
    print("loss_image: {} loss_sar {} loss_kspace {} scale_param {}".format(loss_image, lbd*loss_sar, lbd_kspace*loss_kspace, scale_param))
    
    phi = loss
  
    ereco = tonumpy((scanner.reco).detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(targetSeq.target_image).ravel(),ereco.ravel())     
    
    #plt.imshow(magimg(ereco))
    
    return (phi,scanner.reco, error)
        
# %% # OPTIMIZATION land

opt = core.opt_helper.OPT_helper(scanner,spins,None,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))
opt.target_seq_holder=targetSeq
opt.experiment_description = experiment_description

opt.optimzer_type = 'Adam'
opt.opti_mode = 'seq'
# 
opt.set_opt_param_idx([1,4]) # ADC, RF, time, grad
opt.custom_learning_rate = [0.01,0.01,0.1,0.1,0.1]

opt.set_handles(init_variables, phi_FRP_model,reparameterize)
opt.scanner_opt_params = opt.init_variables()

#opt.train_model_with_restarts(nmb_rnd_restart=20, training_iter=10,do_vis_image=True)

opt.train_model(training_iter=1000, do_vis_image=True, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later

_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)

# plot
targetSeq.print_status(True, reco=None)
opt.print_status(True, reco)

stop()

# %% # save optimized parameter history
targetSeq.export_to_matlab(experiment_id, today_datestr)
opt.export_to_matlab(experiment_id, today_datestr)

opt.save_param_reco_history(experiment_id,today_datestr,sequence_class,generate_pulseq=False)
opt.save_param_reco_history_matlab(experiment_id,today_datestr)
opt.export_to_pulseq(experiment_id, today_datestr, sequence_class)
            