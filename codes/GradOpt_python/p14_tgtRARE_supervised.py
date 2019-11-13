#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
"""

experiment_id = 'p14_tgtRARE_supervised_basic96_div1'
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
gpu_dev = 1

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
sz = np.array([96,96])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 6**2                                # number of spin sims in each voxel
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
#real_phantom = scipy.io.loadmat('../../data/numerical_brain_cropped.mat')['cropped_brain']


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

csz = 4
nmb_samples = 128
spin_db_input = np.zeros((nmb_samples, sz[0], sz[1], 5), dtype=np.float32)
spin_db_input[:,:,:,1:3] = cutoff

for i in range(nmb_samples):
    rvx = np.int(np.floor(np.random.rand() * (sz[0] - csz)))
    rvy = np.int(np.floor(np.random.rand() * (sz[0] - csz)))
    
    b0 = (np.random.rand() - 0.5) * 120                            # -60..60 Hz
    b1 = 1 #+ 2*(np.random.rand() - 0.5) * 1e-1
    
    for j in range(rvx,rvx+csz):
        for k in range(rvy,rvy+csz):
            pd = 0.5 + np.random.rand()
            t2 = 0.3 + np.random.rand()
            t1 = t2 + np.random.rand()
              
            spin_db_input[i,j,k,0] = pd
            spin_db_input[i,j,k,1] = t1
            spin_db_input[i,j,k,2] = t2
            spin_db_input[i,j,k,3] = b0
            spin_db_input[i,j,k,4] = b1
            
    # smooth block
    spin_db_input[i,:,:,0] = scipy.ndimage.filters.gaussian_filter(spin_db_input[i,:,:,0].squeeze(), 1)

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
scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev)
adc_mask = torch.from_numpy(np.ones((T,1))).float()
adc_mask[:2]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

# RF events: flips and phases
flips = torch.zeros((T,NRep,2), dtype=torch.float32)
flips[0,0,0] = 90*np.pi/180  # RARE specific, RARE preparation part 1 : 90 degree excitation 
flips[0,0,1] = 90*np.pi/180  # SE preparation part 1 : 90 phase
flips[1,:,0] = 180*np.pi/180  # RARE specific, RARE preparation part 2 : 180 degree excitation 

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
TEd= 1.1*1e-3 # increase to reduce SAR
event_time = torch.from_numpy(0.05*1e-4*np.ones((scanner.T,scanner.NRep))).float()
event_time[0,1:] = 0.2*1e-3     # for TE2_180_2   delay only

event_time[1,:] =  1.7*1e-3 +TEd      # for TE2_180     180 + prewinder   
event_time[0,0] = torch.sum(event_time[1:int(sz[0]/2+2),1])     # for TE2_90      90 +  rewinder
#event_time[1:,0,0] = 0.2*1e-3
event_time[-2,:] = 0.7*1e-3                         # spoiler
event_time[-1,:] = 0.8*1e-3    +TEd # for TE2_180_2   delay only
event_time = setdevice(event_time)

TE2_90   = torch.sum(event_time[0,0])*1000  # time after 90 until 180
TE2_180  = torch.sum(event_time[1:int(sz[0]/2+2),1])*1000 # time after 180 til center k-space
TE2_180_2= (torch.sum(event_time[int(sz[0]/2+2):,1])+event_time[0,1])*1000 # time after center k-space til next 180
TACQ = torch.sum(event_time)*1000

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

# xgradmom
grad_moms[2:-2,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep]) # read
grad_moms[1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))        # yblip
grad_moms[-2,:,1] = -torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))      # backblip
grad_moms[0,0,0] = torch.ones((1,1))*sz[0]/2+ torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom

grad_moms[1,:,0] =  torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[-2,:,0] =  torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom

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
scanner.forward_fast_supermem(spins, event_time)
#scanner.init_signal()
scanner.adjoint()

# try to fit this
# scanner.reco = scanner.do_ifft_reco()
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target)
if True: # check sanity: is target what you expect and is sequence what you expect
    #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(tonumpy(scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(scanner.T)*scanner.NRep]) )
        plt.title("ROI_def %d" % scanner.ROI_def)
        fig = plt.gcf()
        fig.set_size_inches(16, 3)
    plt.show()

    scanner.do_SAR_test(flips, event_time)    
    targetSeq.export_to_matlab(experiment_id, today_datestr)
    targetSeq.export_to_pulseq(experiment_id,today_datestr,sequence_class)
    
    if do_scanner_query:
        scanner.send_job_to_real_system(experiment_id,today_datestr)
        scanner.get_signal_from_real_system(experiment_id,today_datestr)
        
        scanner.adjoint()
        
        targetSeq.meas_sig = scanner.signal.clone()
        targetSeq.meas_reco = scanner.reco.clone()
        
    targetSeq.print_status(True, reco=None, do_scanner_query=do_scanner_query)  
    
# Prepare target db: iterate over all samples in the DB
target_db = setdevice(torch.zeros((nmb_samples,NVox,2)).float())
    
for i in range(nmb_samples):
    spins.set_system(spin_db_input[i,:,:,:])
#    spins.omega = setdevice(omega)
    
    B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
    B1plus[:,0,:,0,0] = torch.from_numpy(spin_db_input[i,:,:,4].reshape([scanner.NCoils, scanner.NVox]))
    scanner.B1plus = setdevice(B1plus)    
    scanner.set_flip_tensor_withB1plus(flips)    
    
    scanner.forward_sparse_fast_supermem(spins, event_time)
    #scanner.forward_fast_supermem(spins, event_time)    
    scanner.adjoint()
    target_db[i,:,:] = scanner.reco.clone().squeeze() 
    print('iteration:', i)
  
# since we optimize only NN reco part, we can save time by doing fwd pass (radial enc) on all training examples
adjoint_reco_db = setdevice(torch.zeros((nmb_samples,NVox,2)).float())
adjoint_reco_phantom = setdevice(torch.zeros((1,NVox,2)).float())      
                    
#stop()
       
    # %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
        
def init_variables():
    adc_mask = targetSeq.adc_mask.clone()
    
    flips = targetSeq.flips.clone()
#    flips[0,:,:]=flips[0,:,:]
    flips = setdevice(flips)
    
    flip_mask = torch.zeros((scanner.T, scanner.NRep, 2)).float()     
    #flip_mask[2:,:,:] = 0
    flip_mask[1,:,:] = 1
    flip_mask = setdevice(flip_mask)
    flips.zero_grad_mask = flip_mask
      
    event_time = targetSeq.event_time.clone()
    #event_time = torch.from_numpy(1e-7*np.random.rand(scanner.T,scanner.NRep)).float()
    #event_time*=0.5
    #event_time[:,0] = 0.4*1e-3  
    #event_time[-1,:] = 0.012 # target is fully relaxed GRE (FA5), task is FLASH with TR>=12ms
    event_time = setdevice(event_time)
    
    event_time_mask = torch.zeros((scanner.T, scanner.NRep)).float()        
    #event_time_mask[2:-2,:] = 0
    event_time_mask[-1,:] = 1
    event_time_mask = setdevice(event_time_mask)
    event_time.zero_grad_mask = event_time_mask
        
    grad_moms = targetSeq.grad_moms.clone()

    grad_moms_mask = torch.zeros((scanner.T, scanner.NRep, 2)).float()        
    grad_moms_mask[1,:,1] = 1
    #grad_moms_mask[-2,:,:] = 1
    grad_moms_mask = setdevice(grad_moms_mask)
    grad_moms.zero_grad_mask = grad_moms_mask
    
    #grad_moms[1,:,0] = grad_moms[1,:,0]*0    # remove rewinder gradients
#    grad_moms[1,:,1] = -grad_moms[1,:,1]*0      # GRE/FID specific, SPOILER
    
    #grad_moms[-2,:,0] = torch.ones(1)*sz[0]*0      # remove spoiler gradients
    #grad_moms[-2,:,1] = -grad_moms[1,:,1]*0      # GRE/FID specific, SPOILER
    
    scale_param = setdevice(torch.tensor(1).float())
        
    return [adc_mask, flips, event_time, grad_moms, scale_param]
    
def reparameterize(opt_params):

    return opt_params

def phi_FRP_model(opt_params,aux_params):
    
    adc_mask,flips,event_time,grad_moms,scale_param = reparameterize(opt_params)
    
    scanner.init_flip_tensor_holder()
    # rotate ADC according to excitation phase
    scanner.set_ADC_rot_tensor(-flips[0,:,1]*0)  # GRE/FID specific, this must be the excitation pulse
          
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms,sequence_class) # RARE specific, maybe adjust for higher echoes
    
    # set input
    samp_idx = np.random.choice(nmb_samples,1)
    spins.set_system(spin_db_input[samp_idx,:,:,:].squeeze())
#    spins.omega = setdevice(omega)
    B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
    B1plus[:,0,:,0,0] = torch.from_numpy(spin_db_input[samp_idx,:,:,4].reshape([scanner.NCoils, scanner.NVox]))
    scanner.B1plus = setdevice(B1plus)    
    scanner.set_flip_tensor_withB1plus(flips)              
    tgt = target_db[samp_idx,:,:].clone()
    opt.set_target(tonumpy(tgt).reshape([sz[0],sz[1],2]))    
         
    # forward/adjoint pass
    scanner.forward_sparse_fast_supermem(spins, event_time)
    scanner.adjoint()

    lbd_sar = sz[0]**2*1e-2 * 1e-3 * 0.5
    
    loss_image = (scale_param*scanner.reco - tgt)
    #loss_image = (magimg_torch(scanner.reco) - magimg_torch(targetSeq.target_image))   # only magnitude optimization
    
#    loss_image = torch.sum(loss_image.squeeze()**2/NVox)
    loss_sar = torch.sum(flips[:,:,0]**2)
    
    loss_diff = loss_image.reshape([sz[0],sz[1],2])
    loss_imageX = loss_diff[1:,:,:] - loss_diff[:-1,:,:]
    loss_imageY = loss_diff[:,1:,:] - loss_diff[:,:-1,:]
    loss_image = torch.sum((loss_imageX.flatten()**2 + loss_imageY.flatten()**2).squeeze()/NVox)    
    
    loss = loss_image + lbd_sar*loss_sar
    
    print("loss_image: {} loss_sar {}  scale_param {} ".format(loss_image, lbd_sar*loss_sar, scale_param))
    
    phi = loss
  
    ereco = tonumpy(scale_param*scanner.reco.detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(tgt).ravel(),ereco.ravel())     
    
    plt.imshow(magimg(ereco))
    
    return (phi,scale_param*scanner.reco, error)
        
# %% # OPTIMIZATION land

opt = core.opt_helper.OPT_helper(scanner,spins,None,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))
opt.target_seq_holder=targetSeq
opt.experiment_description = experiment_description

opt.optimzer_type = 'Adam'
opt.opti_mode = 'seq'
# 
opt.set_opt_param_idx([1,4]) # ADC, RF, time, grad
opt.custom_learning_rate = [0.01,1e-2,1e-2,1e-2,1e-2]

opt.set_handles(init_variables, phi_FRP_model,reparameterize)
opt.scanner_opt_params = opt.init_variables()

#opt.train_model_with_restarts(nmb_rnd_restart=20, training_iter=10,do_vis_image=True)

opt.train_model(training_iter=10000, do_vis_image=True, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later

_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)

# plot
targetSeq.print_status(True, reco=None)
opt.print_status(True, reco)

stop()

# %% # save optimized parameter history
targetSeq.export_to_matlab(experiment_id, today_datestr)
opt.export_to_pulseq(experiment_id, today_datestr, sequence_class)
opt.save_param_reco_history_compact(experiment_id,today_datestr,sequence_class,generate_pulseq=False)

#opt.export_to_matlab(experiment_id, today_datestr)
#opt.save_param_reco_history(experiment_id,today_datestr,sequence_class,generate_pulseq=False)
#opt.save_param_reco_history_matlab(experiment_id,today_datestr)

