#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
"""

experiment_id = 't03_freektraj'
sequence_class = "GRE"
experiment_description = """
tgt FLASHspoiled_relax20ms, with spoilers and random phase cycling
task find all grads except read ADC grads
opt: SARloss, kloss, 

this is the same as e05_tgtGRE_tskGREnogspoil.py, but now with more automatic restarting
and high initial learning rate
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

from importlib import reload
reload(core.scanner)

#import warnings
#warnings.simplefilter("error")

print('32x float forwardfast oS')

double_precision = False
use_supermem = False
do_scanner_query = False

use_gpu = 1
gpu_dev = 0

if sys.platform != 'linux':
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
def phaseimg(x):
    return np.angle(1j*x[:,:,1]+x[:,:,0])

# device setter
def setdevice(x):
    if double_precision:
        x = x.double()
    else:
        x = x.float()
    if use_gpu:
        x = x.cuda(gpu_dev)    
    return x
    
def stop():
    sys.tracebacklimit = 0
    class ExecutionControl(Exception): pass
    raise ExecutionControl('stopped by user')
    sys.tracebacklimit = 1000
# define setup
sz = np.array([8,8])                                           # image size
NRep = sz[1]                                          # number of repetitions
NEvnt = sz[0] + 4                                        # number of events F/R/P
NSpins = 20**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements

noise_std = 0*1e0                               # additive Gaussian noise std
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*sz[1]

#############################################################################
## Init spin system ::: #####################################

# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)
cutoff = 1e-12
real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
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
R2 = 30.0
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
scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,NEvnt,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)
adc_mask = torch.from_numpy(np.ones((NEvnt,1))).float()
adc_mask[:2]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

# RF events: rf_event and phases
rf_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
rf_event[0,:,0] = 5*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
#rf_event[0,0,0] = 45*np.pi/180 
#rf_event[0,:,1] = torch.rand(rf_event.shape[1])*90*np.pi/180

# randomize RF phases
rf_event[0,:,1] = scanner.get_phase_cycler(NRep,117)*np.pi/180

rf_event = setdevice(rf_event)

scanner.init_flip_tensor_holder()
B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([scanner.NCoils, scanner.NVox]))
B1plus[B1plus == 0] = 1    # set b1+ to one, where we dont have phantom measurements
scanner.B1plus = setdevice(B1plus)    
scanner.set_flip_tensor_withB1plus(rf_event)

# rotate ADC according to excitation phase
rfsign = ((rf_event[0,:,0]) < 0).float()
scanner.set_ADC_rot_tensor(-rf_event[0,:,1] + 0*np.pi/2 + np.pi*rfsign) #GRE/FID specific


# event timing vector 
event_time = torch.from_numpy(0.08*1e-3*np.ones((NEvnt,NRep))).float()
event_time[0,:] =  2e-3  + 0e-3 
event_time[1,:] =  0.5*1e-3   # for 96
event_time[-2,:] = 2*1e-3
event_time[-1,:] = 0
event_time = setdevice(event_time)

TR=torch.sum(event_time[:,1])
TE=torch.sum(event_time[:11,1])

# gradient-driver precession
# Cartesian encoding
gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32) 

gradm_event[1,:,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
gradm_event[1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))  # phase encoding blip in second event block
gradm_event[2:-2,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep]) # ADC open, readout, freq encoding
gradm_event[-2,:,0] = torch.ones(1)*sz[0]*2  # GRE/FID specific, SPOILER
gradm_event[-2,:,1] = -gradm_event[1,:,1]      # GRE/FID specific, yblip rewinder
gradm_event = setdevice(gradm_event)

#     centric ordering
if True:
    gradm_event[1,:,1] = 0
    for i in range(1,int(sz[1]/2)+1):
        gradm_event[1,i*2-1,1] = (-i)
        if i < sz[1]/2:
            gradm_event[1,i*2,1] = i
    gradm_event[-2,:,1] = -gradm_event[1,:,1]     # backblip

# end sequence 

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(gradm_event,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
# forward/adjoint pass
#scanner.forward_fast_supermem(spins, event_time)
scanner.forward_fast(spins, event_time,kill_transverse=False)
#scanner.init_signal()
scanner.adjoint()

# try to fit this
# scanner.reco = scanner.do_ifft_reco()
target = scanner.reco.clone()
  

import core.target_seq_holder
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,target)
targetSeq.print_seq_pic(True,plotsize=[12,9])
targetSeq.print_seq(plotsize=[12,9])
    # %% ###  

if False: # check sanity: is target what you expect and is sequence what you expect
    #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(tonumpy(scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(scanner.NEvnt)*scanner.NRep]) )
        plt.title("ROI_def %d" % scanner.ROI_def)
        fig = plt.gcf()
        fig.set_size_inches(16, 3)
    plt.show()

    scanner.do_SAR_test(rf_event, event_time)    
    targetSeq.export_to_matlab(experiment_id, today_datestr)
    if do_scanner_query:
        
        targetSeq.export_to_pulseq(experiment_id,today_datestr,sequence_class)
        scanner.send_job_to_real_system(experiment_id,today_datestr)
        scanner.get_signal_from_real_system(experiment_id,today_datestr)
        
        scanner.adjoint()
        
        targetSeq.meas_sig = scanner.signal.clone()
        targetSeq.meas_reco = scanner.reco.clone()
        
    targetSeq.print_status(True, reco=None, do_scanner_query=do_scanner_query)
        
    
NRep_orig = NRep
           
# reset scanner class to t2o repetitions
NRep = 2
NEvnt = sz[0] * 2 + 4
scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,NEvnt,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)
adc_mask = torch.from_numpy(np.ones((NEvnt,1))).float()
adc_mask[:4]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))
B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([scanner.NCoils, scanner.NVox]))
B1plus[B1plus == 0] = 1    # set b1+ to one, where we dont have phantom measurements
scanner.B1plus = setdevice(B1plus)    
rfsign = rfsign[:NRep]

# rescale target
targetSeq.target_image *= NRep / float(NRep_orig)

#    stop()
        
    # %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
        
def init_variables():
    adc_mask = targetSeq.adc_mask.clone()
    
    rf_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
    rf_event[0,:,0] = 5*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
    rf_event[0,:,1] = scanner.get_phase_cycler(NRep,117)*np.pi/180

    rf_event = setdevice(rf_event)
    
    flip_mask = torch.ones((NEvnt, NRep, 2)).float()     
    flip_mask[1:,:,:] = 0
    flip_mask = setdevice(flip_mask)
    rf_event.zero_grad_mask = flip_mask
      
    event_time = torch.from_numpy(0.08*1e-3*np.ones((NEvnt,NRep))).float()
    event_time[0,:] =  2e-3  + 0e-3 
    event_time[1,:] =  0.5*1e-3   # for 96
    event_time[-2,:] = 2*1e-3
    event_time[-1,:] = 0
    event_time = setdevice(event_time)
    
    event_time_mask = torch.ones((NEvnt, NRep)).float()        
    event_time_mask[2:-2,:] = 0
    event_time_mask = setdevice(event_time_mask)
    event_time.zero_grad_mask = event_time_mask
        
    gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32) 
    gradm_event[1,:,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
    gradm_event[1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))  # phase encoding blip in second event block
    gradm_event[2:-2,:,0] = torch.ones(int(NEvnt-4)).view(int(NEvnt-4),1).repeat([1,NRep]) # ADC open, readout, freq encoding
    gradm_event[-2,:,0] = torch.ones(1)*sz[0]*2  # GRE/FID specific, SPOILER
    gradm_event[-2,:,1] = -gradm_event[1,:,1]      # GRE/FID specific, yblip rewinder  
    gradm_event[:2,:,:] = 0
    gradm_event = setdevice(gradm_event)

    gradm_event_mask = torch.ones((NEvnt, NRep, 2)).float()        
    gradm_event_mask[:1,:,:] = 0
    gradm_event_mask[-1:,:,:] = 0
    gradm_event_mask = setdevice(gradm_event_mask)
    gradm_event.zero_grad_mask = gradm_event_mask

    #gradm_event[1,:,0] = gradm_event[2,:,0]*torch.rand(1)*0.1    # remove rewinder gradients 
    #gradm_event[1,:,0] = gradm_event[1,:,0]*0    # remove rewinder gradients
    #gradm_event[1,:,1] = -gradm_event[1,:,1]*0      # GRE/FID specific, SPOILER
    
    #gradm_event[-2,:,0] = torch.ones(1)*sz[0]*0      # remove spoiler gradients
    #gradm_event[-2,:,1] = -gradm_event[1,:,1]*0      # GRE/FID specific, SPOILER
    
    scale_param = setdevice(torch.tensor(0).float()) 
        
    return [adc_mask, rf_event, event_time, gradm_event, scale_param]
    
def reparameterize(opt_params):

    return opt_params

def phi_FRP_model(opt_params,aux_params):
    
    adc_mask,rf_event,event_time,gradm_event,scale_param = reparameterize(opt_params)

    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor_withB1plus(rf_event) 
    # rotate ADC according to excitation phase
#    scanner.set_ADC_rot_tensor(-rf_event[0,:,1] + np.pi/2)  # GRE/FID specific, this must be the excitation pulse
    scanner.set_ADC_rot_tensor(-rf_event[0,:,1] + 0*np.pi/2 + np.pi*rfsign) #GRE/FID specific
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(gradm_event,sequence_class) # GRE/FID specific, maybe adjust for higher echoes
         
    # forward/adjoint pass
    scanner.forward_fast(spins, event_time)
    
    if 1:
#        scanner.adjoint()
        scanner.generalized_adjoint(alpha=1e-2, nmb_iter=5)
    else:
        genalpha = 1.5*1e-2  
        scanner.generalized_adjoint(alpha=genalpha,nmb_iter=100)
        
    scanner.reco *= scale_param
    

    lbd = 0.01*1e1         # switch on of SAR cost
    loss_image = (scanner.reco - targetSeq.target_image)
    #loss_image = (magimg_torch(scanner.reco) - magimg_torch(targetSeq.target_image))   # only magnitude optimization
    loss_image = torch.sum(loss_image.squeeze()**2/NVox)
    loss_sar = torch.sum(rf_event[:,:,0]**2)
    
    lbd_kspace = 1e1
    
    k = torch.cumsum(gradm_event, 0)
    k = k*torch.roll(scanner.adc_mask, -1).view([NEvnt,1,1])
    k = k.flatten()
    mask = setdevice((torch.abs(k) > sz[0]/2))
    k = k * mask
    loss_kspace = torch.sum(k**2) / (NRep*torch.sum(scanner.adc_mask))
    
    loss = loss_image + lbd*loss_sar + lbd_kspace*loss_kspace
    
    print("loss_image: {} loss_sar {} loss_kspace {}".format(loss_image, lbd*loss_sar, lbd_kspace*loss_kspace))
    
    phi = loss
  
    ereco = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(targetSeq.target_image).ravel(),ereco.ravel())     
    
    return (phi,scanner.reco, error)
        
# %% # OPTIMIZATION land

opt = core.opt_helper.OPT_helper(scanner,spins,None,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))
opt.target_seq_holder=targetSeq
opt.experiment_description = experiment_description

opt.optimzer_type = 'Adam'
opt.opti_mode = 'seq'
# 
opt.set_opt_param_idx([3,4]) # ADC, RF, time, grad
opt.custom_learning_rate = [0.01,0.1,0.1,0.001,0.01]

opt.set_handles(init_variables, phi_FRP_model,reparameterize)
opt.scanner_opt_params = opt.init_variables()
print('<seq> Optimization starts now, use_gpu = ' +str(use_gpu)) 


lr_inc=np.array([0.1, 0.2, 0.5, 0.7, 0.5, 0.2, 0.1, 0.01])
lr_inc=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
#opt.train_model_with_restarts(nmb_rnd_restart=20, training_iter=10,do_vis_image=True)

query_kwargs = experiment_id, today_datestr, sequence_class

import imageio 
for i in range(7):
    opt.custom_learning_rate = [0.01,0.01,0.1,lr_inc[i],0.01]
    print('<seq> Optimization ' + str(i+1) + ' with 10 iters starts now. lr=' +str(lr_inc[i]))
    opt.train_model(training_iter=20*i+5, do_vis_image=True, save_intermediary_results=True,query_scanner=do_scanner_query,query_kwargs=query_kwargs) # save_intermediary_results=1 if you want to plot them later   
#    imageio.mimsave("current_32_rew.gif",opt.gif_array[::4],format='GIF', duration=0.06)  
opt.train_model(training_iter=5000, do_vis_image=True, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later
imageio.mimsave("current_32_rew.gif",opt.gif_array[::2],format='GIF', duration=0.06)  

_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)

# plot
targetSeq.print_status(True, reco=None)
opt.print_status(True, reco)

stop()

# %% # save gif
import imageio    
imageio.mimsave("current.gif",opt.gif_array,format='GIF', duration=0.05)  

# %% # save optimized parameter history
targetSeq.export_to_matlab(experiment_id, today_datestr)
opt.export_to_matlab(experiment_id, today_datestr)

opt.save_param_reco_history(experiment_id,today_datestr,sequence_class,generate_pulseq=False)
opt.save_param_reco_history_matlab(experiment_id,today_datestr)
opt.export_to_pulseq(experiment_id, today_datestr, sequence_class)
           


