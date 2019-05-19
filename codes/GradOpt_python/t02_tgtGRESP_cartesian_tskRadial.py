#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
"""

experiment_id = 't02_tgtGRESP_cartesian_tskRadial'
sequence_class = "GRE"
experiment_description = """
target - cartesian grid GRE
learn: radial  (dont optimize gradmoms, hardset them to radial) + NN reco to compensate for density
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
sz = np.array([32,32])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 35**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements

noise_std = 0*1e0                               # additive Gaussian noise std

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
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
# forward/adjoint pass
scanner.forward_fast(spins, event_time)
scanner.adjoint(spins)

# try to fit this2
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target)

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
    
    if do_scanner_query:
        targetSeq.export_to_pulseq(experiment_id,sequence_class)
        scanner.send_job_to_real_system(experiment_id)
        scanner.get_signal_from_real_system(experiment_id)
        
        plt.subplot(121)
        scanner.adjoint(spins)
        plt.imshow(magimg(tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])), interpolation='none')
        plt.title("real measurement IFFT")
        plt.subplot(122)
        scanner.reco = scanner.do_ifft_reco()
        plt.imshow(magimg(tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])), interpolation='none')
        plt.title("real measurement ADJOINT")    
    
#############################################################################
## Optimization land ::: ####################################################
    
use_only_mag_asinput = True
use_multichannel_input = False
    
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
    scanner.set_gradient_precession_tensor(grad_moms,sequence_class) # GRE/FID specific, maybe adjust for higher echoes
         
    # forward/adjoint pass
    scanner.forward_fast(spins, event_time)
    scanner.adjoint(spins)
        
    return [adc_mask, flips, event_time, grad_moms]


    
def phi_FRP_model(opt_params,aux_params):
   
    if use_multichannel_input:
        reco_input = scanner.adjoint_separable(spins)
        reco_input = reco_input.permute([1,0,2])
        reco_input = reco_input.reshape([1,NVox,NRep*2])
    else:
        reco_input = scanner.reco.clone().reshape([1,NVox,2])
    tgt = targetSeq.target_image.clone()
    
    if use_only_mag_asinput:
        mag = magimg_torch(reco_input.squeeze())
        reco_input[0,:,0] = mag
        reco_input[0,:,1] = 0
        
        mag = magimg_torch(tgt)
        tgt[:,0] = mag
        tgt[:,1] = 0            
    cnn_output = CNN(reco_input)
    
    loss_image = (cnn_output - tgt)
    loss_image = torch.sum(loss_image.squeeze()**2/NVox)
    
    loss = loss_image
    
    print("loss_image: {}".format(loss_image))
    
    phi = loss
  
    ereco = tonumpy(cnn_output.detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(tgt).ravel(),ereco.ravel())     
    
    plt.imshow(magimg(ereco))
    
    return (phi,cnn_output, error)
        
# %% # OPTIMIZATION land
    
# set number of convolution neurons (number of elements in the list - number of layers, each element of the list - number of conv neurons)
if use_multichannel_input:
    nmb_conv_neurons_list = [NRep*2,32,2]
else:
    nmb_conv_neurons_list = [2,32,2]

# initialize reconstruction module
CNN = core.nnreco.RecoConvNet_basic(spins.sz, nmb_conv_neurons_list,3).cuda()

opt = core.opt_helper.OPT_helper(scanner,spins,CNN,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))
opt.target_seq_holder=targetSeq
opt.experiment_description = experiment_description
opt.learning_rate = 1e-1

opt.optimzer_type = 'Adam'
opt.opti_mode = 'nn'
# 
opt.set_opt_param_idx([]) # ADC, RF, time, grad

opt.set_handles(init_variables, phi_FRP_model)
opt.scanner_opt_params = opt.init_variables()

opt.train_model(training_iter=1000, do_vis_image=False, save_intermediary_results=False) # save_intermediary_results=1 if you want to plot them later

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
if use_multichannel_input:
    reco_input = scanner.adjoint_separable(spins)
    reco_input = reco_input.permute([1,0,2])
    reco_input = reco_input.reshape([1,NVox,NRep*2])    
else:
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
            

# %% export model to matlab with onnx
dummy_input = torch.randn(1, NVox, 2, device='cuda:0')
model = CNN

path=os.path.join('./out/',experiment_id)
try:
    os.mkdir(path)
except:
    print('export_to_onnx: directory already exists')

torch.onnx.export(model, dummy_input, os.path.join(path,"cnn_basic.onnx"), verbose=True)


