#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
"""

experiment_id = 'p02_tgtSE_T2mapping'
sequence_class = "SE"
experiment_description = """
SE ME, spin echo multi echo
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
gpu_dev = 0

if sys.platform != 'linux':
    use_gpu = 0
    gpu_dev = 0
    
do_scanner_query = True
double_precision = False
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
sz = np.array([20,20])                                           # image size
extraRep = 3
NRep = extraRep*sz[1]                                 # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 4**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
import time; today_datestr = time.strftime('%y%m%d')
noise_std = 0*1e0                               # additive Gaussian noise std

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
scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)

# begin sequence definition
# allow for relaxation and spoiling in the first two and last two events (after last readout event)
adc_mask = torch.from_numpy(np.ones((T,1))).float()
adc_mask[:2]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

# RF events: flips and phases
flips = torch.zeros((T,NRep,2), dtype=torch.float32)
flips[0,:,0] = 90*np.pi/180  # RARE specific, RARE preparation part 1 : 90 degree excitation 
flips[0,:,1] = 90*np.pi/180  # SE preparation part 1 : 90 phase
flips[1,:,0] = 180*np.pi/180  # RARE specific, RARE preparation part 2 : 180 degree excitation 

# randomize RF phases
measRepStep = NRep//extraRep

# flips[0,1:measRepStep+1,1] = 0
# flips[0,1+measRepStep:1+2*measRepStep,1] = 0
# flips[0,1+2*measRepStep:1+3*measRepStep,1] = 0

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

dTE= 700*1e-3


# event timing vector 
#event_time = torch.from_numpy(0.2*1e-3*np.ones((scanner.T,scanner.NRep))).float()
event_time = torch.from_numpy(0.05*1e-4*np.ones((scanner.T,scanner.NRep))).float()  # 128
#event_time[0,:] = 2*1e-3   
#event_time[0,1:] = 0.01*1e-3    + dTE/2
event_time[-1,:] = 0.01*1e-3
event_time[-2,:] = 1.98*1e-3 + 0.5

event_time = setdevice(event_time)

TE2_90   = torch.sum(event_time[0,0])  # time after 90 until 180
TE2_180  = torch.sum(event_time[1:int(sz[0]/2+2),1]) # time after 180 til center k-space
TE2_180_2= torch.sum(event_time[int(sz[0]/2+2):,1])+event_time[0,1] # time after center k-space til next 180
measRepStep = NRep//extraRep
first_meas = np.arange(0,measRepStep)
second_meas = np.arange(measRepStep,2*measRepStep)
third_meas = np.arange(2*measRepStep,3*measRepStep)

dTE= np.array([0,0.3,1.5])  # dTE in s
RFd = 2*1e-3
# first measurement
event_time[1,first_meas] = 2*1e-3    + dTE[0]/2
event_time[0,first_meas] = (sz[0]/2)*0.05*1e-4+ RFd + dTE[0]/2

event_time[-1,measRepStep-1] = event_time[-1,measRepStep-1] + 0.3   # add some relaxation between scans

# second measurement
event_time[1,second_meas] = 2*1e-3    + dTE[1]/2
event_time[0,second_meas] = (sz[0]/2)*0.05*1e-4+ RFd + dTE[1]/2
event_time[-1,2*measRepStep-1] = event_time[-1,2*measRepStep-1] + 0.3   # add some relaxation between scans

# third measurement
event_time[1,third_meas] = 2*1e-3    + dTE[2]/2
event_time[0,third_meas] = (sz[0]/2)*0.05*1e-4+ RFd + dTE[2]/2

event_time = setdevice(event_time)

TR=torch.sum(event_time[:,1])
TE=torch.sum(event_time[:11,1])

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 


# first measurement
grad_moms[0,first_meas,0] = torch.ones((1,1))*sz[0]/2+ torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[1,first_meas,0] = torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[1,first_meas,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(measRepStep))        # yblip
grad_moms[2:-2,first_meas,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,measRepStep]) # read
grad_moms[-2,first_meas,0] = torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[-2,first_meas,1] = -torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(measRepStep))      # backblip


# second measurement
grad_moms[0,second_meas,0] = torch.ones((1,1))*sz[0]/2+ torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[1,second_meas,0] = torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[1,second_meas,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(measRepStep))        # yblip
grad_moms[2:-2,second_meas,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,measRepStep]) # read
grad_moms[-2,second_meas,0] = torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[-2,second_meas,1] = -torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(measRepStep))      # backblip


# third measurement
grad_moms[0,third_meas,0] = torch.ones((1,1))*sz[0]/2+ torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[1,third_meas,0] = torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[1,third_meas,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(measRepStep))        # yblip
grad_moms[2:-2,third_meas,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,measRepStep]) # read
grad_moms[-2,third_meas,0] = torch.ones((1,1))*sz[0]  # RARE: rewinder after 90 degree half length, half gradmom
grad_moms[-2,third_meas,1] = -torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(measRepStep))      # backblip

grad_moms = setdevice(grad_moms)

# end sequence 
scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=True for RARE, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
#scanner.init_signal()
scanner.forward_fast(spins, event_time,kill_transverse=True)
reco_sep = scanner.adjoint_separable()

first_scan = reco_sep[first_meas,:,:].sum(0)
second_scan = reco_sep[second_meas,:,:].sum(0)
third_scan = reco_sep[third_meas,:,:].sum(0)

first_scan_kspace = tonumpy(scanner.signal[0,2:-2,first_meas,:2,0])
second_scan_kspace = tonumpy(scanner.signal[0,2:-2,second_meas,:2,0])
third_scan_kspace = tonumpy(scanner.signal[0,2:-2,third_meas,:2,0])

first_scan_kspace_mag = magimg(first_scan_kspace)
second_scan_kspace_mag = magimg(second_scan_kspace)
third_scan_kspace_mag = magimg(third_scan_kspace)

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
        
        plt.subplot(232, sharex=ax1, sharey=ax1)
        ax=plt.imshow(magimg(tonumpy(second_scan).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('second scan')
        plt.ion()
        
        # print results
        ax1=plt.subplot(234)
        ax=plt.imshow(first_scan_kspace_mag, interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('first scan kspace')
        plt.ion()
        
        plt.subplot(235, sharex=ax1, sharey=ax1)
        ax=plt.imshow(second_scan_kspace_mag, interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('second scan kspace')
        plt.ion()    
        
        plt.subplot(236, sharex=ax1, sharey=ax1)
        ax=plt.imshow(third_scan_kspace_mag, interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('third scan kspace')
        plt.ion()           
        
        # print results
        ax1=plt.subplot(233)
        ax=plt.imshow(magimg(tonumpy(third_scan).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('third scan')
        plt.ion()        
        
        fig.set_size_inches(18, 7)
        
        plt.show()
        
if True:
    targetSeq.export_to_pulseq(experiment_id,today_datestr,sequence_class,plot_seq=False)
    
    if do_scanner_query:
        scanner.send_job_to_real_system(experiment_id,today_datestr)
        scanner.get_signal_from_real_system(experiment_id,today_datestr)
        
        reco_sep = scanner.adjoint_separable()
        
        first_scan = reco_sep[first_meas,:,:].sum(0)
        second_scan = reco_sep[second_meas,:,:].sum(0)
        third_scan = reco_sep[third_meas,:,:].sum(0)
        
        first_scan_kspace = tonumpy(scanner.signal[0,2:-2,first_meas,:2,0])
        second_scan_kspace = tonumpy(scanner.signal[0,2:-2,second_meas,:2,0])
        third_scan_kspace = tonumpy(scanner.signal[0,2:-2,third_meas,:2,0])

        first_scan_kspace_mag = magimg(first_scan_kspace)
        second_scan_kspace_mag = magimg(second_scan_kspace)
        third_scan_kspace_mag = magimg(third_scan_kspace)        
        
        ax1=plt.subplot(231)
        ax=plt.imshow(magimg(tonumpy(first_scan).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('meas: first scan')
        plt.ion()
        
        plt.subplot(232, sharex=ax1, sharey=ax1)
        ax=plt.imshow(magimg(tonumpy(second_scan).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('meas: second scan')
        plt.ion()
        
        # print results
        ax1=plt.subplot(234)
        ax=plt.imshow(first_scan_kspace_mag, interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('first scan kspace')
        plt.ion()
        
        plt.subplot(235, sharex=ax1, sharey=ax1)
        ax=plt.imshow(second_scan_kspace_mag, interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('second scan kspace')
        plt.ion()            
        
        plt.subplot(233, sharex=ax1, sharey=ax1)
        ax=plt.imshow(magimg(tonumpy(third_scan).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('meas: third scan')
        plt.ion()   
        
        plt.subplot(236, sharex=ax1, sharey=ax1)
        ax=plt.imshow(third_scan_kspace_mag, interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('third scan kspace')
        plt.ion()         
        
        fig.set_size_inches(12, 7)
        
        plt.show()  
        
mag_echo1 = magimg(tonumpy(first_scan).reshape([sz[0],sz[1],2]))
magmask = mag_echo1 > np.mean(mag_echo1.ravel())/5
        
mag_echo1 = magimg(tonumpy(first_scan).reshape([sz[0],sz[1],2]))
mag_echo2 = magimg(tonumpy(second_scan).reshape([sz[0],sz[1],2]))
mag_echo3 = magimg(tonumpy(third_scan).reshape([sz[0],sz[1],2]))

mag_echo21 = (mag_echo1/mag_echo2)*magmask
mag_echo31 = (mag_echo1/mag_echo3)*magmask

dTE21 = torch.sum(event_time[:sz[0]+2,second_meas[0]]) - torch.sum(event_time[:sz[0]+2,first_meas[0]]) 
dTE31 = torch.sum(event_time[:sz[0]+2,third_meas[0]]) - torch.sum(event_time[:sz[0]+2,first_meas[0]]) 

T2map1= tonumpy(dTE21)/np.log(mag_echo21 + 1e-7)
T2map2= tonumpy(dTE31)/np.log(mag_echo31 + 1e-7)
T2map1 = T2map1*magmask
T2map2 = T2map2*magmask

T2map1 = T2map1.transpose([1,0])
T2map1 = T2map1[::-1,::-1]
T2map2 = T2map2.transpose([1,0])
T2map2 = T2map2[::-1,::-1]

plt.subplot(141)
ax1=plt.imshow(real_phantom_resized[:,:,2], interpolation='none')
fig = plt.gcf()
fig.colorbar(ax1)  
plt.clim(0,np.max(np.abs(real_phantom_resized[:,:,2])))
plt.title("t2 map real")
plt.ion() 

plt.subplot(142)
ax=plt.imshow(T2map1)
plt.clim(0,np.max(np.abs(real_phantom_resized[:,:,2]))) 
plt.title("T2 map echo 1")
plt.ion() 

plt.subplot(143)
ax=plt.imshow(T2map2)
plt.clim(0,np.max(np.abs(real_phantom_resized[:,:,2])))
plt.title("T2 map echo 2")
plt.ion() 

plt.subplot(144)
ax=plt.imshow((T2map1+T2map2)/2)
plt.clim(0,np.max(np.abs(real_phantom_resized[:,:,2])))
plt.title("T2 map avgd")
plt.ion() 

fig.set_size_inches(18, 7)


plt.show()  

#np.save("../../data/current_T2map.npy",T2map1)

stop()
        
    # %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
        
def init_variables():
    adc_mask = targetSeq.adc_mask.clone()
    
    flips = targetSeq.flips.clone()
#    flips[0,:,:]=flips[0,:,:]
    flips = setdevice(flips)
    
    flip_mask = torch.ones((scanner.T, scanner.NRep, 2)).float()     
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
        
    return [adc_mask, flips, event_time, grad_moms]
    
def reparameterize(opt_params):

    return opt_params

def phi_FRP_model(opt_params,aux_params):
    
    adc_mask,flips,event_time,grad_moms = reparameterize(opt_params)

    scanner.init_flip_tensor_holder()
    scanner.set_flipXY_tensor(flips)    
    # rotate ADC according to excitation phase
    scanner.set_ADC_rot_tensor(-flips[0,:,1]*0)  # GRE/FID specific, this must be the excitation pulse
          
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms,sequence_class) # RARE specific, maybe adjust for higher echoes
         
    # forward/adjoint pass
    scanner.forward_sparse_fast(spins, event_time)
    scanner.adjoint()

    lbd = 0.4*1e1         # switch on of SAR cost
    loss_image = (scanner.reco - targetSeq.target_image)
    #loss_image = (magimg_torch(scanner.reco) - magimg_torch(targetSeq.target_image))   # only magnitude optimization
    loss_image = torch.sum(loss_image.squeeze()**2/NVox)
    loss_sar = torch.sum(flips[:,:,0]**2)
    
    lbd_kspace = 0.3*1e1
    
    k = torch.cumsum(grad_moms, 0)
    k = k*torch.roll(scanner.adc_mask, -1).view([T,1,1])
    k = k.flatten()
    mask = (torch.abs(k) > sz[0]/2).float()
    k = k * mask
    loss_kspace = torch.sum(k**2) / (NRep*torch.sum(scanner.adc_mask))
    
    loss = loss_image + lbd*loss_sar + lbd_kspace*loss_kspace
    
    print("loss_image: {} loss_sar {} loss_kspace {}".format(loss_image, lbd*loss_sar, lbd_kspace*loss_kspace))
    
    phi = loss
  
    ereco = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(targetSeq.target_image).ravel(),ereco.ravel())     
    
    plt.imshow(magimg(ereco))
    
    return (phi,scanner.reco, error)
        
# %% # OPTIMIZATION land

opt = core.opt_helper.OPT_helper(scanner,spins,None,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))
opt.target_seq_holder=targetSeq
opt.experiment_description = experiment_description

opt.optimzer_type = 'Adam'
opt.opti_mode = 'seq'
# 
opt.set_opt_param_idx([1]) # ADC, RF, time, grad
opt.custom_learning_rate = [0.01,0.1,0.1,0.1]

opt.set_handles(init_variables, phi_FRP_model,reparameterize)
opt.scanner_opt_params = opt.init_variables()

#opt.train_model_with_restarts(nmb_rnd_restart=20, training_iter=10,do_vis_image=True)

opt.train_model(training_iter=1000, do_vis_image=False, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later

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
            