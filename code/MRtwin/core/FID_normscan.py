#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
"""

experiment_id = 'FID_normscan'
sequence_class = "GRE"
experiment_description = """
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

import warnings
warnings.simplefilter("error")

print('32x float forwardfast oS')

double_precision = False
use_supermem = False


use_gpu = 0
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
    
def make_FID(phantom=None,do_scanner_query = False):    
    # define setup
    sz = np.array([phantom.shape[0],phantom.shape[1]])                                           # image size
    NRep = 1                                          # number of repetitions
    T = sz[0] + 4                                        # number of events F/R/P
    NSpins = 25**2                                # number of spin sims in each voxel
    NCoils = 1                                  # number of receive coil elements
    
    noise_std = 0*1e0                               # additive Gaussian noise std
    import time; today_datestr = time.strftime('%y%m%d')
    NVox = sz[0]*sz[1]
    
    #############################################################################
    ## Init spin system ::: #####################################
    
    # initialize scanned object
    spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)
    cutoff = 1e-12
    #real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
    #real_phantom = scipy.io.loadmat('../../data/numerical_brain_cropped.mat')['cropped_brain']
    real_phantom = np.zeros((128,128,5), dtype=np.float32); real_phantom[64:80,64:80,:]=1; real_phantom[64:80,64:80,3]=0
    
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
    
    if phantom.any()==None:
        spins.set_system(real_phantom_resized)
    else:
       spins.set_system(phantom)
    
    
    # end initialize scanned object
    
#    plt.subplot(151)
#    plt.imshow(real_phantom_resized[:,:,0], interpolation='none')
#    plt.title("PD")
#    plt.subplot(152)
#    plt.imshow(real_phantom_resized[:,:,1], interpolation='none')
#    plt.title("T1")
#    plt.subplot(153)
#    plt.imshow(real_phantom_resized[:,:,2], interpolation='none')
#    plt.title("T2")
#    plt.subplot(154)
#    plt.imshow(real_phantom_resized[:,:,3], interpolation='none')
#    plt.title("inhom")
#    plt.subplot(155)
#    plt.imshow(real_phantom_resized[:,:,4], interpolation='none')
#    plt.title("B1")
#    plt.show()
#    print('use_gpu = ' +str(use_gpu)) 
    
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
    scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)
    adc_mask = torch.from_numpy(np.ones((T,1))).float()
    adc_mask[:2]  = 0
    adc_mask[-2:] = 0
    scanner.set_adc_mask(adc_mask=setdevice(adc_mask))
    
    # RF events: rf_event and phases
    rf_event = torch.zeros((T,NRep,2), dtype=torch.float32)
    rf_event[0,:,0] = 90*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
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
    event_time = torch.from_numpy(0.08*1e-3*np.ones((scanner.T,scanner.NRep))).float()
    event_time[0,:] =  2e-3  + 0e-3 
    event_time[1,:] =  0.08*1e-3   # for 96
    event_time[-2,:] = 0.1*1e-3
    event_time[-1,:] = 0
    event_time = setdevice(event_time)
    
    
    # gradient-driver precession
    # Cartesian encoding
    gradm_event = torch.zeros((T,NRep,2), dtype=torch.float32) 
    gradm_event = setdevice(gradm_event)
    
    
    
    # end sequence 
    
    scanner.init_gradient_tensor_holder()
    scanner.set_gradient_precession_tensor(gradm_event,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes
    
    #############################################################################
    ## Forward process ::: ######################################################
        
    # forward/adjoint pass
    #scanner.forward_fast_supermem(spins, event_time)
    scanner.forward_sparse_fast(spins, event_time)
    #scanner.init_signal()
    scanner.signal=scanner.signal/NVox
        
    kspace = magimg(tonumpy(scanner.signal[0,2:-2,:,:2,0]))
#    plt.plot(kspace)
#    plt.plot([0,sz[1]],[kspace.mean(),kspace.mean()])
#    plt.show()
    normsim= kspace.mean()
    np.save("auxutil/normsim.npy", normsim)
    scanner.adjoint()
#    
#    # try to fit this
#    # scanner.reco = scanner.do_ifft_reco()
    target = scanner.reco.clone()
#       
#    # save sequence parameters and target image to holder object
    targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,target)
#    if True: # check sanity: is target what you expect and is sequence what you expect
#        #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')
#        for i in range(3):
#            plt.subplot(1, 3, i+1)
#            plt.plot(tonumpy(scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(scanner.T)*scanner.NRep]) )
#            plt.title("ROI_def %d" % scanner.ROI_def)
#            fig = plt.gcf()
#            fig.set_size_inches(16, 3)
#        plt.show()
#    
#        scanner.do_SAR_test(rf_event, event_time)    
#            
    if do_scanner_query:
#        targetSeq.export_to_matlab(experiment_id, today_datestr)
        targetSeq.export_to_pulseq(experiment_id,today_datestr,sequence_class)
        scanner.send_job_to_real_system(experiment_id,today_datestr)
        scanner.get_signal_from_real_system(experiment_id,today_datestr)
        
        scanner.signal=scanner.signal/NVox
        kspace = magimg(tonumpy(scanner.signal[0,2:-2,:,:2,0]))
        normmeas= kspace.mean()
        np.save("auxutil/normmeas.npy", normmeas)
#            
#            scanner.adjoint()
#            
#            targetSeq.meas_sig = scanner.signal.clone()
#            targetSeq.meas_reco = scanner.reco.clone()
#            
#        targetSeq.print_status(True, reco=None, do_scanner_query=do_scanner_query)
        
        
           