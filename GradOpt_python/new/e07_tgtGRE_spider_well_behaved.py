#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: mzaiss

experiment desciption:

2D imaging: GRE with spoilers and random phase cycling
GRE90spoiled_relax2s

"""

experiment_id = 'e07_tgtGRE_spider_kspaceloss_centered_wellbehaved'
experiment_description = """
tgt FLASHspoiled_relax0.1s task find all grads except read ADC grads
this is the same as e05_tgtGRE_tskGREnogspoil.py, but now with more automatic restarting
and high initial learning rate
optimize for frequency encoding gradients in both X/Y, by assuming constant increment at each action
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

if sys.version_info[0] < 3:
    reload(core.spins)
    reload(core.scanner)
    reload(core.opt_helper)
else:
    import importlib

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
sz = np.array([16,16])                                           # image size
NRep = sz[1]+0                                          # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 25**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]


#############################################################################
## Init spin system ::: #####################################

# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev)

real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
real_phantom_resized = np.zeros((sz[0],sz[1],5), dtype=np.float32)
for i in range(5):
    t = cv2.resize(real_phantom[:,:,i], dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
    if i != 3:
        t[t < 0] = 0
    real_phantom_resized[:,:,i] = t
    
spins.set_system(real_phantom_resized)

cutoff = 1e-12
spins.T1[spins.T1<cutoff] = cutoff
spins.T2[spins.T2<cutoff] = cutoff
# end initialize scanned object
spins.T1*=1
spins.T2*=1
#plt.subplot(121)
#plt.imshow(real_phantom_resized[:,:,0], interpolation='none')
#plt.title("PD")
#plt.subplot(122)
#plt.imshow(real_phantom_resized[:,:,3], interpolation='none')
#plt.title("inhom")
#plt.show()

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
event_time[1,:] = 1e-3
event_time[-2,:] = 1e-3
event_time[-1,:] = 0.1 * 10
event_time = setdevice(event_time)

TR=torch.sum(event_time[:,1])

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 

grad_moms[1,:,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
grad_moms[1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))  # phase encoding in second event block
grad_moms[2:-2,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep]) # ADC open, readout, freq encoding
#grad_moms[-2,:,0] = torch.ones(1)*sz[0]*3      # GRE/FID specific, SPOILER
#grad_moms[-2,:,1] = -grad_moms[1,:,1]      # GRE/FID specific, SPOILER
grad_moms = setdevice(grad_moms)

# end sequence 

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,refocusing=False,wrap_k=False)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
# forward/adjoint pass

scanner.forward_fast(spins, event_time)
scanner.adjoint(spins)

# try to fit this
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target)

if True: # check sanity: is target what you expect and is sequence what you expect
    targetSeq.print_status(True, reco=None)
    
    #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(tonumpy(scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(scanner.T+1)*scanner.NRep]) )
        plt.title("ROI_def %d" % scanner.ROI_def)
        fig = plt.gcf()
        fig.set_size_inches(16, 3)
    plt.show()
    
    targetSeq.export_to_matlab(experiment_id)
                
    
#    stop()
    
    
    # %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
    
    
def init_variables():
    
    adc_mask = targetSeq.adc_mask.clone()
    #adc_mask.requires_grad = True     

    adc_grad_mask = torch.ones(adc_mask.shape).float()     
    adc_grad_mask[:2,:] = 0
    adc_grad_mask[-2:,:] = 0
    adc_grad_mask = setdevice(adc_grad_mask)
    adc_mask.zero_grad_mask = adc_grad_mask
    
    flips = targetSeq.flips.clone()
    flips = setdevice(flips)
    
    flip_mask = torch.ones((scanner.T, scanner.NRep, 2)).float()     
    flip_mask[1:,:,:] = 0
    flip_mask = setdevice(flip_mask)
    flips.zero_grad_mask = flip_mask
      
    event_time = targetSeq.event_time.clone()
    event_time = setdevice(event_time)
    
    event_time_mask = torch.ones((scanner.T, scanner.NRep)).float()        
    event_time_mask = setdevice(event_time_mask)
    event_time.zero_grad_mask = event_time_mask
    
    grad_moms = setdevice(torch.zeros((1, scanner.NRep, 2)).float())
    grad_moms_mask = torch.ones((1, scanner.NRep, 2)).float()        
    grad_moms_mask = setdevice(grad_moms_mask)
    grad_moms.zero_grad_mask = grad_moms_mask
    
    return [adc_mask, flips, event_time, grad_moms]

def reparameterize(opt_params):
    adc_mask,flips,event_time, grad_moms = opt_params
    
    regrad = setdevice(torch.zeros(scanner.T, scanner.NRep, 2).float())
    regrad[1,:,:] = -grad_moms[2,:,:]*sz[0]/2   # rewinder coupled to adc read
#    regrad[1,:,:] = grad_moms[1,:,:] # rewinder free
    regrad[2:-2,:,:] = grad_moms[2,:,:]*setdevice(torch.ones((sz[1],NRep,2)).float())
    regrad[-2,:,:] = grad_moms[-2,:,:]
    grad_moms_upd = regrad    
    grad_moms_upd.zero_grad_mask = grad_moms.zero_grad_mask
    
    return (adc_mask,flips,event_time, grad_moms_upd)


    
def phi_FRP_model(opt_params,aux_params):
    
    adc_mask,flips,event_time,grad_moms = reparameterize(opt_params)
    use_periodic_grad_moms_cap,_ = aux_params
    
    scanner.adc_mask = adc_mask
    
    #print(adc_mask)
        
    scanner.init_flip_tensor_holder()
    scanner.set_flipXY_tensor(flips)    
    # rotate ADC according to excitation phase
    scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2)  # GRE/FID specific, this must be the excitation pulse
          
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms,refocusing=False,wrap_k=False) # GRE/FID specific, maybe adjust for higher echoes
         
    # forward/adjoint pass
    scanner.forward_fast(spins, event_time)
    scanner.adjoint(spins)

    lbd = 0*0.1       # switch on of SAR cost
    loss_image = (scanner.reco - targetSeq.target_image)
    #loss_image = (magimg_torch(scanner.reco) - magimg_torch(targetSeq.target_image))   # only magnitude optimization
    loss_image = torch.sum(loss_image.squeeze()**2/NVox)
    loss_sar = torch.sum(flips[:,:,0]**2)
    
    lbd_kspace = 1e1
    
    k = torch.cumsum(grad_moms, 0)
    #k = k*torch.roll((scanner.adc_mask > 0).float(), -1).view([T,1,1])
    k = k.flatten()
    mask = (torch.abs(k) > sz[0]/2).float()
    k = k * mask
    loss_kspace = torch.sum(k**2) / np.prod(sz)    
    
    ffwd = scanner.G_adj[2:-2,:,:,:2,:2].permute([0,1,2,4,3]).permute([0,1,3,2,4]).contiguous().view([NRep*(sz[1]+0)*2,NVox*2])
    back = scanner.G_adj[2:-2,:,:,:2,:2].permute([0,1,2,4,3]).permute([0,1,3,2,4]).contiguous().view([NRep*(sz[1]+0)*2,NVox*2]).permute([1,0])
    TT = torch.matmul(back,ffwd) / NVox    
    TT[range(TT.shape[0]), range(TT.shape[0])] = setdevice(torch.zeros((TT.shape[0],)))
    
    use_ortho_loss = True
    if use_ortho_loss:
        lbd_ortho = 1e2
    else:
        lbd_ortho = 0
    loss_ortho = torch.sum(TT**2) / (NVox)    
    
    loss = loss_image + lbd*loss_sar + lbd_kspace*loss_kspace + lbd_ortho*loss_ortho
    
    print("loss_image: {} loss_sar {} loss_kspace {} loss_ortho {}".format(loss_image, lbd*loss_sar, lbd_kspace*loss_kspace,lbd_ortho*loss_ortho))
    
    phi = loss
  
    ereco = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(targetSeq.target_image).ravel(),ereco.ravel())     
    
    return (phi,scanner.reco, error)
        
# %% # OPTIMIZATION land

opt = core.opt_helper.OPT_helper(scanner,spins,None,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))
opt.target_seq_holder=targetSeq
opt.experiment_description = experiment_description

opt.use_periodic_grad_moms_cap = 0           # GRE/FID specific, do not sample above Nyquist flag
opt.optimzer_type = 'Adam'
opt.opti_mode = 'seq'
# 
opt.set_opt_param_idx([0,3]) # ADC, RF, time, grad
opt.custom_learning_rate = [0.05,0.01,0.1,0.1]

opt.set_handles(init_variables, phi_FRP_model,reparameterize)
opt.scanner_opt_params = opt.init_variables()

#opt.train_model_with_restarts(nmb_rnd_restart=20, training_iter=10,do_vis_image=True)
print('<seq> Optimizing starts now...')
#for i in range(5):
#    opt.train_model(training_iter=2*i, do_vis_image=True, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later
#lr_inc=np.array([0.1, 0.2, 0.5, 0.7, 0.5, 0.2, 0.1, 0.1])
lr_inc=np.ones((7,))*0.1
#opt.train_model_with_restarts(nmb_rnd_restart=20, training_iter=10,do_vis_image=True)

for i in range(7):
    opt.custom_learning_rate = [0.01,0.1,0.1,lr_inc[i]]
    print('<seq> Optimization ' + str(i+1) + ' with 10 iters starts now. lr=' +str(lr_inc[i]))
    opt.train_model(training_iter=200, do_vis_image=True, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later
opt.train_model(training_iter=10000, do_vis_image=True, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later

_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)

# plot
targetSeq.print_status(True, reco=None)
opt.print_status(True, reco)

print("e: %f, total flipangle is %f °, total scan time is %f s," % (error, np.abs(tonumpy(opt.scanner_opt_params[1].permute([1,0]))).sum()*180/np.pi, tonumpy(torch.abs(opt.scanner_opt_params[2])[:,:,0].permute([1,0])).sum() ))

stop()

# %% # save optimized parameter history

# propagate readout gradient step from action 2 to all actions

opt.save_param_reco_history(experiment_id)
opt.export_to_matlab(experiment_id)
            