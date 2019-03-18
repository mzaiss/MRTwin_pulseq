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
B0inhomo = B0inhomo*150 / np.max(B0inhomo)
B0inhomo = np.expand_dims(B0inhomo,2)
numerical_phantom = np.concatenate((numerical_phantom,B0inhomo),2)

spins.set_system(numerical_phantom)

cutoff = 1e-12
spins.T1[spins.T1<cutoff] = cutoff
spins.T2[spins.T2<cutoff] = cutoff
# end initialize scanned object
spins.T1*=1
spins.T2*=1
plt.subplot(121)
plt.imshow(numerical_phantom[:,:,0], interpolation='none')
plt.title("PD")
plt.subplot(122)
plt.imshow(numerical_phantom[:,:,3], interpolation='none')
plt.title("inhom")
plt.show()

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

# randomize pulse phases
flips[0,:,1] = torch.rand(flips.shape[1])*90*np.pi/180

flips = setdevice(flips)

scanner.init_flip_tensor_holder()
scanner.set_flipXY_tensor(flips)

# rotate ADC according to excitation phase
scanner.set_ADC_rot_tensor(-flips[0,:,1])

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
    # %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
    
def phi_FRP_model(opt_params,aux_params):
    
    flips,grads,event_time,adc_mask = opt_params
    use_periodic_grad_moms_cap,_ = aux_params
    
    flip_mask = torch.ones((scanner.T, scanner.NRep, 2)).float()        
#    flip_mask[2:,:,:] = 0
#    flip_mask[0,:,:] = 0
#    flip_mask[1,:,1] = 0                                        # all phases 0
    flip_mask = setdevice(flip_mask)
    flips = flips * flip_mask    
    
    #flips.data[0,0,0] = 90*np.pi/180  # SE preparation part 1 : 90 degree excitation
    #flips.data[0,0,1] = 90*np.pi/180  # SE preparation part 1 : 90 phase    
    
    flips[0,0,:] += 90*np.pi/180
    
    scanner.init_flip_tensor_holder()
    scanner.set_flipXY_tensor(flips)
    
    # gradients
    #grad_moms = torch.cumsum(grads,0)
    grad_moms = grads*1
    
    # dont optimize y  grads
    #grad_moms[:,:,1] = 0
    
    if use_periodic_grad_moms_cap:
        fmax = torch.ones([1,1,2]).float()
        fmax = setdevice(fmax)
        fmax[0,0,0] = sz[0]/2
        fmax[0,0,1] = sz[1]/2

        grad_moms = torch.sin(grad_moms)*fmax

    scanner.init_gradient_tensor_holder()          
    #scanner.set_gradient_precession_tensor_adjhistory(grad_moms)
    scanner.set_gradient_precession_tensor(grad_moms,refocusing=False,wrap_k=False)
    #scanner.set_gradient_precession_tensor(grad_moms,refocusing=False,wrap_k=True)
    
    #scanner.adc_mask = adc_mask
          
    # forward/adjoint pass
    scanner.forward(spins, event_time)
    scanner.adjoint(spins)

    lbd = 1e1
    loss_image = (scanner.reco - targetSeq.target_image)
    #loss_image = (magimg_torch(scanner.reco) - magimg_torch(targetSeq.target_image))
    loss_image = torch.sum(loss_image.squeeze()**2/NVox)
    loss_sar = torch.sum(flips[:,:,0]**2)
    
    loss = loss_image + lbd*loss_sar
    
    print("loss_image: {} loss_sar {}".format(loss_image, lbd*loss_sar))
    
    #phi = torch.sum((1.0/NVox)*torch.abs(loss.squeeze())**2)
    phi = loss

    #ereco = tonumpy(magimg_torch(scanner.reco.detach()))
    #error = e(tonumpy(magimg_torch(targetSeq.target_image)).ravel(),ereco.ravel())     
    
    ereco = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(targetSeq.target_image).ravel(),ereco.ravel())     
    
    return (phi,scanner.reco, error)
    

def init_variables():
    
    use_gtruth_grads = True    # if this is changed also use_periodic_grad_moms_cap must be changed
    if use_gtruth_grads:
        grad_moms = targetSeq.grad_moms.clone()
        
#        padder = torch.zeros((1,scanner.NRep,2),dtype=torch.float32)
#        padder = scanner.setdevice(padder)
#        temp = torch.cat((padder,grad_moms),0)
#        grads = temp[1:,:,:] - temp[:-1,:,:]   
    else:
        g = (np.random.rand(T,NRep,2) - 0.5)*2*np.pi
        
        grad_moms = torch.from_numpy(g).float()
        #grad_moms[:,:,1] = 0        
        grad_moms = setdevice(grad_moms)
    
    #grad_moms.requires_grad = True
    
    #flips = torch.zeros((T,NRep,2), dtype=torch.float32) 
    #flips[0,0,0] = 90*np.pi/180  # SE preparation part 1 : 90 degree excitation
    #flips[0,0,1] = 90*np.pi/180  # SE preparation part 1 : 90 phase
    
    flips = targetSeq.flips.clone()
    flips = setdevice(flips)
    flips.requires_grad = True
   
    event_time = targetSeq.event_time.clone()
    #event_time = torch.from_numpy(1e-7*np.random.rand(scanner.T,scanner.NRep,1)).float()
    #event_time*=0.5
    #event_time[:,0,0] = 0.4*1e-3  
    
    event_time = setdevice(event_time)
    #event_time.requires_grad = True

    adc_mask = targetSeq.adc_mask.clone()
    #adc_mask.requires_grad = True     
    
    return [flips, grad_moms, event_time, adc_mask]
    

    
# %% # OPTIMIZATION land

opt = core.opt_helper.OPT_helper(scanner,spins,None,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))

opt.use_periodic_grad_moms_cap = 0           # do not sample above Nyquist flag
opt.learning_rate = 0.01                                        # ADAM step size
opt.optimzer_type = 'Adam'

# fast track
# opt.training_iter = 10; opt.training_iter_restarts = 5

print('<seq> now (with 10 iterations and several random initializations)')
opt.opti_mode = 'seq'

#target_numpy = tonumpy(target).reshape([sz[0],sz[1],2])
#imshow(magimg(target_numpy), 'target')

opt.set_opt_param_idx([0])
#opt.custom_learning_rate = [0.1,0.01,0.1,0.1]
opt.custom_learning_rate = [0.2,0.01,0.001,0.01]

opt.set_handles(init_variables, phi_FRP_model)
opt.scanner_opt_params = opt.init_variables()


#opt.train_model_with_restarts(nmb_rnd_restart=2, training_iter=10,do_vis_image=True)
#opt.train_model_with_restarts(nmb_rnd_restart=1, training_iter=1)

#stop()

print('<seq> now (100 iterations with best initialization')
#opt.scanner_opt_params = opt.init_variables()
opt.train_model(training_iter=20000, do_vis_image=True, save_intermediary_results=False)
#opt.train_model(training_iter=10)


#event_time = torch.abs(event_time)  # need to be positive
#opt.scanner_opt_params = init_variables()

_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)
#reco = tonumpy(reco).reshape([sz[0],sz[1],2])

# plot
targetSeq.print_status(True, reco=None)
opt.print_status(True, reco)

print("e: %f, total flipangle is %f °, total scan time is %f s," % (error, np.abs(tonumpy(opt.scanner_opt_params[0].permute([1,0]))).sum()*180/np.pi, tonumpy(torch.abs(opt.scanner_opt_params[2])[:,:,0].permute([1,0])).sum() ))


stop()

# %% # plot M as function of events

#plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:4]), label='x')
    
legs=['x','y','z']
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(tonumpy(scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(scanner.T+1)*scanner.NRep]) )
    plt.title("ROI_def %d, %s" % (scanner.ROI_def,legs[i]))
    fig = plt.gcf()
    fig.set_size_inches(16, 3)
plt.show()

# %% # plot M as function of time
for i in range(3):
    plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,:,0]).transpose([1,0]).reshape([(scanner.T+1)*scanner.NRep])),tonumpy(scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(scanner.T+1)*scanner.NRep,1]), label='x')
    plt.title("ROI_def %d" % scanner.ROI_def)
    plt.show()
    
# %% # save optimized parameter history
experiment_id = 'RARE_FA_OPT_fixrep1_90_adjflipgrad_spoiled'
#opt.save_param_reco_history(experiment_id)

opt.scanner_opt_params[0][0,0,:] = 90*np.pi/180
opt.export_to_matlab(experiment_id)
    
    

# %% # export to matlab
#experiment_id='RARE_FA_OPT_fixrep1_90_adjflipgrad'
experiment_id='RARE_FA_OPT_fixrep1_90_balanced'
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


    