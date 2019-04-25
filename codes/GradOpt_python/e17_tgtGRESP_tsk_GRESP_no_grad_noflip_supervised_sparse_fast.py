#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:10:34 2019

@author: aloktyus
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: mzaiss

experiment desciption:

2D imaging: GRE with spoilers and random phase cycling
GRE90spoiled_relax2s

"""

experiment_id = 'e17_tgtGRESP_tsk_GRESP_no_grad_noflip_supervised_32_bsz8_sparse_fast'
experiment_description = """
tgt FLASHspoiled_relax0.1s task find all grads except read ADC grads
this is the same as e05_tgtGRE_tskGREnogspoil.py, but now with more automatic restarting
and high initial learning rate
lets try supervised learning here, learn flips+grads, and never show phantom during training
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
import core.scanner_batched
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
sz = np.array([16,16])                                               # image size
NRep = sz[1]                                            # number of repetitions
T = sz[0] + 4                                          # number of events F/R/P
NSpins = 40**2                               # number of spin sims in each voxel
NCoils = 1                                    # number of receive coil elements

noise_std = 0*1e0                                 # additive Gaussian noise std
NVox = sz[0]*sz[1]

batch_size = 1

#############################################################################
## Prepare spin DB  ::: #####################################

# initialize phantom object
real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
real_phantom_resized = np.zeros((sz[0],sz[1],5), dtype=np.float32)
for i in range(5):
    t = cv2.resize(real_phantom[:,:,i], dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
    if i != 3:
        t[t < 0] = 0
    real_phantom_resized[:,:,i] = t
    
real_phantom_resized[real_phantom_resized < 0] = 0

# initialize the training database, let it be just a bunch squares (<csz> x <csz>) with random PD/T1/T2
# ignore B0 inhomogeneity:-> since non-zero PD regions are just tiny squares, the effect of B0 is just constant phase accum in respective region
csz = 2
nmb_samples = 32
spin_db_input = np.zeros((nmb_samples, sz[0], sz[1], 5), dtype=np.float32)

for i in range(nmb_samples):
    rvx = np.int(np.floor(np.random.rand() * (sz[0] - csz)))
    rvy = np.int(np.floor(np.random.rand() * (sz[0] - csz)))
    
    pd = 0.5 + np.random.rand()
    t2 = 0.3 + np.random.rand()               # t2 is always smaller than t1...
    t1 = t2 + np.random.rand()
    
    for j in range(rvx,rvx+csz):
        for k in range(rvy,rvy+csz):
            spin_db_input[i,j,k,0] = pd
            spin_db_input[i,j,k,1] = t1
            spin_db_input[i,j,k,2] = t2
            
real_phantom_resized = spin_db_input[0,:,:,:]


#############################################################################
## Init spin system ::: #####################################
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev)
spins.set_system(real_phantom_resized)

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
event_time[-2,:] = 2*1e-3
event_time[-1,:] = 13.4*1e-3
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
scanner.set_gradient_precession_tensor(grad_moms,refocusing=False,wrap_k=False)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################

# forward/adjoint pass
scanner.forward_sparse(spins, event_time)
scanner.adjoint(spins)

# try to fit this
target_phantom = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target_phantom)

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
    
    #stop()
    
# Prepare target db: iterate over all samples in the DB
target_db = setdevice(torch.zeros((nmb_samples,NVox,2)).float())
    
for i in range(nmb_samples):
    spins.set_system(spin_db_input[i,:,:,:])
    
    scanner.forward_sparse(spins, event_time)
    scanner.adjoint(spins)
    
    target_db[i,:,:] = scanner.reco.clone().squeeze()
    

    # %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
    
    
def init_variables():
    adc_mask = targetSeq.adc_mask.clone()
    #adc_mask.requires_grad = True     
    
    flips = targetSeq.flips.clone()
    flips[0,:,:] = flips[0,:,:]*0
    flips = setdevice(flips)
    
    flip_mask = torch.ones((scanner.T, scanner.NRep, 2)).float()     
    flip_mask[1:,:,:] = 0
    flip_mask = setdevice(flip_mask)
    flips.zero_grad_mask = flip_mask
      
    event_time = targetSeq.event_time.clone()
    event_time = setdevice(event_time)
    
    event_time_mask = torch.ones((scanner.T, scanner.NRep)).float()        
    event_time_mask[2:-2,:] = 0
    event_time_mask = setdevice(event_time_mask)
    event_time.zero_grad_mask = event_time_mask
        
    use_gtruth_grads = True    # if this is changed also use_periodic_grad_moms_cap must be changed
    if use_gtruth_grads:
        grad_moms = targetSeq.grad_moms.clone()

    else:
        g = (np.random.rand(T,NRep,2) - 0.5)*2*np.pi
        grad_moms = torch.from_numpy(g).float()      
        grad_moms = setdevice(grad_moms)
        
    grad_moms_mask = torch.zeros((scanner.T, scanner.NRep, 2)).float()        
    grad_moms_mask[1,:,:] = 1
    grad_moms_mask[-2,:,:] = 1
    grad_moms_mask = setdevice(grad_moms_mask)
    grad_moms.zero_grad_mask = grad_moms_mask
    
    grad_moms[1,:,0] = grad_moms[1,:,0]*0    # remove rewinder gradients
    grad_moms[1,:,1] = -grad_moms[1,:,1]*0      # GRE/FID specific, SPOILER
    
    #grad_moms[-2,:,0] = torch.ones(1)*sz[0]*0      # remove spoiler gradients
    #grad_moms[-2,:,1] = -grad_moms[1,:,1]*0      # GRE/FID specific, SPOILER
        
    return [adc_mask, flips, event_time, grad_moms]
    
    
def phi_FRP_model(opt_params,aux_params,do_test_onphantom=False):
    
    adc_mask,flips,event_time, grad_moms = opt_params
    use_periodic_grad_moms_cap,_ = aux_params
        
    scanner.init_flip_tensor_holder()
    scanner.set_flipXY_tensor(flips)    
    # rotate ADC according to excitation phase
    scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2)  # GRE/FID specific, this must be the excitation pulse
          
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms,refocusing=False,wrap_k=False) # GRE/FID specific, maybe adjust for higher echoes
    
    # SAR cost
    lbd = 1*1e1         # switch on of SAR cost
    loss_sar = torch.sum(flips[:,:,0]**2)
    
    # out-of-kspace cost
    lbd_kspace = 1e1
    
    k = torch.cumsum(grad_moms, 0)
    k = k*torch.roll(scanner.adc_mask, -1).view([T,1,1])
    k = k.flatten()
    mask = (torch.abs(k) > sz[0]/2).float()
    k = k * mask
    loss_kspace = torch.sum(k**2) / (NRep*torch.sum(scanner.adc_mask))
    
    # once in a while we want to do test on real phantom, set batch_size to 1 in this case
    local_batchsize = batch_size
    if do_test_onphantom:
        local_batchsize = 1
      
    # loop over all samples in the batch, and do forward/backward pass for each sample
    loss_image = 0
    for btch in range(local_batchsize):
        if do_test_onphantom == False:
            samp_idx = np.random.choice(nmb_samples,1)
            spins.set_system(spin_db_input[samp_idx,:,:,:])
        
            tgt = target_db[samp_idx,:,:]
            opt.set_target(tonumpy(tgt).reshape([sz[0],sz[1],2]))
        else:
            tgt = target_phantom
        
        scanner.forward_sparse(spins, event_time)
        scanner.adjoint(spins)
        
        loss_diff = (scanner.reco - tgt)
        loss_image += torch.sum(loss_diff.squeeze()**2/(NVox))

    # accumulate final cost
    loss = loss_image + (lbd*loss_sar + lbd_kspace*loss_kspace)
    
    print("loss_image: {} loss_sar {} loss_kspace {}".format(loss_image, lbd*loss_sar, lbd_kspace*loss_kspace))
    
    phi = loss
  
    ereco = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(tgt).ravel(),ereco.ravel())     
    
    plt.imshow(magimg(ereco))
    
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
opt.set_opt_param_idx([1,3]) # ADC, RF, time, grad
opt.custom_learning_rate = [0.01,0.01,0.1,0.1]

opt.set_handles(init_variables, phi_FRP_model)
opt.scanner_opt_params = opt.init_variables()

lr_inc=np.array([0.1, 0.2, 0.5, 0.7, 0.5, 0.2, 0.1, 0.1])

for i in range(7):
    opt.custom_learning_rate = [0.01,0.01,0.1,lr_inc[i]]
    print('<seq> Optimization ' + str(i+1) + ' with 10 iters starts now. lr=' +str(lr_inc[i]))
    opt.train_model(training_iter=2000, do_vis_image=True, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later
opt.train_model(training_iter=10000, do_vis_image=True, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later


spins.set_system(real_phantom_resized)
_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params, do_test_onphantom=True)

# plot
targetSeq.print_status(True, reco=None)
opt.print_status(True, reco)

print("e: %f, total flipangle is %f °, total scan time is %f s," % (error, np.abs(tonumpy(opt.scanner_opt_params[1].permute([1,0]))).sum()*180/np.pi, tonumpy(torch.abs(opt.scanner_opt_params[2])[:,:,0].permute([1,0])).sum() ))

stop()

# %% # save optimized parameter history

opt.save_param_reco_history(experiment_id)
opt.export_to_matlab(experiment_id)




