"""
Created on Tue Jan 29 14:38:26 2019

@author: mzaiss

experiment desciption:

2D imaging: GRE with spoilers and random phase cycling
GRE90spoiled_relax2s

"""

experiment_id = 'p11_tgtGRESPcart_tskFLASHradial_FA_ET_NNscaler_24_extrarep'
sequence_class = "GRE"
experiment_description = """
opt pitcher try different fwd procs
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

print(experiment_id)

double_precision = False
do_scanner_query = False

use_gpu = 1
gpu_dev = 3

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

# device setter
def setdevice(x):
    if double_precision:
        x = x.double()
    else:
        x = x.float()
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
sz = np.array([24,24])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 25**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*1e-3                               # additive Gaussian noise std
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
print('use_gpu = ' +str(use_gpu)) 

# B1plus
B1plus = torch.zeros((NCoils,1,NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([NCoils, NVox]))

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
flips[0,:,0] = 25*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
#flips[0,:,1] = torch.rand(flips.shape[1])*90*np.pi/180

# randomize RF phases
flips[0,:,1] = scanner.get_phase_cycler(NRep, 117)*np.pi/180

flips = setdevice(flips)

scanner.init_flip_tensor_holder()

scanner.B1plus = setdevice(B1plus)    
scanner.set_flip_tensor_withB1plus(flips)

# rotate ADC according to excitation phase
rfsign = (flips[0,:,0] < 0).float()
scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

relax_time = 0.0
# event timing vector 
event_time = torch.from_numpy(0.08*1e-3*np.ones((scanner.T,scanner.NRep))).float()
event_time[0,:] =  2e-3
event_time[1,:] =  0.5*1e-3   # for 96
event_time[-2,:] = 2*1e-3
event_time[-1,:] = 0.08e-3       + relax_time
event_time = setdevice(event_time)

TR=torch.sum(event_time[:,1])*1000
TE=torch.sum(event_time[:11,1])*1000

# gradient-driver precession
# Cartesian encoding
if False:
    grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 
    grad_moms[1,:,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
    grad_moms[1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))  # phase encoding in second event block
    grad_moms[2:-2,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep]) # ADC open, readout, freq encoding
    grad_moms[-2,:,0] = torch.ones(1)*sz[0]*2      # GRE/FID specific, SPOILER
    grad_moms[-2,:,1] = -grad_moms[1,:,1]      # GRE/FID specific, SPOILER
    grad_moms = setdevice(grad_moms)


# Radial encoding
if True:
    grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32) 
    grad_moms[1,:,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
    #grad_moms[1,:,1] = 0*torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))  # phase encoding in second event block
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

# end sequence 

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
# forward/adjoint pass
#scanner.forward_fast_supermem(spins, event_time)
scanner.forward_fast(spins, event_time)
for i in range(10):
    scanner.forward_fast(spins, event_time,do_dummy_scans=True)
    
genalpha = 7.5*1e-5
        
scanner.generalized_adjoint(alpha=genalpha,nmb_iter=55)

# try to fit this
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target)
if True: # check sanity: is target what you expect and is sequence what you expect
    #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')

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
                
stop()
    
    # %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
    
    
def init_variables():
    adc_mask = targetSeq.adc_mask.clone()
    #adc_mask.requires_grad = True     
    
    flips = targetSeq.flips.clone()
#    flips[0,:,:]=flips[0,:,:]
    flips = setdevice(flips)
    
    flip_mask = torch.ones((scanner.T, scanner.NRep, 2)).float()     
    flip_mask[1:,:,:] = 0
    flip_mask = setdevice(flip_mask)
    flips.zero_grad_mask = flip_mask
      
    event_time = targetSeq.event_time.clone()
#    event_time[-1,:] -=  relax_time           # remove the 1s of relaxation
    event_time = setdevice(event_time)
    
    event_time_mask = torch.ones((scanner.T, scanner.NRep)).float()        
    event_time_mask[:-1,:] = 0
    event_time_mask = setdevice(event_time_mask)
    event_time.zero_grad_mask = event_time_mask
        

    grad_moms = targetSeq.grad_moms.clone()
    
    grad_moms_mask = torch.zeros((scanner.T, scanner.NRep, 2)).float()        
    grad_moms_mask[1,:,:] = 1
    grad_moms_mask[-2,:,:] = 1
    grad_moms_mask = setdevice(grad_moms_mask)
    grad_moms.zero_grad_mask = grad_moms_mask
   
#    grad_moms[1,:,0] = grad_moms[1,:,0]*0    # remove rewinder gradients
#    grad_moms[1,:,1] = -grad_moms[1,:,1]*0 + setdevice(torch.rand(grad_moms[1,:,1].shape)-0.5)
#    
#    grad_moms[-2,:,0] = torch.ones(1)*sz[0]*0      # remove spoiler gradients
#    grad_moms[-2,:,1] = -grad_moms[1,:,1]*0      # GRE/FID specific, SPOILER
    
    scale_param = setdevice(torch.tensor(1).float())
        
    return [adc_mask, flips, event_time, grad_moms, scale_param]
    
    
def phi_FRP_model(opt_params,aux_params):
    adc_mask,flips,event_time,grad_moms,scale_param = opt_params
        
    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor_withB1plus(flips)
    # rotate ADC according to excitation phase
    rfsign = (flips[0,:,0] < 0).float()
    scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

          
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms,sequence_class) # GRE/FID specific, maybe adjust for higher echoes
         
    # forward/adjoint pass
    scanner.forward_fast_supermem(spins, event_time)
    scanner.generalized_adjoint(alpha=genalpha,nmb_iter=55)

    lbd_sar = 0*sz[0]**2         # switch on of SAR cost
    loss_image = (scale_param*scanner.reco - targetSeq.target_image)
    #loss_image = (magimg_torch(scanner.reco) - magimg_torch(targetSeq.target_image))   # only magnitude optimization
    loss_image = torch.sum(loss_image.squeeze()**2/NVox)
    
    #loss_diff = loss_image.reshape([sz[0],sz[1],2])
    #loss_imageX = loss_diff[1:,:,:] - loss_diff[:-1,:,:]
    #loss_imageY = loss_diff[:,1:,:] - loss_diff[:,:-1,:]
    
    #loss_image = torch.sum((loss_imageX.flatten()**2 + loss_imageY.flatten()**2).squeeze()/NVox)    
    
    loss_sar = torch.sum(flips[:,:,0]**2)/NRep
    
    lbd_kspace = 0
    k = torch.cumsum(grad_moms, 0)
    k = k*torch.roll(scanner.adc_mask, -1).view([T,1,1])
    k = k.flatten()
    mask = setdevice((torch.abs(k) > sz[0]/2).float())
    k = k * mask
    loss_kspace = torch.sum(k**2) / np.prod(sz)
    
    lbd_time = 0
    loss_time = torch.sum(event_time[-1,:]**2)/NRep    
    
#    ffwd = scanner.G_adj[2:-2,:,:,:2,:2].permute([0,1,2,4,3]).permute([0,1,3,2,4]).contiguous().view([NRep*(sz[1]+0)*2,NVox*2])
#    back = scanner.G_adj[2:-2,:,:,:2,:2].permute([0,1,2,4,3]).permute([0,1,3,2,4]).contiguous().view([NRep*(sz[1]+0)*2,NVox*2]).permute([1,0])
#    TT = torch.matmul(back,ffwd) / NVox    
#    lbd_ortho = 1e4
#    loss_ortho = torch.sum((TT-setdevice(torch.eye(TT.shape[0])))**2) / (NVox)    
    
    loss = loss_image + lbd_sar*loss_sar + lbd_kspace*loss_kspace + lbd_time*loss_time
    
    print("loss_image: {} loss_sar {} loss_kpspace {} loss_time {} scale_param {}".format(loss_image, lbd_sar*loss_sar, lbd_kspace*loss_kspace, lbd_time*loss_time, scale_param))
    
    phi = loss
  
    ereco = tonumpy(scale_param*scanner.reco.detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(targetSeq.target_image).ravel(),ereco.ravel())     
    
    return (phi,scale_param*scanner.reco, error)
        
# %% # OPTIMIZATION land

opt = core.opt_helper.OPT_helper(scanner,spins,None,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))
opt.target_seq_holder=targetSeq
opt.experiment_description = experiment_description

opt.optimzer_type = 'Adam'
opt.opti_mode = 'seq'
# 
opt.set_opt_param_idx([1]) # ADC, RF, time, grad
opt.custom_learning_rate = [0.01,0.01,0.1,0.1,0.1]

opt.set_handles(init_variables, phi_FRP_model)
opt.scanner_opt_params = opt.init_variables()

lr_inc=np.array([0.1, 0.2, 0.5, 0.5, 0.5, 0.5,0.5, 0.5,0.5, 0.5, 0.1, 0.1, 0.1, 0.05,0.01, 0.7 , 0.7, 0.7, 0.5, 0.2, 0.1,0.01])

for i in range(len(lr_inc)):
    opt.custom_learning_rate = [0.01,0.01,1e-2,lr_inc[i],0.1]
    print('<seq> Optimization ' + str(i+1) + ' starts now. lr=' +str(lr_inc[i]))
    opt.train_model(training_iter=300, do_vis_image=True, save_intermediary_results=True, adaptive_stopping=True) # save_intermediary_results=1 if you want to plot them later
opt.train_model(training_iter=10000, do_vis_image=False, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later


    
# %% # save optimized parameter history

opt.export_to_matlab(experiment_id, today_datestr)
opt.save_param_reco_history(experiment_id,today_datestr,sequence_class,generate_pulseq=False)
opt.save_param_reco_history_matlab(experiment_id,today_datestr)
opt.export_to_pulseq(experiment_id, today_datestr, sequence_class)
stop()

_,reco,error = phi_FRP_model(opt.scanner_opt_params, opt.aux_params)

# plot
targetSeq.print_status(True, reco=None)
opt.print_status(True, reco)

print("e: %f, total flipangle is %f °, total scan time is %f s," % (error, np.abs(tonumpy(opt.scanner_opt_params[1].permute([1,0]))).sum()*180/np.pi, tonumpy(torch.abs(opt.scanner_opt_params[2])[:,:,0].permute([1,0])).sum() ))

stop()



            