"""
Created on Tue Jan 29 14:38:26 2019

@author: mzaiss

experiment desciption:

2D imaging: learn to predict T2 from GRE-optimized variations

"""

experiment_id = 'e31_predictT2_fromGRE_2Xmeas_gpu0'
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
import core.nnreco
import core.opt_helper
import core.target_seq_holder

from importlib import reload
reload(core.scanner)

print(experiment_id)

double_precision = False
use_supermem = True
do_scanner_query = False

use_gpu = 1
gpu_dev = 2

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
sz = np.array([32,32])                                           # image size
extraRep = 2
NRep = extraRep*sz[1] + 1                                   # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 26**2                                # number of spin sims in each voxel
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
flips[0,0,0] = 90*np.pi/180 

flips[0,1:,0] = 5*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 

# randomize RF phases
flips[0,1:,1] = torch.tensor(scanner.phase_cycler[:(NRep-1)]).float()*np.pi/180

flips = setdevice(flips)

scanner.init_flip_tensor_holder()

B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([scanner.NCoils, scanner.NVox]))
B1plus[B1plus == 0] = 1    # set b1+ to one, where we dont have phantom measurements
scanner.B1plus = setdevice(B1plus)    
scanner.set_flip_tensor_withB1plus(flips)

# rotate ADC according to excitation phase
rfsign = ((flips[0,:,0]) < 0).float()
scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific


# event timing vector 
event_time = torch.from_numpy(0.1*1e-4*np.ones((scanner.T,scanner.NRep))).float()

# first repetition
event_time[0,0] =  2e-3
event_time[1,0] =  0.5*1e-3
event_time[-2,0] = 2*1e-3
event_time[-1,0] = 2.9*1e-3 + 0.2

measRepStep = NRep//extraRep
first_meas = np.arange(1,measRepStep+1)
second_meas = np.arange(1+measRepStep,2*measRepStep+1)

# first measurement
event_time[0,first_meas] =  2e-3
event_time[1,first_meas] =  0.5*1e-3   # for 96
event_time[1,first_meas] =  5.5*1e-3   # for 96
event_time[-2,first_meas] = 2*1e-3
event_time[-1,first_meas] = 2.9*1e-3

event_time[-1,measRepStep] = 2.9*1e-3 + 0.5

# second measurement
event_time[0,second_meas] =  2e-3
event_time[1,second_meas] =  0.5*1e-3   # for 96
event_time[1,second_meas] =  5.5*1e-3   # for 96
event_time[-2,second_meas] = 2*1e-3
event_time[-1,second_meas] = 2.9*1e-3

event_time = setdevice(event_time)

TR=torch.sum(event_time[:,1])
TE=torch.sum(event_time[:11,1])

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32)

# first repetition
grad_moms[:,0,:] = 0
grad_moms[-2,0,0] = torch.ones(1)*sz[0]
grad_moms[-2,0,1] = torch.ones(1)*sz[1]

# first measurement
grad_moms[1,first_meas,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
grad_moms[1,first_meas,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(first_meas.size))  # phase encoding blip in second event block
grad_moms[2:-2,first_meas,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,first_meas.size]) # ADC open, readout, freq encoding
grad_moms[-2,first_meas,0] = torch.ones(1)*sz[0]*2  # GRE/FID specific, SPOILER
grad_moms[-2,first_meas,1] = -grad_moms[1,first_meas,1]      # GRE/FID specific, yblip rewinder

grad_moms[1,first_meas,1] = 0
for i in range(1,int(sz[1]/2)+1):
    grad_moms[1,1+i*2-1,1] = (-i)
    if i < sz[1]/2:
        grad_moms[1,1+i*2,1] = i
grad_moms[-2,first_meas,1] = -grad_moms[1,first_meas,1]

# second measurement
grad_moms[1,second_meas,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
grad_moms[1,second_meas,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(second_meas.size))  # phase encoding blip in second event block
grad_moms[2:-2,second_meas,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,second_meas.size]) # ADC open, readout, freq encoding
grad_moms[-2,second_meas,0] = torch.ones(1)*sz[0]*2  # GRE/FID specific, SPOILER
grad_moms[-2,second_meas,1] = -grad_moms[1,first_meas,1]      # GRE/FID specific, yblip rewinder

grad_moms[1,second_meas,1] = 0
for i in range(1,int(sz[1]/2)+1):
    grad_moms[1,1+measRepStep+i*2-1,1] = (-i)
    if i < sz[1]/2:
        grad_moms[1,1+measRepStep+i*2,1] = i
grad_moms[-2,second_meas,1] = -grad_moms[1,second_meas,1]


grad_moms = setdevice(grad_moms)

# end sequence 

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
#scanner.forward_sparse_fast_supermem(spins, event_time)
#scanner.forward_sparse_fast(spins, event_time)
scanner.forward_fast(spins, event_time)
#scanner.forward_mem(spins, event_time)
#scanner.init_signal()
reco_sep = scanner.adjoint_separable()

first_scan = reco_sep[first_meas,:,:].sum(0)
second_scan = reco_sep[second_meas,:,:].sum(0)

# try to fit this
# scanner.reco = scanner.do_ifft_reco()
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target)
if True: # check sanity: is target what you expect and is sequence what you expect
    #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')

    if True:
        # print results
        ax1=plt.subplot(121)
        ax=plt.imshow(magimg(tonumpy(first_scan).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('first scan')
        plt.ion()
        
        plt.subplot(122, sharex=ax1, sharey=ax1)
        ax=plt.imshow(magimg(tonumpy(second_scan).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('second scan')
        plt.ion()
        
        plt.show()
                    
    stop()
    
stop()

# %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
def init_variables():
    adc_mask = targetSeq.adc_mask.clone()
    #adc_mask.requires_grad = True     
    
    flips = targetSeq.flips.clone()
    #flips[0,:,:]=flips[0,:,:]*0
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
        
    grad_moms = targetSeq.grad_moms.clone()

    grad_moms_mask = torch.zeros((scanner.T, scanner.NRep, 2)).float()        
    grad_moms_mask = setdevice(grad_moms_mask)
    grad_moms.zero_grad_mask = grad_moms_mask
    
    return [adc_mask, flips, event_time, grad_moms]
    
    
def phi_FRP_model(opt_params,aux_params):
    adc_mask,flips,event_time, grad_moms = opt_params
        
    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor_withB1plus(flips)
    
    # rotate ADC according to excitation phase
    rfsign = (torch.sign(flips[0,:,0]) < 0).float()
    scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms,sequence_class) # GRE/FID specific, maybe adjust for higher echoes
    
    samp_idx = np.random.choice(nmb_samples,1)[0]

    spins.set_system(spin_db_input[samp_idx,:,:,:])
    #scanner.forward_sparse_fast_supermem(spins, event_time)
    scanner.forward_fast(spins, event_time)
    reco_all_rep = scanner.adjoint_separable()
    reco_all_rep = torch.reshape(reco_all_rep,[extraRep, NRep//extraRep,NVox,2]).sum(1)
    reco_all_rep = (torch.sum(reco_all_rep**2,2)).unsqueeze(2)
    
    reco_all_rep = torch.log(reco_all_rep)
    
    cnn_output = CNN(reco_all_rep).reshape([sz[0],sz[1]])
    
    target_image = target_db[samp_idx,:,:].reshape([sz[0],sz[1]])
    
    #non_zero_voxel_mask = setdevice(torch.from_numpy(spin_db_input[samp_idx,:,:,0] > 0))
    
    loss_image = (cnn_output - target_image)# * non_zero_voxel_mask
    loss_image = torch.sum(loss_image.squeeze()**2/NVox)
    
    loss = loss_image
    
    print("loss_image: {} ".format(loss_image))
    
    phi = loss
  
    ereco = tonumpy(cnn_output).reshape([sz[0],sz[1]])
    error = e(tonumpy(target_image).ravel(),ereco.ravel())     
    
    if True:
        # print results
        ax1=plt.subplot(121)
        ax=plt.imshow(tonumpy(target_image), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('target')
        plt.ion()
        
        plt.subplot(122, sharex=ax1, sharey=ax1)
        ax=plt.imshow(tonumpy(cnn_output), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('reco')
        plt.ion()
        
        plt.show()
    
    return (phi,cnn_output, error)
        
# %% # OPTIMIZATION land
#nmb_hidden_neurons_list = [2*NRep,8,1]
nmb_hidden_neurons_list = [2,12,1]
CNN = core.nnreco.VoxelwiseNet(spins.sz, nmb_hidden_neurons_list,use_gpu=use_gpu,gpu_device=gpu_dev)

opt = core.opt_helper.OPT_helper(scanner,spins,CNN,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],1]))
opt.target_seq_holder=targetSeq
opt.experiment_description = experiment_description

opt.optimzer_type = 'Adam'
opt.opti_mode = 'seqnn'
opt.batch_size = 1

# 
opt.set_opt_param_idx([1,2]) # ADC, RF, time, grad
opt.custom_learning_rate = [0.01,1e-4,1e-4,0.1]

opt.set_handles(init_variables, phi_FRP_model)
opt.scanner_opt_params = opt.init_variables()

opt.train_model(training_iter=350, do_vis_image=False, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later
 
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


            