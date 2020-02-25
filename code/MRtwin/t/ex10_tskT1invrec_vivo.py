"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""
experiment_id = 'ex01_invrec_vivo'
sequence_class = "gre_dream"
experiment_description = """
2D imaging: T1 mapping from invrec GRE
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
import core.FID_normscan

from importlib import reload
reload(core.scanner)

double_precision = False
do_scanner_query = False

use_gpu = 1
gpu_dev = 0

if sys.platform != 'linux':
    use_gpu = 0
    gpu_dev = 0
print(experiment_id)    
print('use_gpu = ' +str(use_gpu)) 

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

def tomag_torch(x):
    return torch.sqrt(torch.sum(torch.abs(x)**2,-1))

# device setter
def setdevice(x):
    if double_precision:
        x = x.double()
    else:
        x = x.float()
    if use_gpu:
        x = x.cuda(gpu_dev)    
    return x 

#############################################################################
## S0: define image and simulation settings::: #####################################
sz = np.array([32,32])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                      # number of total repetitions
NEvnt = sz[0] + 7                               # number of events F/R/P
NSpins = 10**2                               # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*1e-3                          # additive Gaussian noise std
kill_transverse = False                     #
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*sz[1]

#############################################################################
## S1: Init spin system and phantom::: #####################################
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)

cutoff = 1e-12
#real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
real_phantom = scipy.io.loadmat('../../data/numerical_brain_cropped.mat')['cropped_brain']
#real_phantom = np.zeros((128,128,5), dtype=np.float32); real_phantom[64:80,64:80,:]=1; real_phantom[64:80,64:80,3]=0

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
real_phantom_resized[:,:,4] *= 1 # Tweak rB1

spins.set_system(real_phantom_resized)

param=['PD','T1','T2','dB0','rB1']
for i in range(5):
    plt.subplot(151+i), plt.title(param[i])
    ax=plt.imshow(real_phantom_resized[:,:,i], interpolation='none')
    fig = plt.gcf()
    fig.colorbar(ax) 
fig.set_size_inches(18, 3)
plt.show()
   
#begin nspins with R*
R2star = 30.0
omega = np.linspace(0+1e-5,1-1e-5,NSpins) - 0.5    # cutoff might bee needed for opt.
omega = np.expand_dims(omega[:],1).repeat(NVox, axis=1)
omega*=0.9  # cutoff large freqs
omega = R2star * np.tan ( np.pi  * omega)
spins.omega = torch.from_numpy(omega.reshape([NSpins,NVox])).float()
spins.omega = setdevice(spins.omega)
## end of S1: Init spin system and phantom ::: #####################################


#############################################################################
## S2: Init scanner system ::: #####################################
scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,NEvnt,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)

B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([scanner.NCoils, scanner.NVox]))
B1plus[B1plus == 0] = 1    # set b1+ to one, where we dont have phantom measurements
B1plus[:] = 1
scanner.B1plus = setdevice(B1plus)

#############################################################################
## S3: MR sequence definition ::: #####################################
# begin sequence definition
# allow for extra events (pulses, relaxation and spoiling) in the first five and last two events (after last readout event)
adc_mask = torch.from_numpy(np.ones((NEvnt,1))).float()
adc_mask[:5]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

# RF events: rf_event and phases
rf_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
rf_event[3,:,0] = 5*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
# randomize RF phases
measRepStep = NRep//extraMeas
for i in range(0,extraMeas):
    rf_event[0,i*measRepStep,0] = 180*np.pi/180 
    rf_event[3,i*measRepStep:(i+1)*measRepStep,1] = torch.tensor(scanner.phase_cycler[:(measRepStep)]).float()*np.pi/180
rf_event = setdevice(rf_event)

scanner.init_flip_tensor_holder()    
scanner.set_flip_tensor_withB1plus(rf_event)

# rotate ADC according to excitation phase
rfsign = ((rf_event[3,:,0]) < 0).float()
scanner.set_ADC_rot_tensor(-rf_event[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.08*1e-3*np.ones((NEvnt,NRep))).float()
#TI = torch.tensor(np.arange(0,0.5,0.05))
TI = torch.tensor([0.5,1,1.5,2,3,4,5,6,8,20])

# all measurements event times
event_time[3,:] =  2e-3
event_time[4,:] =  5.5*1e-3   # for 96
event_time[-2,:] = 2*1e-3
event_time[-1,:] = 2.9*1e-3

# first action
for i in range(0,extraMeas):
    event_time[2,i*measRepStep] = TI[i]
    if i>0:
        event_time[-1,i*measRepStep-1] = 12       #delay after readout before next inversion ( only after the first event played out)
event_time = setdevice(event_time)

TR=torch.sum(event_time[:,1])
TE=torch.sum(event_time[:11,1])

# gradient-driver precession
# Cartesian encoding
gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)

meas_indices=np.zeros((extraMeas,measRepStep))
for i in range(0,extraMeas):
    meas_indices[i,:] = np.arange(i*measRepStep,(i+1)*measRepStep)

for j in range(0,extraMeas):
    # second action after inversion pulse (chrusher)
    gradm_event[1,j*measRepStep] = 1e-2

    #  measurement
    gradm_event[4,meas_indices[j,:] ,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
    gradm_event[4,meas_indices[j,:] ,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(meas_indices[j,:].size))  # phase encoding blip in second event block
    gradm_event[5:-2,meas_indices[j,:] ,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,meas_indices[j,:].size]) # ADC open, readout, freq encoding
    gradm_event[-2,meas_indices[j,:] ,0] = torch.ones(1)*sz[0]*2  # GRE/FID specific, SPOILER
    gradm_event[-2,meas_indices[j,:] ,1] = -gradm_event[1,meas_indices[j,:] ,1]      # GRE/FID specific, yblip rewinder
    
    gradm_event[4,meas_indices[j,:] ,1] = 0
    for i in range(1,int(sz[1]/2)+1):
        gradm_event[4,j*measRepStep+i*2-1,1] = (-i)
        if i < sz[1]/2:
            gradm_event[4,j*measRepStep+i*2,1] = i
    gradm_event[-2,meas_indices[j,:] ,1] = -gradm_event[4,meas_indices[j,:] ,1]
    
#gradm_event[:] = 0
gradm_event = setdevice(gradm_event)

# end sequence 

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(gradm_event,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes


#############################################################################
## S4: MR simulation forward process ::: #####################################
scanner.init_signal()
scanner.forward_sparse_fast(spins, event_time,kill_transverse=kill_transverse)
scanner.signal=scanner.signal/NVox
    

#############################################################################
## S5: MR reconstruction of signal ::: #####################################
reco_sep = scanner.adjoint_separable()

reco_all_rep=torch.zeros((extraMeas,reco_sep.shape[1],2))
for j in range(0,extraMeas):
    reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)

scale = torch.max(tomag_torch(reco_all_rep)) #last point for normalization
reco_testset = reco_all_rep / scale

first_scan_kspace = tonumpy(scanner.signal[0,5:-2,meas_indices[0,:],:2,0])
first_scan_kspace_mag = magimg(first_scan_kspace)

# try to fit this
# scanner.reco = scanner.do_ifft_reco()
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,target)
if True: # check sanity: is target what you expect and is sequence what you expect
    #plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')

    if True:
        # print results
        ax1=plt.subplot(231)
        ax=plt.imshow(magimg(tonumpy(reco_all_rep[0,:,:]).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('first scan')
        plt.ion()
        
        # print results
        ax1=plt.subplot(234)
        ax=plt.imshow(first_scan_kspace_mag, interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('first scan kspace')
        plt.ion()

        fig.set_size_inches(18, 7)
        
        plt.show()
        
if True:
    targetSeq.export_to_pulseq(experiment_id,today_datestr,sequence_class,plot_seq=True)
    
    if do_scanner_query:
        scanner.send_job_to_real_system(experiment_id,today_datestr)
        scanner.get_signal_from_real_system(experiment_id,today_datestr)
        
        normmeas=torch.from_numpy(np.load("auxutil/normmeas.npy"))
        scanner.signal=scanner.signal/normmeas/NVox
        
        reco_sep = scanner.adjoint_separable()
        
        first_scan = reco_sep[meas_indices[0,:],:,:].sum(0)
        second_scan = reco_sep[meas_indices[1,:],:,:].sum(0)
        third_scan = reco_sep[meas_indices[2,:],:,:].sum(0)
        
        first_scan_kspace = tonumpy(scanner.signal[0,2:-2,meas_indices[0,:],:2,0])
        second_scan_kspace = tonumpy(scanner.signal[0,2:-2,meas_indices[1,:],:2,0])
        
        first_scan_kspace_mag = magimg(first_scan_kspace)
        second_scan_kspace_mag = magimg(second_scan_kspace)
        
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
        plt.title('meas: first scan kspace')
        plt.ion()
        
        plt.subplot(235, sharex=ax1, sharey=ax1)
        ax=plt.imshow(second_scan_kspace_mag, interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('meas: second scan kspace')
        plt.ion()
        
        ax1=plt.subplot(233)
        ax=plt.imshow(magimg(tonumpy(third_scan).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('meas: first scan')
        plt.ion()    
        
        fig.set_size_inches(18, 7)
        
        plt.show()        
                        
    

    #
    #nn_input_muster = torch.stack((magimg_torch(first_scan),magimg_torch(second_scan)),1)
    
    # do real meas


    
# %% ###    stuff for supervised  ######################################################

with torch.no_grad():
    # target = T21
    target = setdevice(torch.from_numpy(real_phantom_resized[:,:,1]).float())
    targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,target)
    
    # Prepare target db: iterate over all samples in the DB
    target_db = setdevice(torch.zeros((nmb_samples,NVox,1)).float())
    reco_all_rep_premeas = setdevice(torch.zeros((nmb_samples,extraMeas,NVox,2)).float())
    
    
    for i in range(nmb_samples):
        print(i)
        tgt = torch.from_numpy(spin_db_input[i,:,:,1:2].reshape([NVox,1]))
        target_db[i,:,:] = tgt.reshape([sz[0],sz[1],1]).reshape([NVox,1])
          
        
        spins.set_system(spin_db_input[i,:,:,:])
        if first_run==0: #opt.opti_mode = 'seqnn'
            
            if True:   # recreate  from learned seqnn
                adc_mask,rf_event,event_time, gradm_event = reparameterize(opt.scanner_opt_params)
                scanner.set_adc_mask(adc_mask=setdevice(adc_mask))
                scanner.init_flip_tensor_holder()      
                scanner.set_flip_tensor_withB1plus(rf_event)
                # rotate ADC according to excitation phase
                rfsign = ((rf_event[3,:,0]) < 0).float()
                scanner.set_ADC_rot_tensor(-rf_event[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific
                scanner.init_gradient_tensor_holder()
                scanner.set_gradient_precession_tensor(gradm_event,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

    #            scanner.init_gradient_tensor_holder()       
            
            scanner.forward_sparse_fast(spins, event_time,kill_transverse=kill_transverse)
        
            core.FID_normscan.make_FID(spin_db_input[i,:,:,:])
            normsim=torch.from_numpy(np.load("auxutil/normsim.npy"))
            print(normsim)
            scanner.signal=scanner.signal/normsim/NVox    
               
            reco_sep = scanner.adjoint_separable()
            
            reco_all_rep=torch.zeros((extraMeas,reco_sep.shape[1],2))
            for j in range(0,extraMeas):
                reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)
                
            scale = torch.max(tomag_torch(reco_all_rep)) #last point for normalization                                      
            reco_all_rep_premeas[i,:] = reco_all_rep / scale
            
#            if i==0:
#                reco_testset = reco_all_rep  
    
    #reco_all_rep_premeas=reco_all_rep_premeas.reshape([nmb_samples,3,sz[0],sz[1],2]).permute([0,1,3,2,4]).flip([2,3]).reshape([nmb_samples,3,NVox,2])
    target_db=target_db.reshape([nmb_samples,sz[0],sz[1],1]).permute([0,2,1,3]).flip([1,2]).reshape([nmb_samples,NVox,1])
    
    
    samp_idx = np.random.choice(nmb_samples,1)[0]
#    samp_idx=0
    
    reco_all_rep = reco_all_rep_premeas[samp_idx,:extraMeas,:]
    
    reco_all_rep = torch.sqrt((reco_all_rep**2).sum(2))
    #    reco_all_rep = reco_all_rep.t()
    reco_all_rep = reco_all_rep.permute([1,0])
     
    target_image = target_db[samp_idx,:,:].reshape([sz[0],sz[1]])
    
    IMG=reco_all_rep =reco_all_rep.reshape([sz[0],sz[1],extraMeas,1]) 
    ax1=plt.subplot(121)
    plt.imshow(tonumpy(IMG[:,:,2,0]), interpolation='none')
    ax1=plt.subplot(122)
    plt.imshow(tonumpy(target_image), interpolation='none')

# %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
count_idx=0

def init_variables():
    adc_mask = targetSeq.adc_mask.clone()
    #adc_mask.requires_grad = True     
    
    rf_event = targetSeq.rf_event.clone()
    #rf_event[0,:,:]=rf_event[0,:,:]*0
    rf_event = setdevice(rf_event)
    
    flip_mask = torch.zeros((scanner.NEvnt, scanner.NRep, 2)).float()     
    flip_mask[3,:,:] = 1
    flip_mask = setdevice(flip_mask)
    rf_event.zero_grad_mask = flip_mask
      
    event_time = targetSeq.event_time.clone()
    event_time = setdevice(event_time)
       
    event_time_mask = torch.zeros((scanner.NEvnt, scanner.NRep)).float()   
    for j in range(0,extraMeas):
        event_time_mask[2,j*measRepStep] = 1         # optimize TI
    
    event_time_mask = setdevice(event_time_mask)
    event_time.zero_grad_mask = event_time_mask
        
    gradm_event = targetSeq.gradm_event.clone()

    gradm_event_mask = torch.zeros((scanner.NEvnt, scanner.NRep, 2)).float()        
    gradm_event_mask = setdevice(gradm_event_mask)
    gradm_event.zero_grad_mask = gradm_event_mask
    
    return [adc_mask, rf_event, event_time, gradm_event]


def reparameterize(opt_params):
    adc_mask,rf_event,event_time,gradm_event = opt_params
    return [adc_mask, rf_event, event_time, gradm_event]
    
    
def phi_FRP_model(opt_params,aux_params):
    adc_mask,rf_event,event_time,gradm_event = opt_params
             
    samp_idx = np.random.choice(nmb_samples,1)[0] 
    
    if opt.opti_mode == 'nn': #only NN from premeas
        plotdiv =150
        reco_all_rep = reco_all_rep_premeas[samp_idx,:extraMeas,:]
    else:
        plotdiv =1
        scanner.set_adc_mask(adc_mask)
        scanner.init_flip_tensor_holder()      
        scanner.set_flip_tensor_withB1plus(rf_event)
        # rotate ADC according to excitation phase
        rfsign = ((rf_event[3,:,0]) < 0).float()
        scanner.set_ADC_rot_tensor(-rf_event[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific
        scanner.init_gradient_tensor_holder()
        scanner.set_gradient_precession_tensor(gradm_event,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes
#        
        spins.set_system(spin_db_input[samp_idx,:,:,:])    
        scanner.forward_sparse_fast(spins, event_time,kill_transverse=kill_transverse)        
        core.FID_normscan.make_FID(spin_db_input[samp_idx,:,:,:])
        normsim=torch.from_numpy(np.load("auxutil/normsim.npy"))
        scanner.signal=scanner.signal/normsim/NVox    
           
        reco_sep = scanner.adjoint_separable()
        
        reco_all_rep=torch.zeros((extraMeas,reco_sep.shape[1],2))
        for j in range(0,extraMeas):
            reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)
    
    
    
    reco_all_rep = torch.sqrt((reco_all_rep**2).sum(2))
#    reco_all_rep = reco_all_rep.t()
    reco_all_rep = reco_all_rep.permute([1,0])
     
    target_image = target_db[samp_idx,:,:].reshape([sz[0],sz[1]])
    
#    IMG=reco_all_rep =reco_all_rep.reshape([sz[0],sz[1],3,1]) 
#    ax1=plt.subplot(121)
#    plt.imshow(tonumpy(IMG[:,:,2,0]), interpolation='none')
#    ax1=plt.subplot(122)
#    plt.imshow(tonumpy(target_image), interpolation='none')
    
    # sanity check
    #reco_all_rep[0,:,0] = target_image.view([NVox])
    
    non_zero_voxel_mask = setdevice(target_image > 1e-3)
    cnn_output = NN(reco_all_rep).reshape([sz[0],sz[1]])
    
    
    loss_image = (cnn_output - target_image) * non_zero_voxel_mask
    loss_image = torch.sum(loss_image.squeeze()**2/NVox)
    
    lbd_sar = 0*0.1*1e1         # switch on of SAR cost
    loss_sar = torch.sum(rf_event[:,:,0]**2)
    
    lbd_t = 1e-4         # switch on of time cost
    loss_t = lbd_t*torch.abs(torch.sum(event_time))
    
    loss = loss_image + lbd_sar*loss_sar + loss_t
        
    phi = loss
  
    ereco = tonumpy(cnn_output*non_zero_voxel_mask).reshape([sz[0],sz[1]])
    error = e(tonumpy(target_image*non_zero_voxel_mask).ravel(),ereco.ravel())     
    
    global count_idx
    count_idx=count_idx+1
    
    if np.mod(count_idx,plotdiv)==0:
        print("loss_image: {} ".format(loss_image),"loss_t: {} ".format(loss_t), "TIs {}".format((event_time[2,np.arange(0,extraMeas*measRepStep,measRepStep)])))
        # print results
        IMGs=reco_all_rep.reshape(sz[0],sz[1],extraMeas)
        
        ax1=plt.subplot(151)
        ax=plt.imshow(tonumpy(IMGs[:,:,2]), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('last contrast')
        plt.ion()
        
        ax1=plt.subplot(152)
        ax=plt.imshow(tonumpy(target_image), interpolation='none')
        plt.clim(np.min(np.abs(tonumpy(target_image))),np.max(np.abs(tonumpy(target_image))))
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('target')
        plt.ion()
        
        plt.subplot(153, sharex=ax1, sharey=ax1)
        ax=plt.imshow(tonumpy(cnn_output), interpolation='none')
        plt.clim(np.min(np.abs(tonumpy(target_image))),np.max(np.abs(tonumpy(target_image))))
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('reco')
        plt.ion()
        
        reco_test = torch.sqrt((reco_testset**2).sum(2))
        reco_test = reco_test.permute([1,0])
        cnn_testoutput = NN(reco_test).reshape([sz[0],sz[1]])
        mask = np.flip(real_phantom_resized[:,:,1].transpose(),(0,1)) != 1.e-12
        
        
        plt.subplot(154)
        plt.plot(tonumpy(target_image.flatten()),tonumpy(cnn_output.flatten()),'.', np.flip(real_phantom_resized[:,:,1].transpose(),(0,1)).flatten(),tonumpy(cnn_testoutput.flatten()),'r.')
        plt.xlabel('tgt')
        plt.ylabel('prd')
        plt.plot([0,3],[0,3])       

        
        plt.subplot(155, sharex=ax1, sharey=ax1)
        ax=plt.imshow(tonumpy(cnn_testoutput)*mask, interpolation='none')
        plt.clim(0,2.2)
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('recotestet')
        plt.ion()
        
        
        #spin_db_input[0,:,:,:] = real_phantom_resized
        
        fig.set_size_inches(15, 3)
        plt.show()
    
    return (phi,cnn_output, error)
        
#r # %% ###     OPTIMIZATION start ######################################################
#############################################################################    
#nmb_hidden_neurons_list = [2*NRep,8,1]
#nmb_hidden_neurons_list = [2*NRep,32,32,2]
#CNN = core.nnreco.VoxelwiseNet(spins.sz, nmb_hidden_neurons_list,use_gpu=use_gpu,gpu_device=gpu_dev)
if first_run:
    nmb_hidden_neurons_list = [extraMeas,16,32,16,1]
    NN = core.nnreco.VoxelwiseNet(spins.sz, nmb_hidden_neurons_list,use_gpu=use_gpu,gpu_device=gpu_dev)
    opt = core.opt_helper.OPT_helper(scanner,spins,NN,1)
    first_run =0;
    opt.set_handles(init_variables, phi_FRP_model)
    opt.scanner_opt_params = opt.init_variables()


    opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],1]))
    opt.target_seq_holder=targetSeq
    opt.experiment_description = experiment_description

opt.learning_rate = 10*1e-4

opt.optimzer_type = 'Adam'
opt.opti_mode = 'seqnn'
opt.batch_size = 3
# 
opt.set_opt_param_idx([2]) # ADC, RF, time, grad
opt.custom_learning_rate = [0.01,0.01,0.1,0.1]

opt.train_model(training_iter=10000, do_vis_image=False, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later

# %% # run for real scan
opt.export_to_pulseq(experiment_id, today_datestr, sequence_class)
scanner.send_job_to_real_system(experiment_id,today_datestr,jobtype="lastiter")
scanner.get_signal_from_real_system(experiment_id,today_datestr,jobtype="lastiter")

core.FID_normscan.make_FID(spin_db_input[samp_idx,:,:,:],do_scanner_query = True)

normmeas=torch.from_numpy(np.load("auxutil/normmeas.npy"))
scanner.signal=scanner.signal/normmeas/NVox    
           
reco_sep = scanner.adjoint_separable()
        
reco_all_rep=torch.zeros((extraMeas,reco_sep.shape[1],2))
for j in range(0,extraMeas):
    reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)
            
reco_all_rep = torch.sqrt((reco_all_rep**2).sum(2))
reco_all_rep = reco_all_rep.permute([1,0])

cnn_output_real = NN(reco_all_rep).reshape([sz[0],sz[1]])
     
plt.subplot(121)   
ax=plt.imshow(tonumpy(cnn_output_real), interpolation='none')
plt.clim(0,2)
fig = plt.gcf()
fig.colorbar(ax)        
plt.title('T1 NN reco')
plt.ion()
plt.subplot(122)   

mag_echo3 = magimg(tonumpy(first_scan).reshape([sz[0],sz[1],2]))
magmask = mag_echo3 > np.mean(mag_echo3.ravel())/6.3
magmask=magmask.astype(float)
plt.imshow(magmask)
mag_echo1 = magimg(tonumpy(first_scan).reshape([sz[0],sz[1],2]))
mag_echo2 = magimg(tonumpy(second_scan).reshape([sz[0],sz[1],2]))
mag_echo3 = magimg(tonumpy(third_scan).reshape([sz[0],sz[1],2]))

mag_echo13 = np.abs((1-mag_echo1/(mag_echo3+1e-12)))
mag_echo23 = np.abs(1-mag_echo2/(mag_echo3+1e-12))
adc_mask,rf_event,event_time, gradm_event = reparameterize(opt.scanner_opt_params)

dTI13 = torch.abs(event_time[-1,0])
dTI23 =  torch.abs(event_time[-1,:measRepStep+1].sum() )

T1map1= -tonumpy(dTI13)/np.log(mag_echo13 + 1e-12)
T1map2= -tonumpy(dTI23)/np.log(mag_echo23 + 1e-12)
T1map1 = T1map1*magmask
T1map2 = T1map2*magmask

ax=plt.imshow(T1map2)
fig = plt.gcf()
plt.clim(0,2)
fig.colorbar(ax)    
plt.title("T1 conventional exp reco")
plt.ion()

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


            