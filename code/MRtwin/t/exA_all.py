"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""
experiment_id = 'ex01_FID'
sequence_class = "gre_dream"
experiment_description = """
FID or 1 D imaging / spectroscopy
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
sz = np.array([4,4])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                      # number of total repetitions
NRep = 3                                    # number of total repetitions
szread=512
NEvnt = szread + 7                               # number of events F/R/P
NSpins = 25**2                               # number of spin sims in each voxel
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
#real_phantom = scipy.io.loadmat('../data/numerical_brain_cropped.mat')['cropped_brain']
real_phantom = np.zeros((128,128,5), dtype=np.float32); real_phantom[64:80,64:80,:2]=1; real_phantom[64:80,64:80,2]=0.1; real_phantom[64:80,64:80,3]=0

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
real_phantom_resized[:,:,3] += 0 # Tweak dB0
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
R2star = 750.0
omega = np.linspace(0+1e-5,1-1e-5,NSpins) - 0.5+1e-5    # cutoff might bee needed for opt.
omega = np.expand_dims(omega[:],1).repeat(NVox, axis=1)
omega*=0.99 # cutoff large freqs
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
rf_event[3,0,0] = 90*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
rf_event[3,1,0] = 180*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
rf_event[3,1,1] = -90*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
# randomize RF phases
measRepStep = NRep//extraMeas
#for i in range(0,extraMeas):
#    rf_event[3,i*measRepStep:(i+1)*measRepStep,1] = torch.tensor(scanner.phase_cycler[:(measRepStep)]).float()*np.pi/180
rf_event = setdevice(rf_event)

scanner.init_flip_tensor_holder()    
scanner.set_flip_tensor_withB1plus(rf_event)

# rotate ADC according to excitation phase
rfsign = ((rf_event[3,:,0]) < 0).float()
scanner.set_ADC_rot_tensor(-rf_event[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.08*1e-3*np.ones((NEvnt,NRep))).float()
event_time[:,0] =  0.04*1e-3
#event_time[3,:] =  2e-3
#event_time[4,:] =  5.5*1e-3   # for 96
#event_time[-2,:] = 2*1e-3
#event_time[-1,:] = 2.9*1e-3
event_time = setdevice(event_time)


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
    gradm_event[5:-2,meas_indices[j,:] ,0] = torch.ones(int(szread)).view(int(szread),1).repeat([1,meas_indices[j,:].size]) # ADC open, readout, freq encoding
    gradm_event[-2,meas_indices[j,:] ,0] = torch.ones(1)*sz[0]*2  # GRE/FID specific, SPOILER
    gradm_event[-2,meas_indices[j,:] ,1] = -gradm_event[1,meas_indices[j,:] ,1]      # GRE/FID specific, yblip rewinder
    
    gradm_event[4,meas_indices[j,:] ,1] = 0
    gradm_event[-2,meas_indices[j,:] ,1] = -gradm_event[4,meas_indices[j,:] ,1]
    
gradm_event[:] = 0
gradm_event = setdevice(gradm_event)

# end sequence 

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(gradm_event,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes


#############################################################################
## S4: MR simulation forward process ::: #####################################
scanner.init_signal()
scanner.forward(spins, event_time)
scanner.signal=scanner.signal/NVox
    
ax1=plt.subplot(131)
ax=plt.plot(tonumpy(scanner.signal[0,:,:,0,0]).transpose().ravel())
plt.plot(tonumpy(scanner.signal[0,:,:,1,0]).transpose().ravel())
plt.title('signal')
plt.ion()
fig=plt.gcf()
fig.set_size_inches(96, 7)
plt.show()

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


# scanner.reco = scanner.do_ifft_reco()
target = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,target)
if True: # check sanity: is target what you expect and is sequence what you expect
    plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')

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
        
if False:
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
                        
    
            