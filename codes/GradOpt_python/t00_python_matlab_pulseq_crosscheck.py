#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
"""

experiment_id = 'e08_GRE_python_scanner_loop'
sequence_class = "GRE"
experiment_description = """
tgt FLASHspoiled_relax20ms, with spoilers and random phase cycling
task find all grads except read ADC grads
opt: SARloss, kloss, 

this is the same as e05_tgtGRE_tskGREnogspoil.py, but now with more automatic restarting
and high initial learning rate
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

sys.path.append("../scannerloop_libs")
from pypulseq.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import makeadc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc import make_sinc_pulse
from pypulseq.make_trap import make_trapezoid
from pypulseq.make_block import make_block_pulse
from pypulseq.opts import Opts

# for trap and sinc
from pypulseq.holder import Holder

do_scanner_query = True
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
    
def stop():
    sys.tracebacklimit = 0
    class ExecutionControl(Exception): pass
    raise ExecutionControl('stopped by user')
    sys.tracebacklimit = 1000

# define setup
sz = np.array([24,24])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 20**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements

noise_std = 0*1e0                               # additive Gaussian noise std

NVox = sz[0]*sz[1]

#############################################################################
## Init spin system ::: #####################################

# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev)

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
grad_moms[1,:,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))  # phase encoding blip in second event block
grad_moms[2:-2,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep]) # ADC open, readout, freq encoding
grad_moms[-2,:,0] = torch.ones(1)*sz[0]*2  # GRE/FID specific, SPOILER
grad_moms[-2,:,1] = -grad_moms[1,:,1]      # GRE/FID specific, yblip rewinder
grad_moms = setdevice(grad_moms)

#     centric ordering
#grad_moms[1,:,1] = 0
#for i in range(1,int(sz[1]/2)+1):
#    grad_moms[1,i*2-1,1] = (-i)
#    if i < sz[1]/2:
#        grad_moms[1,i*2,1] = i
#grad_moms[-2,:,1] = -grad_moms[1,:,1]     # backblip

# end sequence 

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
# forward/adjoint pass
scanner.forward_fast(spins, event_time)
scanner.adjoint(spins)

# try to fit this
# scanner.reco = scanner.do_ifft_reco()
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
    
#############################################################################
## PREPARE sequence ::: #####################################


hgfhgfh

# gradient tranform
grad_moms_numpy = np.array(scanner_dict['grad_moms'], dtype=np.float32)

seq_fn = os.path.join(seq_dir,experiment_id+".seq")

resolution = sz
FOV = 220e-3
maxSlew = 140

# gradients
Nx = sz[0]
Ny = sz[1]

kwargs_for_opts = {"rf_ring_down_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": maxSlew, "slew_unit": "T/m/s"}
system = Opts(kwargs_for_opts)
seq = Sequence(system)

# init sequence and system
deltak = 1 / FOV

# read gradient

T = grad_moms.shape[0]
NRep = grad_moms.shape[1]
flips = np.array(scanner_dict['flips'], dtype=np.float32)
event_times = np.array(scanner_dict['event_times'], dtype=np.float32)
grad_moms *= deltak

# put blocks together
for rep in range(NRep):
#for rep in range(1):
    
    # first action
    idx_T = 0
    if np.abs(flips[idx_T,rep,0]) > 1e-8:
        use = "excitation"
        
        # alternatively slice selective:
        slice_thickness = 5e-3     # slice
        
        kwargs_for_sinc = {"flip_angle": flips[idx_T,rep,0], "system": system, "duration": 0.6*1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4}
        rf, gz, gzr = make_sinc_pulse(kwargs_for_sinc, 3)
        
        seq.add_block(rf, gz)
        seq.add_block(gzr)
        
    delay = make_delay(event_times[idx_T,rep])
    seq.add_block(delay)
    
    # second  (rewinder)
    idx_T = 1
    
    # calculated here, update in next event
    gradmom_rewinder = np.squeeze(grad_moms[idx_T,rep,:])
    eventtime_rewinder = np.squeeze(event_times[idx_T,rep])
    
    # line acquisition T(3:end-1)
    idx_T = np.arange(2, grad_moms.shape[0] - 2) # T(2)
    dur = np.sum(event_times[idx_T,rep])
    
    kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": np.sum(grad_moms[idx_T,rep,0],0), "flat_time": dur}
    gx = make_trapezoid(kwargs_for_gx)    
    
    kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": np.sum(grad_moms[idx_T,rep,1],0), "flat_time": dur}
    gy = make_trapezoid(kwargs_for_gy)
    
    kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": gx.rise_time, "phase_offset": rf.phase_offset}
    adc = makeadc(kwargs_for_adc)    
    
    #update rewinder for gxgy ramp times, from second event
    kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
    gx_pre = make_trapezoid(kwargs_for_gxpre)
    
    kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1]-gy.amplitude*gy.rise_time/2, "duration": eventtime_rewinder}
    gy_pre = make_trapezoid(kwargs_for_gypre)

    seq.add_block(gx_pre, gy_pre)
    seq.add_block(gx,gy,adc)
    
    # second last extra event  T(end)  # adjusted also for fallramps of ADC
    idx_T = grad_moms.shape[0] - 2     # T(2)
    
    kwargs_for_gxpost = {"channel": 'x', "system": system, "area": grad_moms[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_times[idx_T,rep]}
    gx_post = make_trapezoid(kwargs_for_gxpost)  
    
    kwargs_for_gypost = {"channel": 'y', "system": system, "area": grad_moms[idx_T,rep,1]-gy.amplitude*gy.fall_time/2, "duration": event_times[idx_T,rep]}
    gy_post = make_trapezoid(kwargs_for_gypost)  
    
    seq.add_block(gx_post, gy_post)
    
    #  last extra event  T(end)
    idx_T = grad_moms.shape[0] - 1 # T(2)
    
    delay = make_delay(event_times[idx_T,rep])
    seq.add_block(delay)

seq.plot()
seq.write(seq_fn)

# append version and definitions
with open(seq_fn, 'r') as fin:
    lines = fin.read().splitlines(True)
    
updated_lines = []
updated_lines.append("# Pulseq sequence file\n")
updated_lines.append("# Created by MRIzero/IMR/GPI pulseq converter\n")
updated_lines.append("\n")
updated_lines.append("[VERSION]\n")
updated_lines.append("major 1\n")
updated_lines.append("minor 2\n")   
updated_lines.append("revision 1\n")  
updated_lines.append("\n")    
updated_lines.append("[DEFINITIONS]\n")
updated_lines.append("FOV "+str(round(FOV*1e3))+" "+str(round(FOV*1e3))+" "+str(round(slice_thickness*1e3))+" \n")   
updated_lines.append("\n")    

updated_lines.extend(lines[3:])

with open(seq_fn, 'w') as fout:
    fout.writelines(updated_lines)  
      

    
    