"""
Created on Tue Jan 29 14:38:26 2019

@author: mzaiss

experiment desciption:

2D imaging: learn to predict T2 from GRE-optimized variations

"""

experiment_id = 'q14_tgtT1_tskT1invrec_supervised_seqNNvivo_short_TI.py'
sequence_class = "gre_dream"
experiment_description = """
opt pitcher try different fwd procs
"""
first_run = 1

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

print(experiment_id)

double_precision = False
use_supermem = True
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
kill_transverse = False
sz = np.array([32,32])                                           # image size
extraRep = 10
NRep = extraRep*sz[1]                                   # number of repetitions
T = sz[0] + 7                                        # number of events F/R/P
NSpins = 20**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*1e-3                               # additive Gaussian noise std
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*sz[1]


#############################################################################
## Init spin system ::: #####################################
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


if first_run:
    csz = 12
    nmb_samples = 10
    spin_db_input = np.zeros((nmb_samples, sz[0], sz[1], 5), dtype=np.float32)
    
    for i in range(nmb_samples):
        rvx = np.int(np.floor(np.random.rand() * (sz[0] - csz)))
        rvy = np.int(np.floor(np.random.rand() * (sz[0] - csz)))
        
        b0 = (np.random.rand() - 0.5) * 30                            # -60..60 Hz
        pdbase = np.random.rand()
        for j in range(rvx,rvx+csz):
            for k in range(rvy,rvy+csz):
                pd = pdbase + np.random.rand()*0.5
                t2 = 0.04 + 3*np.random.rand()**1.5
                t1 =0.3+np.abs( 1.3 + 1.5*np.random.randn())
                b0 += 2*(np.random.randn())     
                
                  
                spin_db_input[i,j,k,0] = pd
                spin_db_input[i,j,k,1] = t1
                spin_db_input[i,j,k,2] = t2
                spin_db_input[i,j,k,3] = b0
            
    VV=spin_db_input[:,:,:,3].flatten(); VV=VV[VV>1e-6]; plt.hist(VV,50)         
#spin_db_input[0,:,:,:] = real_phantom_resized
#VV=spin_db_input[0,:,:,3].flatten(); VV=VV[VV>1e-6]; plt.hist(VV*123,50,alpha = 0.5) 
#VV=spin_db_input[:,:,:,3].flatten();  VV=VV[np.abs(VV)>1e-6]; plt.hist(VV,50)   
#VV=spin_db_input[0,:,:,3].flatten();  VV=VV[np.abs(VV)>1e-6]; plt.hist(VV*123,50,alpha = 0.5) 

            
tmp = spin_db_input[:,:,:,1:3]
tmp[tmp < cutoff] = cutoff
spin_db_input[:,:,:,1:3] = tmp

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


spins.set_system(real_phantom_resized)
core.FID_normscan.make_FID(real_phantom_resized,do_scanner_query=do_scanner_query)

#sigma = 0.8
#for i in range(nmb_samples):
#    for j in range(3):
#        spin_db_input[i,:,:,j] = scipy.ndimage.filters.gaussian_filter(spin_db_input[i,:,:,j], sigma)

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
adc_mask[:5]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

# RF events: flips and phases
flips = torch.zeros((T,NRep,2), dtype=torch.float32)
flips[3,:,0] = 5*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
# randomize RF phases
measRepStep = NRep//extraRep
for i in range(0,extraRep):
    flips[0,i*measRepStep,0] = 180*np.pi/180 
    flips[3,i*measRepStep:(i+1)*measRepStep,1] = torch.tensor(scanner.phase_cycler[:(measRepStep)]).float()*np.pi/180
flips = setdevice(flips)

scanner.init_flip_tensor_holder()

B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([scanner.NCoils, scanner.NVox]))
B1plus[B1plus == 0] = 1    # set b1+ to one, where we dont have phantom measurements
B1plus[:] = 1
scanner.B1plus = setdevice(B1plus)    
scanner.set_flip_tensor_withB1plus(flips)

# rotate ADC according to excitation phase
rfsign = ((flips[3,:,0]) < 0).float()
scanner.set_ADC_rot_tensor(-flips[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.08*1e-3*np.ones((scanner.T,scanner.NRep))).float()
#TI = torch.tensor(np.arange(0,0.5,0.05))
TI = torch.tensor([0.1,0.2,0.3,0.5,0.75,1,1.5,3,5,8])

# all measurements event times
event_time[3,:] =  2e-3
event_time[4,:] =  5.5*1e-3   # for 96
event_time[-2,:] = 2*1e-3
event_time[-1,:] = 2.9*1e-3

# first action
for i in range(0,extraRep):
    event_time[0,i*measRepStep] = 1*1e-3
    event_time[2,i*measRepStep] = TI[i]
    if i>0:
        event_time[-1,i*measRepStep-1] = 12       #delay after readout before next inversion ( only after the first event played out)
event_time = setdevice(event_time)

TR=torch.sum(event_time[:,1])
TE=torch.sum(event_time[:11,1])

# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32)

meas_indices=np.zeros((extraRep,measRepStep))
for i in range(0,extraRep):
    meas_indices[i,:] = np.arange(i*measRepStep,(i+1)*measRepStep)

for j in range(0,extraRep):
    # second action after inversion pulse (chrusher)
    grad_moms[1,j*measRepStep] = 1e-2

    #  measurement
    grad_moms[4,meas_indices[j,:] ,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
    grad_moms[4,meas_indices[j,:] ,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(meas_indices[j,:].size))  # phase encoding blip in second event block
    grad_moms[5:-2,meas_indices[j,:] ,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,meas_indices[j,:].size]) # ADC open, readout, freq encoding
    grad_moms[-2,meas_indices[j,:] ,0] = torch.ones(1)*sz[0]*2  # GRE/FID specific, SPOILER
    grad_moms[-2,meas_indices[j,:] ,1] = -grad_moms[1,meas_indices[j,:] ,1]      # GRE/FID specific, yblip rewinder
    
    grad_moms[4,meas_indices[j,:] ,1] = 0
    for i in range(1,int(sz[1]/2)+1):
        grad_moms[4,j*measRepStep+i*2-1,1] = (-i)
        if i < sz[1]/2:
            grad_moms[4,j*measRepStep+i*2,1] = i
    grad_moms[-2,meas_indices[j,:] ,1] = -grad_moms[4,meas_indices[j,:] ,1]
    
#grad_moms[:] = 0
grad_moms = setdevice(grad_moms)

# end sequence 

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
#scanner.forward_sparse_fast_supermem(spins, event_time)
#scanner.forward_sparse_fast(spins, event_time)
scanner.forward_sparse_fast(spins, event_time,kill_transverse=kill_transverse)
#scanner.forward_mem(spins, event_time)
#scanner.forward(spins, event_time)
#scanner.init_signal()
#scanner.signal[:,:,0,:,:] = 0

normsim=torch.from_numpy(np.load("auxutil/normsim.npy"))
print(normsim)
scanner.signal=scanner.signal/normsim/NVox
    
reco_sep = scanner.adjoint_separable()

reco_all_rep=torch.zeros((extraRep,reco_sep.shape[1],2))
for j in range(0,extraRep):
    reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)

scale = torch.max(tomag_torch(reco_all_rep)) #last point for normalization
reco_testset = reco_all_rep / scale

first_scan_kspace = tonumpy(scanner.signal[0,5:-2,meas_indices[0,:],:2,0])
second_scan_kspace = tonumpy(scanner.signal[0,5:-2,meas_indices[1,:],:2,0])
third_scan_kspace = tonumpy(scanner.signal[0,5:-2,meas_indices[2,:],:2,0])

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
        ax=plt.imshow(magimg(tonumpy(reco_all_rep[0,:,:]).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('first scan')
        plt.ion()
        
        plt.subplot(232, sharex=ax1, sharey=ax1)
        ax=plt.imshow(magimg(tonumpy(reco_all_rep[1,:,:]).reshape([sz[0],sz[1],2])), interpolation='none')
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
        
        # print results
        ax1=plt.subplot(233)
        ax=plt.imshow(magimg(tonumpy(reco_all_rep[2,:,:]).reshape([sz[0],sz[1],2])), interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('third scan')
        plt.ion()        
        
        plt.subplot(236, sharex=ax1, sharey=ax1)
        ax=plt.imshow(third_scan_kspace_mag, interpolation='none')
        fig = plt.gcf()
        fig.colorbar(ax)        
        plt.title('third scan kspace')
        plt.ion()    
        
        fig.set_size_inches(18, 7)
        
        plt.show()
        
if True:
    targetSeq.export_to_pulseq(experiment_id,today_datestr,sequence_class,plot_seq=True)
    
    if do_scanner_query:
        scanner.send_job_to_real_system(experiment_id,today_datestr)
        scanner.get_signal_from_real_system(experiment_id,today_datestr)
        
#        normmeas=torch.from_numpy(np.load("auxutil/normmeas.npy"))
#        scanner.signal=scanner.signal/normmeas/NVox
        
        reco_sep = scanner.adjoint_separable()
        
        reco_all_rep=torch.zeros((extraRep,reco_sep.shape[1],2))
        for j in range(0,extraRep):
            reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)
        
        reco_test = torch.sqrt((reco_all_rep**2).sum(2))
        S = reco_test.reshape(10,32,32)
        S = tonumpy(S)
        
        plt.subplot(341)
        plt.imshow(S[0,:,:])
        plt.colorbar()
        plt.subplot(342)
        plt.imshow(S[1,:,:])
        plt.colorbar()
        plt.subplot(343)
        plt.imshow(S[2,:,:])
        plt.colorbar()
        plt.subplot(344)
        plt.imshow(S[3,:,:])
        plt.colorbar()
        plt.subplot(345)
        plt.imshow(S[4,:,:])
        plt.colorbar()
        plt.subplot(346)
        plt.imshow(S[5,:,:])
        plt.colorbar()
        plt.subplot(347)
        plt.imshow(S[6,:,:])
        plt.colorbar()
        plt.subplot(348)
        plt.imshow(S[7,:,:])
        plt.colorbar()
        plt.subplot(349)
        plt.imshow(S[8,:,:])
        plt.colorbar()
        plt.subplot(3410)
        plt.imshow(S[9,:,:])
        plt.colorbar()
        
        reco_testset_complex = tonumpy(reco_all_rep[:,:,0])+1j*tonumpy(reco_all_rep[:,:,1])
        phase = np.angle(reco_testset_complex)
        
        plt.subplot(341)
        plt.imshow(phase[0,:].reshape(32,32))
        plt.colorbar()
        plt.subplot(342)
        plt.imshow(phase[1,:].reshape(32,32))
        plt.colorbar()
        plt.subplot(343)
        plt.imshow(phase[2,:].reshape(32,32))
        plt.colorbar()
        plt.subplot(344)
        plt.imshow(phase[3,:].reshape(32,32))
        plt.colorbar()
        plt.subplot(345)
        plt.imshow(phase[4,:].reshape(32,32))
        plt.colorbar()
        plt.subplot(346)
        plt.imshow(phase[5,:].reshape(32,32))
        plt.colorbar()
        plt.subplot(347)
        plt.imshow(phase[6,:].reshape(32,32))
        plt.colorbar()
        plt.subplot(348)
        plt.imshow(phase[7,:].reshape(32,32))
        plt.colorbar()
        plt.subplot(349)
        plt.imshow(phase[8,:].reshape(32,32))
        plt.colorbar()
        plt.subplot(3410)
        plt.imshow(phase[9,:].reshape(32,32))
        plt.colorbar()    
        
        from scipy.optimize import curve_fit
                        
        def signal_T1 (TI, S_0, T1):
            return np.abs(S_0 * (1 - 2*np.exp(-TI/T1)))
        
        def signal2_T1 (TI, S_0, T1, Z_i):
            return np.abs(S_0 - (S_0 - Z_i)*np.exp(-TI/T1))
        
        def signal3_T1 (TI, S_0, T1):
            return np.abs(S_0 * (1 - 2*np.exp(-TI/T1)+np.exp(-0.0152 / T1)))
        
        def quantify_T1 (TI, S, p0):
            popt, pcov = curve_fit(signal2_T1, TI, S, p0 = p0, maxfev=1000000)#, bounds=([S[-1]/2,0.5,-S[-1]*2],[S[-1]*2,5,S[-1]]))
            return popt

        xdata = np.array([0.1,0.2,0.3,0.5,0.75,1,1.5,3,5,8,20])
        xdata = tonumpy(TI)
        T1_map = np.zeros((sz[0],sz[1]))
        for l in range(sz[0]):
            for m in range(sz[1]):
                try:
                    popt = quantify_T1(xdata[0:], S[0:,l,m], p0=[S[-1,l,m],1,-S[-1,l,m]])
                    popt = quantify_T1(xdata[0:], S[0:,l,m], p0=None)
                    T1_map[l,m] = popt[1]
                except:
                    T1_map[l,m] = 0
        
        plt.imshow(T1_map)
        plt.clim(0,5)
        plt.colorbar()
        
        x,y = 18,21 #(R,C)
        start_idx = 0
        points = S[start_idx:,x,y]
        reduced_xdata = xdata[start_idx:]
        popt = quantify_T1(reduced_xdata, points, p0=[S[-1,x,y],1,-S[-1,x,y]])#, p0 = [np.flip(real_phantom_resized[:,:,1].transpose(),(0,1))[x,y],0.1,-0.1])
        popt = quantify_T1(reduced_xdata, points, p0=None)
        plt.plot(reduced_xdata, points, 'b-', label='data, S_0_true = %5.3f, T1_true=%5.3f' % tuple([np.flip(real_phantom_resized[:,:,0].transpose(),(0,1))[x,y],np.flip(real_phantom_resized[:,:,1].transpose(),(0,1))[x,y]]), marker='.')
        plt.plot(reduced_xdata, signal2_T1(reduced_xdata, *popt), 'r-', label='fit: S_0=%5.3f, T1=%5.3f, Z_i=%5.3f' % tuple(popt))
        plt.legend()
        plt.show()
        
        plt.plot(reduced_xdata, points)
        import plotly.express as px
                        
    

    #
    #nn_input_muster = torch.stack((magimg_torch(first_scan),magimg_torch(second_scan)),1)
    
    # do real meas


    
# %% ###    stuff for supervised  ######################################################

with torch.no_grad():
    # target = T21
    target = setdevice(torch.from_numpy(real_phantom_resized[:,:,1]).float())
    targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target)
    
    # Prepare target db: iterate over all samples in the DB
    target_db = setdevice(torch.zeros((nmb_samples,NVox,1)).float())
    reco_all_rep_premeas = setdevice(torch.zeros((nmb_samples,extraRep,NVox,2)).float())
    
    
    for i in range(nmb_samples):
        print(i)
        tgt = torch.from_numpy(spin_db_input[i,:,:,1:2].reshape([NVox,1]))
        target_db[i,:,:] = tgt.reshape([sz[0],sz[1],1]).reshape([NVox,1])
          
        
        spins.set_system(spin_db_input[i,:,:,:])
        if first_run==0: #opt.opti_mode = 'seqnn'
            
            if True:   # recreate  from learned seqnn
                adc_mask,flips,event_time, grad_moms = reparameterize(opt.scanner_opt_params)
                scanner.set_adc_mask(adc_mask=setdevice(adc_mask))
                scanner.init_flip_tensor_holder()      
                scanner.set_flip_tensor_withB1plus(flips)
                # rotate ADC according to excitation phase
                rfsign = ((flips[3,:,0]) < 0).float()
                scanner.set_ADC_rot_tensor(-flips[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific
                scanner.init_gradient_tensor_holder()
                scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

    #            scanner.init_gradient_tensor_holder()       
            
            scanner.forward_sparse_fast(spins, event_time,kill_transverse=kill_transverse)
        
            core.FID_normscan.make_FID(spin_db_input[i,:,:,:])
            normsim=torch.from_numpy(np.load("auxutil/normsim.npy"))
            print(normsim)
            scanner.signal=scanner.signal/normsim/NVox    
               
            reco_sep = scanner.adjoint_separable()
            
            reco_all_rep=torch.zeros((extraRep,reco_sep.shape[1],2))
            for j in range(0,extraRep):
                reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)
                
            scale = torch.max(tomag_torch(reco_all_rep)) #last point for normalization                                      
            reco_all_rep_premeas[i,:] = reco_all_rep / scale
            
            
#            if i==0:
#                reco_testset = reco_all_rep  
    
    #reco_all_rep_premeas=reco_all_rep_premeas.reshape([nmb_samples,3,sz[0],sz[1],2]).permute([0,1,3,2,4]).flip([2,3]).reshape([nmb_samples,3,NVox,2])
    target_db=target_db.reshape([nmb_samples,sz[0],sz[1],1]).permute([0,2,1,3]).flip([1,2]).reshape([nmb_samples,NVox,1])
    
    
    samp_idx = np.random.choice(nmb_samples,1)[0]
#    samp_idx=0
    
    reco_all_rep = reco_all_rep_premeas[samp_idx,:extraRep,:]
    
    reco_all_rep = torch.sqrt((reco_all_rep**2).sum(2))
    #    reco_all_rep = reco_all_rep.t()
    reco_all_rep = reco_all_rep.permute([1,0])
     
    target_image = target_db[samp_idx,:,:].reshape([sz[0],sz[1]])
    
    IMG=reco_all_rep =reco_all_rep.reshape([sz[0],sz[1],extraRep,1]) 
    ax1=plt.subplot(121)
    plt.imshow(tonumpy(IMG[:,:,2,0]), interpolation='none')
    ax1=plt.subplot(122)
    plt.imshow(tonumpy(target_image), interpolation='none')

# %% ###    
with torch.no_grad():    
    # Prepare target db: iterate over all samples in the DB
          
    spins.set_system(real_phantom_resized)
        
    adc_mask,flips,event_time, grad_moms = reparameterize(opt.scanner_opt_params)
    scanner.set_adc_mask(adc_mask=setdevice(adc_mask))
    scanner.init_flip_tensor_holder()      
    scanner.set_flip_tensor_withB1plus(flips)
    # rotate ADC according to excitation phase
    rfsign = ((flips[3,:,0]) < 0).float()
    scanner.set_ADC_rot_tensor(-flips[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific
    scanner.init_gradient_tensor_holder()
    scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes
        
    scanner.forward_sparse_fast(spins, event_time,kill_transverse=kill_transverse)           
    reco_sep = scanner.adjoint_separable()
    
    reco_testset=torch.zeros((extraRep,reco_sep.shape[1],2))
    for j in range(0,extraRep):
        reco_testset[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)
        
    scale = torch.max(tomag_torch(reco_testset)) #last point for normalization                                      
    reco_testset = reco_testset / scale
    reco_test = tomag_torch(reco_testset)
    
    IMG=reco_test =reco_test.reshape(10,32,32) 
    ax1=plt.subplot(121)
    plt.imshow(tonumpy(IMG[2,:,:]), interpolation='none')
    ax1=plt.subplot(122)
    plt.imshow(np.flip(real_phantom_resized[:,:,1].transpose(),(0,1)))

# %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################    
count_idx=0

def init_variables():
    adc_mask = targetSeq.adc_mask.clone()
    #adc_mask.requires_grad = True     
    
    flips = targetSeq.flips.clone()
    #flips[0,:,:]=flips[0,:,:]*0
    flips = setdevice(flips)
    
    flip_mask = torch.zeros((scanner.T, scanner.NRep, 2)).float()     
    flip_mask[3,:,:] = 1
    flip_mask = setdevice(flip_mask)
    flips.zero_grad_mask = flip_mask
      
    event_time = targetSeq.event_time.clone()
    event_time = setdevice(event_time)
       
    event_time_mask = torch.zeros((scanner.T, scanner.NRep)).float()   
    for j in range(0,extraRep):
        event_time_mask[2,j*measRepStep] = 1         # optimize TI
    
    event_time_mask = setdevice(event_time_mask)
    event_time.zero_grad_mask = event_time_mask
        
    grad_moms = targetSeq.grad_moms.clone()

    grad_moms_mask = torch.zeros((scanner.T, scanner.NRep, 2)).float()        
    grad_moms_mask = setdevice(grad_moms_mask)
    grad_moms.zero_grad_mask = grad_moms_mask
    
    return [adc_mask, flips, event_time, grad_moms]


def reparameterize(opt_params):
    adc_mask,flips,event_time,grad_moms = opt_params
    return [adc_mask, flips, event_time, grad_moms]
    
    
def phi_FRP_model(opt_params,aux_params):
    adc_mask,flips,event_time,grad_moms = opt_params
             
    samp_idx = np.random.choice(nmb_samples,1)[0] 
    
    if opt.opti_mode == 'nn': #only NN from premeas
        plotdiv =150
        reco_all_rep = reco_all_rep_premeas[samp_idx,:extraRep,:]
    else:
        plotdiv =1
        scanner.set_adc_mask(adc_mask)
        scanner.init_flip_tensor_holder()      
        scanner.set_flip_tensor_withB1plus(flips)
        # rotate ADC according to excitation phase
        rfsign = ((flips[3,:,0]) < 0).float()
        scanner.set_ADC_rot_tensor(-flips[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific
        scanner.init_gradient_tensor_holder()
        scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes
#        
        spins.set_system(spin_db_input[samp_idx,:,:,:])    
        scanner.forward_sparse_fast(spins, event_time,kill_transverse=kill_transverse)        
        #core.FID_normscan.make_FID(spin_db_input[samp_idx,:,:,:])
        #normsim=torch.from_numpy(np.load("auxutil/normsim.npy"))
        #scanner.signal=scanner.signal/normsim/NVox    
           
        reco_sep = scanner.adjoint_separable()
        
        reco_all_rep=torch.zeros((extraRep,reco_sep.shape[1],2))
        for j in range(0,extraRep):
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
    loss_sar = torch.sum(flips[:,:,0]**2)
    
    lbd_t = 1e-4         # switch on of time cost
    loss_t = lbd_t*torch.abs(torch.sum(event_time))
    
    loss = loss_image + lbd_sar*loss_sar + loss_t
        
    phi = loss
  
    ereco = tonumpy(cnn_output*non_zero_voxel_mask).reshape([sz[0],sz[1]])
    error = e(tonumpy(target_image*non_zero_voxel_mask).ravel(),ereco.ravel())     
    
    global count_idx
    count_idx=count_idx+1
    
    if np.mod(count_idx,plotdiv)==0:
        print("loss_image: {} ".format(loss_image),"loss_t: {} ".format(loss_t), "TIs {}".format((event_time[2,np.arange(0,extraRep*measRepStep,measRepStep)])))
        # print results
        IMGs=reco_all_rep.reshape(sz[0],sz[1],extraRep)
        
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
    nmb_hidden_neurons_list = [extraRep,16,32,16,1]
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
        
reco_all_rep=torch.zeros((extraRep,reco_sep.shape[1],2))
for j in range(0,extraRep):
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
adc_mask,flips,event_time, grad_moms = reparameterize(opt.scanner_opt_params)

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
for ni in range(len(opt.param_reco_history)):
    opt.param_reco_history[ni]['reco_image'] = np.dstack((opt.param_reco_history[ni]['reco_image'],np.zeros((32,32))))


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


            