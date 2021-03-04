"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""
experiment_id = 'solE06_radialssfp'
sequence_class = "super"
experiment_description = """
2 D imaging
"""
excercise = """
The current sequence is very long. 
A10.1. calculate the total scan time of the sequence
A10.1. lower the recovery time after each repetition to event_time[-1,:] =  0.1 . What do you observe?
A10.2. lower the flip angle to 5 degree.
A10.3. find a way to get rid of transverse magnetization from the previous rep using a gradient. (spoiler or crusher gradient)
A10.4. remove the spoiler gradient. find a way to get rid of transverse magnetization from the previous rep using the rf phase
"""
#%%
#matplotlib.pyplot.close(fig=None)
#%%
import os, sys
import numpy as np
import scipy
import scipy.io
from  scipy import ndimage
import scipy.interpolate
import torch
import cv2
import matplotlib.pyplot as plt
from torch import optim
import core.spins
import core.scanner
import core.nnreco
import core.target_seq_holder
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

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

def phaseimg(x):
    return np.angle(1j*x[:,:,1]+x[:,:,0])

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
sz = np.array([24,24])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                      # number of total repetitions
szread=sz[1]
NEvnt = szread + 5 + 2                          # number of events F/R/P
NSpins = 16**2                              # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*1e-3                          # additive Gaussian noise std
kill_transverse = True                      # kills transverse when above 1.5 k.-spaces
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*szread

#############################################################################
## S1: Init spin system and phantom::: #####################################
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)

# either (i) load phantom (third dimension: PD, T1 T2 dB0 rB1)
phantom = spins.get_phantom(sz[0],sz[1],type='object1')  # type='object1' or 'brain1'

# or (ii) set phantom  manually to single pixel phantom
#phantom = np.zeros((sz[0],sz[1],5), dtype=np.float32); 
#phantom[1,1,:]=np.array([1, 1, 0.1, 0, 1]) # third dimension: PD, T1 T2 dB0 rB1

# adjust phantom
phantom[:,:,1] *= 1 # Tweak T1
phantom[:,:,2] *= 1 # Tweak T2
phantom[:,:,3] += 0 # Tweak dB0
phantom[:,:,4] *= 1 # Tweak rB1

if 1: # switch on for plot
    plt.figure("""phantom"""); plt.clf();  param=['PD','T1 [s]','T2 [s]','dB0 [Hz]','rB1 [rel.]']
    for i in range(5):
        plt.subplot(151+i), plt.title(param[i])
        ax=plt.imshow(phantom[:,:,i], interpolation='none')
        fig = plt.gcf(); fig.colorbar(ax) 
    fig.set_size_inches(18, 3); plt.show()

spins.set_system(phantom,R2dash=30.0)  # set phantom variables with overall constant R2' = 1/T2'  (R2*=R2+R2')

## end of S1: Init spin system and phantom ::: #####################################


#############################################################################
## S2: Init scanner system ::: #####################################
scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,NEvnt,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)
#scanner.set_B1plus(phantom[:,:,4])  # use as defined in phantom
scanner.set_B1plus(1)               # overwrite with homogeneous excitation

#############################################################################
## S3: MR sequence definition ::: #####################################
# begin sequence definition
# allow for extra events (pulses, relaxation and spoiling) in the first five and last two events (after last readout event)
adc_mask = torch.from_numpy(np.ones((NEvnt,1))).float()
adc_mask[:5]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

# RF events: rf_event and phases
rf_event = torch.zeros((NEvnt,NRep,4), dtype=torch.float32)
rf_event[0,0,0] = 180*np.pi/180  # 90deg excitation now for every rep
rf_event[2,0,0] = 10*np.pi/180  # 90deg excitation now for every rep
rf_event[2,0,1] = 180*np.pi/180  # 90deg excitation now for every rep
rf_event[3,:,0] = 20*np.pi/180  # 90deg excitation now for every rep
rf_event[3,:,3] = 1

alternate= torch.tensor([0,1])
rf_event[3,:,1]=np.pi*alternate.repeat(NRep//2)

rf_event = setdevice(rf_event)
scanner.init_flip_tensor_holder()    
scanner.set_flip_tensor_withB1plus(rf_event)
# rotate ADC according to excitation phase
rfsign = ((rf_event[3,:,0]) < 0).float()

scanner.set_ADC_rot_tensor(-rf_event[3,:,1]+ np.pi/2 + np.pi*rfsign) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.08*1e-3*np.ones((NEvnt,NRep))).float()
event_time[0,0] =  1e-3
event_time[3,:] =  2e-3
event_time[4,:] =  0.3*1e-3
event_time[-2,:] =  0.3*1e-3
event_time[1,0] =  3
event_time[2,0] =  0.002*0.5  
#event_time[-1,:] =  0.001
event_time = setdevice(event_time)
TA = tonumpy(torch.sum(event_time))
# gradient-driver precession
# Cartesian encoding
gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
gradm_event[4,:,1] = -0.5*szread
gradm_event[5:-2,:,1] = 1
gradm_event[-2,:,1] = -0.5*szread # readback
gradm_event[4,:,0] = torch.arange(0,NRep,1)-NRep/2  #phaseblip
gradm_event[-2,:,0] = -gradm_event[4,:,0]            #phasebackblip

gradm_event = setdevice(gradm_event)

if True:
    gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32) 
    gradm_event[4,:,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
    #grad_moms[1,:,1] = 0*torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(NRep))  # phase encoding in second event block
    gradm_event[5:-2,:,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,NRep]) # ADC open, readout, freq encoding
    
    for rep in range(NRep):
        alpha = torch.tensor(rep * (1.0/(NRep)) * np.pi)
        rotomat = torch.zeros((2,2)).float()
        rotomat[0,0] = torch.cos(alpha)
        rotomat[0,1] = -torch.sin(alpha)
        rotomat[1,0] = torch.sin(alpha)
        rotomat[1,1] = torch.cos(alpha)
        
        # rotate grid
        gradm_event[4,rep,:] = (torch.matmul(rotomat,gradm_event[4,rep,:].unsqueeze(1))).squeeze()
        gradm_event[5:-2,rep,:] = (torch.matmul(rotomat.unsqueeze(0),gradm_event[5:-2,rep,:].unsqueeze(2))).squeeze()
    
    gradm_event[-2,:,:] = gradm_event[4,:,:]      # GRE/FID specific, SPOILER
    gradm_event = setdevice(gradm_event)



scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor_super(gradm_event,rf_event)  # refocusing=False for GRE/FID, adjust for higher echoes
## end S3: MR sequence definition ::: #####################################



#############################################################################
## S4: MR simulation forward process ::: #####################################
scanner.init_signal()
#%%
targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,scanner.signal)
targetSeq.export_to_pulseq(experiment_id,today_datestr,sequence_class,plot_seq=True,single_folder=True)

scanner.get_signal_from_real_system(experiment_id,today_datestr,single_folder=True)
       
        
#%%
#scanner.forward_fast(spins, event_time)

#targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,scanner.signal)
targetSeq.print_seq_pic(True,plotsize=[12,9])
  
#%% ############################################################################
## S5: MR reconstruction of signal ::: #####################################

spectrum = tonumpy(scanner.signal[0,adc_mask.flatten()!=0,:,:2,0].clone()) 
spectrum = spectrum[:,:,0]+spectrum[:,:,1]*1j # get all ADC signals as complex numpy array
spectrum_adc= spectrum
kspace= spectrum
space = np.zeros_like(spectrum)

if 0:#FFT
    spectrum = np.roll(spectrum,szread//2,axis=0)
    spectrum = np.roll(spectrum,NRep//2,axis=1)
    
    space = np.fft.ifft2(spectrum)

if 1: # NUFFT
    adc_idx = np.where(scanner.adc_mask.cpu().numpy())[0]        
    grid = scanner.kspace_loc[adc_idx,:,:]
    NCol=adc_idx.size
    
    X, Y = np.meshgrid(np.linspace(0,NCol-1,NCol) - NCol / 2, np.linspace(0,NRep-1,NRep) - NRep/2)
    grid = np.double(grid.detach().cpu().numpy())
    grid[np.abs(grid) < 1e-5] = 0
    
    plt.subplot(336); plt.plot(grid[:,:,0].ravel('F'),grid[:,:,1].ravel('F'),'rx',markersize=3);  plt.plot(X,Y,'k.',markersize=2);
    plt.show()
    
    spectrum_resampled_x = scipy.interpolate.griddata((grid[:,:,0].ravel(), grid[:,:,1].ravel()), np.real(kspace[:,:]).ravel(), (X, Y), method='cubic')
    spectrum_resampled_y = scipy.interpolate.griddata((grid[:,:,0].ravel(), grid[:,:,1].ravel()), np.imag(kspace[:,:]).ravel(), (X, Y), method='cubic')

    kspace=spectrum_resampled_x+1j*spectrum_resampled_y
    kspace[np.isnan(kspace)] = 0
    
    # fftshift
    kspace = np.roll(kspace,NCol//2,axis=0)
    kspace = np.roll(kspace,NRep//2,axis=1)
            
    space = np.fft.ifft2(kspace)

space = np.roll(space,szread//2-1,axis=0)
space = np.roll(space,NRep//2-1,axis=1)
space = np.flip(space,(0,1))



if 0:
    scanner.adjoint()
    space = scanner.reco.clone().cpu().numpy().reshape([sz[0],sz[1],2])
    space = magimg(space)


if 0: 
    genalpha = 2*1e-2
    
    adc_idx = np.where(scanner.adc_mask.cpu().numpy())[0]
    A = scanner.G_adj[adc_idx,:,:,:2,:2].permute([2,3,0,1,4]).contiguous().view([scanner.NVox*2,adc_idx.size*scanner.NRep*2]).permute([1,0])
    AtA = torch.matmul(A.permute([1,0]),A)
            
    
    scanner.generalized_adjoint(alpha=genalpha,nmb_iter=100)
    space = scanner.reco.clone().cpu().numpy().reshape([sz[0],sz[1],2])
    space = magimg(space)
    
targetSeq.print_seq(plotsize=[12,9])
      
plt.subplot(4,6,19)
plt.imshow(phantom[:,:,0].transpose(), interpolation='none'); plt.xlabel('PD')
plt.subplot(4,6,20)
plt.imshow(phantom[:,:,3].transpose(), interpolation='none'); plt.xlabel('dB0')
plt.subplot(4,6,21)
plt.imshow(np.abs(spectrum_adc).transpose(), interpolation='none'); plt.xlabel('spectrum')
plt.subplot(4,6,22)
plt.imshow(np.abs(kspace).transpose(), interpolation='none'); plt.xlabel('kspace')
plt.subplot(4,6,23)
plt.imshow(np.abs(space).transpose(), interpolation='none'); plt.xlabel('mag_img')
plt.subplot(4,6,24)
plt.imshow(np.angle(space).transpose(), interpolation='none'); plt.xlabel('phase_img')
plt.show()                       
