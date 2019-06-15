"""
Created on Tue Jan 29 14:38:26 2019

@author: mzaiss

experiment desciption:

2D imaging: GRE with spoilers and random phase cycling
GRE90spoiled_relax2s

"""

experiment_id = 'e25_opt_pitcher64_supervised_onlygrad'
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
os.environ['QT_QPA_PLATFORM']='offscreen'

print('e25_opt_pitcher64_supervised_onlygrad')

double_precision = False
use_supermem = True
do_scanner_query = False

use_gpu = 1
gpu_dev = 0

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
sz = np.array([16,16])                                           # image size
NRep = sz[1]                                          # number of repetitions
T = sz[0] + 4                                        # number of events F/R/P
NSpins = 8**2                                # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*1e-3                               # additive Gaussian noise std
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*sz[1]

batch_size = 4


#############################################################################
## Init spin system ::: #####################################
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)

real_phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
real_phantom_resized = np.zeros((sz[0],sz[1],5), dtype=np.float32)
cutoff = 1e-12
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

# initialize the training database, let it be just a bunch squares (<csz> x <csz>) with random PD/T1/T2
# ignore B0 inhomogeneity:-> since non-zero PD regions are just tiny squares, the effect of B0 is just constant phase accum in respective region
csz = 8
nmb_samples = 64
spin_db_input = np.zeros((nmb_samples, sz[0], sz[1], 5), dtype=np.float32)

for i in range(nmb_samples):
    rvx = np.int(np.floor(np.random.rand() * (sz[0] - csz)))
    rvy = np.int(np.floor(np.random.rand() * (sz[0] - csz)))
    
    b0 = (np.random.rand() - 0.5) * 120                            # -60..60 Hz
    
    for j in range(rvx,rvx+csz):
        for k in range(rvy,rvy+csz):
            pd = 0.5 + np.random.rand()
            t2 = 0.3 + np.random.rand()
            t1 = t2 + np.random.rand()
              
            spin_db_input[i,j,k,0] = pd
            spin_db_input[i,j,k,1] = t1
            spin_db_input[i,j,k,2] = t2
            spin_db_input[i,j,k,3] = b0

pd_mask_db = setdevice(torch.from_numpy((spin_db_input[:,:,:,0] > 0).astype(np.float32)))
pd_mask_db = pd_mask_db.flip([1,2]).permute([0,2,1])
            
spins.set_system(real_phantom_resized)

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
flips[0,:,0] = 5*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
#flips[0,:,1] = torch.rand(flips.shape[1])*90*np.pi/180

# randomize RF phases
flips[0,:,1] = torch.tensor(scanner.phase_cycler[:NRep]).float()*np.pi/180

flips = setdevice(flips)

scanner.init_flip_tensor_holder()

B1plus = torch.zeros((scanner.NCoils,1,scanner.NVox,1,1), dtype=torch.float32)
B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([scanner.NCoils, scanner.NVox]))
scanner.B1plus = setdevice(B1plus)    
scanner.set_flip_tensor_withB1plus(flips)

# rotate ADC according to excitation phase
scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.1*1e-4*np.ones((scanner.T,scanner.NRep))).float()
event_time[0,:] =  2e-3
event_time[1,:] =  0.5*1e-3   # for 96
event_time[-2,:] = 2*1e-3
event_time[-1,:] = 2.9*1e-3 + 1
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
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes

#############################################################################
## Forward process ::: ######################################################
    
# forward/adjoint pass
scanner.forward_sparse_fast(spins, event_time)
scanner.adjoint()

# try to fit this
target_phantom = scanner.reco.clone()
   
# save sequence parameters and target image to holder object
targetSeq = core.target_seq_holder.TargetSequenceHolder(flips,event_time,grad_moms,scanner,spins,target_phantom)
if True: # check sanity: is target what you expect and is sequence what you expect
    plt.plot(np.cumsum(tonumpy(scanner.ROI_signal[:,0,0])),tonumpy(scanner.ROI_signal[:,0,1:3]), label='x')

    scanner.do_SAR_test(flips, event_time)    
    targetSeq.export_to_matlab(experiment_id, today_datestr)
    targetSeq.export_to_pulseq(experiment_id,today_datestr,sequence_class, plot_seq=False)
    
    targetSeq.print_status(True, reco=None, do_scanner_query=do_scanner_query)
    
# Prepare target db: iterate over all samples in the DB
target_db = setdevice(torch.zeros((nmb_samples,NVox,2)).float())
    
for i in range(nmb_samples):
    spins.set_system(spin_db_input[i,:,:,:])
    scanner.forward_sparse_fast(spins, event_time)
    scanner.adjoint()
    target_db[i,:,:] = scanner.reco.clone().squeeze() 
  
# since we optimize only NN reco part, we can save time by doing fwd pass (radial enc) on all training examples
adjoint_reco_db = setdevice(torch.zeros((nmb_samples,NVox,2)).float())
adjoint_reco_phantom = setdevice(torch.zeros((1,NVox,2)).float())
    
#stop()
    
    
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
    event_time[-1,:] -= 1
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
    grad_moms[1,:,1] = -grad_moms[1,:,1]*0 + setdevice(torch.rand(grad_moms[1,:,1].shape)-0.5)
    
    grad_moms[-2,:,0] = torch.ones(1)*sz[0]*0      # remove spoiler gradients
    grad_moms[-2,:,1] = -grad_moms[1,:,1]*0      # GRE/FID specific, SPOILER
        
    return [adc_mask, flips, event_time, grad_moms]
    
    
def phi_FRP_model(opt_params,aux_params,do_test_onphantom=False):
    adc_mask,flips,event_time, grad_moms = opt_params
        
    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor_withB1plus(flips)
    # rotate ADC according to excitation phase
    scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2)  # GRE/FID specific, this must be the excitation pulse
          
    scanner.init_gradient_tensor_holder()          
    scanner.set_gradient_precession_tensor(grad_moms,sequence_class) # GRE/FID specific, maybe adjust for higher echoes
         
    lbd = 2*0.3*sz[0]         # switch on of SAR cost
    loss_sar = torch.sum(flips[:,:,0]**2)/NRep
    
    lbd_kspace = 1e1
    
    k = torch.cumsum(grad_moms, 0)
    k = k*torch.roll(scanner.adc_mask, -1).view([T,1,1])
    k = k.flatten()
    mask = setdevice((torch.abs(k) > sz[0]/2).float())
    k = k * mask
    loss_kspace = torch.sum(k**2) / np.prod(sz)
    
    ffwd = scanner.G_adj[2:-2,:,:,:2,:2].permute([0,1,2,4,3]).permute([0,1,3,2,4]).contiguous().view([NRep*(sz[1]+0)*2,NVox*2])
    back = scanner.G_adj[2:-2,:,:,:2,:2].permute([0,1,2,4,3]).permute([0,1,3,2,4]).contiguous().view([NRep*(sz[1]+0)*2,NVox*2]).permute([1,0])
    TT = torch.matmul(back,ffwd) / NVox    
    lbd_ortho = 1e4
    loss_ortho = torch.sum((TT-setdevice(torch.eye(TT.shape[0])))**2) / (NVox)    
    
    # once in a while we want to do test on real phantom, set batch_size to 1 in this case
    local_batchsize = batch_size
    if do_test_onphantom:
        local_batchsize = 1
      
    # loop over all samples in the batch, and do forward/backward pass for each sample
    loss_image = 0
    for btch in range(local_batchsize):
        if do_test_onphantom == False:
            samp_idx = np.random.choice(nmb_samples,1)
            spins.set_system(spin_db_input[samp_idx,:,:,:].squeeze())
        
            tgt = target_db[samp_idx,:,:]
            opt.set_target(tonumpy(tgt).reshape([sz[0],sz[1],2]))
        else:
            tgt = target_phantom
        
        scanner.forward_sparse_fast(spins, event_time)
        scanner.adjoint()
        
        loss_diff = (scanner.reco - tgt)
        loss_image += torch.sum(loss_diff.squeeze()**2/(NVox))    
        
    loss = loss_image + lbd*loss_sar + lbd_kspace*loss_kspace + lbd_ortho*loss_ortho        
    
    print("loss_image: {} loss_sar {} loss_kspace {}".format(loss_image, lbd*loss_sar, lbd_kspace*loss_kspace))
    
    phi = loss
  
    ereco = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    error = e(tonumpy(targetSeq.target_image).ravel(),ereco.ravel())     
    
    return (phi,scanner.reco, error)
        
# %% # OPTIMIZATION land
opt = core.opt_helper.OPT_helper(scanner,spins,None,1)
opt.set_target(tonumpy(targetSeq.target_image).reshape([sz[0],sz[1],2]))
opt.target_seq_holder=targetSeq
opt.experiment_description = experiment_description

opt.optimzer_type = 'Adam'
opt.opti_mode = 'seq'
# 
opt.set_opt_param_idx([3]) # ADC, RF, time, grad
opt.custom_learning_rate = [0.01,0.1,0.1,0.1]

opt.set_handles(init_variables, phi_FRP_model)
opt.scanner_opt_params = opt.init_variables()

lr_inc=np.array([0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.01])

query_kwargs = experiment_id, today_datestr, sequence_class
current_iteration = opt.query_cluster_job(query_kwargs)
opt.scanner_opt_params = [setdevice(par) for par in opt.scanner_opt_params]

opt.custom_learning_rate = [0.01,0.1,0.1,lr_inc[current_iteration]]
print('<seq> Optimization ' + str(i+1) + ' with 10 iters starts now. lr=' +str(lr_inc[current_iteration]))

opt.train_model(training_iter=150, do_vis_image=False, save_intermediary_results=True) # save_intermediary_results=1 if you want to plot them later

isfinished = False
if current_iteration == lr_inc.size - 1:
    isfinished = True
    
    opt.export_to_matlab(experiment_id, today_datestr)
    opt.save_param_reco_history(experiment_id,today_datestr,sequence_class,generate_pulseq=False)
    opt.save_param_reco_history_matlab(experiment_id,today_datestr)
    opt.export_to_pulseq(experiment_id, today_datestr, sequence_class)    

opt.update_cluster_job(query_kwargs,current_iteration,isfinished)
    
    


            