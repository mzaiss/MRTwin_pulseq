# -*- coding: utf-8 -*-
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

sequence_class = "gre_dream"
# fullpath_seq = r"C:\Users\danghi\Documents\MRzero\q14_tgtT1_tskT1invrec_supervised_seqNNvivo_time_zero.py"
fullpath_seq = r"/home/moritz/nam_storage/Sequences/200506/q14_tgtT1_64"

fn_alliter_array = "alliter_arr.npy"
alliter_array = np.load(os.path.join(os.path.join(fullpath_seq, fn_alliter_array)), allow_pickle=True)
alliter_array = alliter_array.item()

# iteration_idx = [4999,6079,7159,8239,9329]
# iteration_idx = [*(np.linspace(5000,5079,15).astype('int')),*(np.linspace(6080,6159,15).astype('int')),*(np.linspace(7160,7239,15).astype('int')),*(np.linspace(8230,8209,15).astype('int'))]
# iteration_idx.extend([4999,6079,7159,8239,9329])
# iteration_idx.sort()
iteration_idx = [*np.arange(49,850,50)]

double_precision = False
use_supermem = True
do_scanner_query = False

experiment_id = 'q14_tgtT1_64'
today_datestr = '200506'
base_path = r"/home/moritz/nam_storage/Sequences/200506/q14_tgtT1_64/"


use_gpu = 1
gpu_dev = 2

if sys.platform != 'linux':
    use_gpu = 1
    gpu_dev = 2


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
sz = np.array([64,64])                                           # image size
extraRep = 6
NRep = extraRep*sz[1]                                   # number of repetitions
T = sz[0] + 7                                        # number of events F/R/P
NSpins = 12**2                                # number of spin sims in each voxel
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


spins.set_system(real_phantom_resized)
core.FID_normscan.make_FID(real_phantom_resized,do_scanner_query=do_scanner_query)


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

event_time_numpy = alliter_array['event_times'][iteration_idx]
TIs = np.abs(event_time_numpy[:,2,[i*32 for i in range(extraRep)]])
waitings = np.abs(event_time_numpy[:,-1,[i*32-1 for i in range(extraRep)]])
waitings[:,0] = 2.9*1e-3
# gradient-driver precession
# Cartesian encoding
grad_moms = torch.zeros((T,NRep,2), dtype=torch.float32)

meas_indices=np.zeros((extraRep,measRepStep))
for i in range(0,extraRep):
    meas_indices[i,:] = np.arange(i*measRepStep,(i+1)*measRepStep)

for j in range(0,extraRep):
    # second action after inversion pulse (chrusher)
    grad_moms[1,j*measRepStep] = sz[0]*2.0#1e-2
    grad_moms[1,j*measRepStep] = 2.0*sz[0]#1e-2

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
grad_moms = setdevice(grad_moms)

# end sequence 

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes


T1_maps = np.zeros((len(iteration_idx),sz[0],sz[1]))
reco = np.zeros((len(iteration_idx),NVox,extraRep))
patch_target = np.zeros((len(iteration_idx),sz[0],sz[1]))
patch_reco = np.zeros((len(iteration_idx),NVox,extraRep))
patch_reco_cnn = np.zeros((len(iteration_idx),sz[0],sz[1]))
vivo_reco = np.zeros((len(iteration_idx),NVox,extraRep))
T1_maps_NN = np.zeros((len(iteration_idx),sz[0],sz[1]))

do_sim = True
do_vivo = False

for i,iter_idx in enumerate(iteration_idx):
    print(i)
    # event timing vector 
    event_time = torch.from_numpy(0.08*1e-3*np.ones((T,NRep))).float()
    # all measurements event times
    event_time[3,:] =  2e-3
    event_time[4,:] =  5.5*1e-3   # for 96
    event_time[-2,:] = 2*1e-3
    event_time[-1,:] = 2.9*1e-3
    
    TI = torch.from_numpy(TIs[i])
    waiting = torch.from_numpy(waitings[i])
    
    # first action
    for j in range(0,extraRep):
        event_time[0,j*measRepStep] = 1*1e-3
        event_time[1,j*measRepStep] = 1*1e-3
        event_time[2,j*measRepStep] = TI[j]
        event_time[1,j*measRepStep] = 2e-3
        if j>0:
            event_time[-1,j*measRepStep-1] = waiting[j]       #delay after readout before next inversion ( only after the first event played out)
    event_time = setdevice(event_time)
    
    #scanner.set_gradient_precession_tensor(grad_moms,'gre')
    #############################################################################
    ## Forward process ::: ######################################################
        
    #scanner.forward_sparse_fast_supermem(spins, event_time)
    #scanner.forward_sparse_fast(spins, event_time)
    # fn_NN_paramlist = "alliter_NNparamlist_" + str(iter_idx) + '.pt'
    # nmb_hidden_neurons_list = [extraRep,16,32,16,1]
    # NN = core.nnreco.VoxelwiseNet(scanner.sz, nmb_hidden_neurons_list, use_gpu=0, gpu_device=gpu_dev)
    # state_dict = torch.load(os.path.join(fullpath_seq, fn_NN_paramlist), map_location=torch.device('cpu'))
    # state_dict = torch.load(os.path.join(r'\\141.67.249.47\MRTransfer\pulseq_zero\sequences\seq200218\q14_tgtT1_tskT1invrec_supervised_seqNNvivo_from_zero_tueb.py', 'NN_last.pt'), map_location=torch.device('cpu'))
    # #state_dict = torch.load(os.path.join(opt.get_base_path(experiment_id, today_datestr)[1], 'last_NN.pt'), map_location=torch.device('cpu'))
    # NN.load_state_dict(state_dict)

    nmb_hidden_neurons_list = [extraRep,16,32,16,1]
    NN = core.nnreco.VoxelwiseNet(sz, nmb_hidden_neurons_list,use_gpu=use_gpu,gpu_device=gpu_dev)    
    state_dict = torch.load(os.path.join(base_path, 'NN_iter'+str(iter_idx+1)+'.pt'), map_location=torch.device('cuda'))
    NN.load_state_dict(state_dict)       
    
    if do_sim:
        spins.set_system(real_phantom_resized)
    
        import stopwatch
        t = stopwatch.Stopwatch()
        scanner.forward_fast(spins, event_time,kill_transverse=kill_transverse)
        t.stop()
        print (t.duration )
        #scanner.forward_mem(spins, event_time)
        #scanner.forward(spins, event_time)
        #scanner.init_signal()
        #scanner.signal[:,:,0,:,:] = 0
        
        reco_all_rep = scanner.adjoint_separable_parallel()
        reco_test = torch.sqrt((reco_all_rep**2).sum(2))
        scale = torch.max(reco_test)
        reco_test = reco_test/scale
        reco_test = reco_test.permute([1,0])
        reco_test = setdevice(reco_test)

    
        reco[i,:,:] = tonumpy(reco_test)
        T1_maps[i,:,:] = tonumpy(NN(reco_test).reshape([sz[0],sz[1]]))

    
        torch.cuda.empty_cache()
    
        
        csz = 32
        nmb_samples = 1
        spin_db_input = np.zeros((nmb_samples, sz[0], sz[1], 5), dtype=np.float32)
        
        for ii in range(nmb_samples):
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
                    
                      
                    spin_db_input[ii,j,k,0] = pd
                    spin_db_input[ii,j,k,1] = t1
                    spin_db_input[ii,j,k,2] = t2
                    spin_db_input[ii,j,k,3] = b0
                
        VV=spin_db_input[:,:,:,3].flatten(); VV=VV[VV>1e-6]; plt.hist(VV,50)         
                    
        tmp = spin_db_input[:,:,:,1:3]
        tmp[tmp < cutoff] = cutoff
        spin_db_input[:,:,:,1:3] = tmp
        
        spins.set_system(spin_db_input[0,:,:,:])
        t = stopwatch.Stopwatch()
        scanner.forward_fast(spins, event_time,kill_transverse=kill_transverse)
        t.stop()
        print (t.duration )
        reco_all_rep = scanner.adjoint_separable_parallel()
        reco_test = torch.sqrt((reco_all_rep**2).sum(2))
        scale = torch.max(reco_test)
        reco_test = reco_test/scale
        reco_test = reco_test.permute([1,0])
        reco_test = setdevice(reco_test)
        
        patch_target[i,:,:] = (np.flip(spin_db_input[0,:,:,1:2].reshape([sz[1],sz[1]]),(0,1))).transpose()
        patch_reco[i,:,:] = tonumpy(reco_test)
        patch_reco_cnn[i,:,:] = tonumpy(NN(reco_test).reshape([sz[0],sz[1]]))
        
    
#    torch.cuda.empty_cache()
        if do_vivo:
            iterfile = 'iter' + str(iter_idx).zfill(6)
            scanner.NCoils = 20
            scanner.init_signal()
            scanner.get_signal_from_real_system('t01_tgtGRESP_tsk_GRESP_no_grad_noflip_kspaceloss_new','200213',jobtype="iter", iterfile=iterfile)
            #reco_sep = scanner.adjoint_separable()
            reco_all_rep = scanner.adjoint_separable_parallel()
            reco_test = torch.sqrt((reco_all_rep**2).sum(2))
            scale = torch.max(reco_test)
            reco_test = reco_test / scale
            reco_test = reco_test.permute([1,0])
            
            vivo_reco[i,:,:] = tonumpy(reco_test)
            T1_maps_NN[i,:,:] = tonumpy(NN(reco_test).reshape([sz[0],sz[1]]))
        

    
    
np.save(base_path+'T1_maps', T1_maps)
np.save(base_path+'reco', reco)
np.save(base_path+'patch_target', patch_target)
np.save(base_path+'patch_reco', patch_reco)
np.save(base_path+'patch_reco_cnn', patch_reco_cnn)
np.save(base_path+'Phantom_T1',np.flip(real_phantom_resized[:,:,1].transpose(),(0,1)))
np.save(base_path+'vivo_reco',vivo_reco)
np.save(base_path+'T1_maps_NN',T1_maps_NN)
# np.save('vivo_fit_T1',T1_map)
np.save('fully_rel_T1',tonumpy(NN(reco_test).reshape([sz[0],sz[1]])))