#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss 
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
import core.nnreco
import core.target_seq_holder
from sys import platform
import scipy.misc

from PIL import Image
import imageio                                # pip install imageio

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

from core.pulseq_exporter import pulseq_write_GRE
from core.pulseq_exporter import pulseq_write_RARE
from core.pulseq_exporter import pulseq_write_BSSFP
from core.pulseq_exporter import pulseq_write_EPI

use_gpu = 0
gpu_dev = 0
recreate_pulseq_files = True
recreate_pulseq_files_for_sim = True
do_real_meas = False
get_real_meas = False               # this is to load existing seq.dat files when they were already measured completeley


max_nmb_iter = 30

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

def phaseimg(x):
    return np.angle(1j*x[:,:,1]+x[:,:,0])

# device setter
def setdevice(x):
    x = x.float()
    if use_gpu:
        x = x.cuda(gpu_dev)    
    return x
    
def stop():
    sys.tracebacklimit = 0
    class ExecutionControl(Exception): pass
    raise ExecutionControl('stopped by user')
    sys.tracebacklimit = 1000
    
if platform == 'linux':
    basepath = '/media/upload3t/CEST_seq/pulseq_zero/sequences'
    dp_control = '/media/upload3t/CEST_seq/pulseq_zero/control'
else:
    if os.path.isfile(os.path.join('core','pathfile_local.txt')):
        pathfile ='pathfile_local.txt'
    else:
        pathfile ='pathfile.txt'
        print('You dont have a local pathfile in core/pathfile_local.txt, so we use standard file: pathfile.txt')
                
    with open(os.path.join('core',pathfile),"r") as f:
            path_from_file = f.readline()
    basepath = os.path.join(path_from_file,'sequences')
    dp_control = os.path.join(path_from_file,'control')
    
#    basepath = "//141.67.249.47/MRtransfer/pulseq_zero/sequences"
#    dp_control = "//141.67.249.47/MRtransfer/pulseq_zero/control"
    
experiment_list = []
experiment_list.append(["191108", "p14_tgtRARE_supervised_basic"])



for exp_current in experiment_list:
    date_str = exp_current[0]
    experiment_id = exp_current[1]
    
    if len(exp_current) > 2:
        use_gen_adjoint = exp_current[2]
        adj_alpha = exp_current[3][0]
        adj_iter = exp_current[3][1]
    else:
        use_gen_adjoint = False
    
    fullpath_seq = os.path.join(basepath, "seq" + date_str, experiment_id)
    
    fn_compact_iter_array = "compact_iter_arr.npy"
    compact_iter_array = np.load(os.path.join(os.path.join(fullpath_seq, fn_compact_iter_array)), allow_pickle=True)
    
    try:
        fn_NN_paramlist = "compact_iter_NNparamlist.npy"
        compact_NN_params = np.load(os.path.join(os.path.join(fullpath_seq, fn_NN_paramlist)), allow_pickle=True)
    except:
        compact_NN_params = None
    
    compact_iter_array = compact_iter_array.item()
    
    # define setup
    sz = compact_iter_array['sz']
    NRep = compact_iter_array['NRep']
    T = sz[0] + 4
    NSpins = 2**2
    NCoils = compact_iter_array['all_signals'].shape[1]
    noise_std = 0*1e0                               # additive Gaussian noise std
    NVox = sz[0]*sz[1]
    
    if use_gen_adjoint:
        scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev)
    else:
        scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev)
        
    scanner.B1 = setdevice(torch.from_numpy(compact_iter_array['B1']))
    sequence_class = compact_iter_array['sequence_class']
    
    ###########################################################################    
    ###########################################################################
    ###########################################################################
    
    # process target
    input_array_target = np.load(os.path.join(fullpath_seq, "target_arr.npy"), allow_pickle=True)
    jobtype = "target"    
    input_array_target = input_array_target.item()
    
    scanner.set_adc_mask(torch.from_numpy(input_array_target['adc_mask']))
    scanner.B1 = setdevice(torch.from_numpy(input_array_target['B1']))
    scanner.signal = setdevice(torch.from_numpy(input_array_target['signal']))
    scanner.reco = setdevice(torch.from_numpy(input_array_target['reco']).reshape([NVox,2]))
    scanner.kspace_loc = setdevice(torch.from_numpy(input_array_target['kloc']))
    sequence_class = input_array_target['sequence_class']
    
    flips = setdevice(torch.from_numpy(input_array_target['flips']))
    event_time = setdevice(torch.from_numpy(input_array_target['event_times']))
    grad_moms = setdevice(torch.from_numpy(input_array_target['grad_moms']))
    
    B1plus = torch.ones((NCoils,1,NVox,1,1), dtype=torch.float32)
    #B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([NCoils, NVox]))
    scanner.B1plus = setdevice(B1plus)     
    
    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor_withB1plus(flips)
    
    # rotate ADC according to excitation phase
    scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2) #GRE/FID specific
    
    TR=torch.sum(event_time[:,1])
    TE=torch.sum(event_time[:11,1])
    
    scanner.init_gradient_tensor_holder()
    scanner.set_gradient_precession_tensor(grad_moms,sequence_class)
    
    ######### SIMULATION
    
    # simulation adjoint
    scanner.adjoint()
    target_sim_reco_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    
    # simulation generalized adjoint
    if use_gen_adjoint:
        scanner.generalized_adjoint(alpha=adj_alpha, nmb_iter=adj_iter)
    else:
        scanner.adjoint()
    target_sim_reco_generalized_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    
    # simulation IFFT
    scanner.do_ifft_reco()
    target_sim_reco_ifft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    
    # simulation NUFFT
    scanner.do_nufft_reco()
    target_sim_reco_nufft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    
    coil_idx = 0
    adc_idx = np.where(scanner.adc_mask.cpu().numpy())[0]
    sim_kspace = scanner.signal[coil_idx,adc_idx,:,:2,0]
    target_sim_kspace = tonumpy(sim_kspace.detach()).reshape([sz[0],sz[1],2])
    
    ######### REAL
    # send to scanner
    if (recreate_pulseq_files and do_real_meas) or recreate_pulseq_files_for_sim:
        fn_pulseq = "target.seq"
        iflips = input_array_target['flips']
        ivent = input_array_target['event_times']
        gmo = input_array_target['grad_moms']
        
        seq_params = iflips, ivent, gmo
        
        today_datestr = date_str
        basepath_out = os.path.join(basepath, "seq" + today_datestr)
        basepath_out = os.path.join(basepath_out, experiment_id)
        
        if sequence_class.lower() == "gre":
            pulseq_write_GRE(seq_params, os.path.join(basepath_out, fn_pulseq), plot_seq=False)
        elif sequence_class.lower() == "rare":
            pulseq_write_RARE(seq_params, os.path.join(basepath_out, fn_pulseq), plot_seq=False)
        elif sequence_class.lower() == "se":
            pulseq_write_RARE(seq_params, os.path.join(basepath_out, fn_pulseq), plot_seq=False)            
        elif sequence_class.lower() == "bssfp":
            pulseq_write_BSSFP(seq_params, os.path.join(basepath_out, fn_pulseq), plot_seq=False)
        elif sequence_class.lower() == "epi":
            pulseq_write_EPI(seq_params, os.path.join(basepath_out, fn_pulseq), plot_seq=False)        
    
    if do_real_meas:
            scanner.send_job_to_real_system(experiment_id, date_str, basepath_seq_override=fullpath_seq, jobtype=jobtype)
            scanner.get_signal_from_real_system(experiment_id, date_str, basepath_seq_override=fullpath_seq, jobtype=jobtype)     
            
    if get_real_meas:
        scanner.get_signal_from_real_system(experiment_id, date_str, basepath_seq_override=fullpath_seq, jobtype=jobtype)
        
    real_kspace = scanner.signal[coil_idx,adc_idx,:,:2,0]
    target_real_kspace = tonumpy(real_kspace.detach()).reshape([sz[0],sz[1],2])
    
    # real adjoint
    scanner.adjoint()
    target_real_reco_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    
    # real generalized adjoint
    if use_gen_adjoint:
        scanner.generalized_adjoint(alpha=adj_alpha, nmb_iter=adj_iter)
    else:
        scanner.adjoint()
    target_real_reco_generalized_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    
    # real IFFT
    scanner.do_ifft_reco()
    target_real_reco_ifft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
    
    # real NUFFT
    scanner.do_nufft_reco()
    target_real_reco_nufft = tonumpy(scanner.reco.detach().reshape([sz[0],sz[1],2]))  

    ###########################################################################    
    ###########################################################################
    ###########################################################################
    # process all optimization iterations
    jobtype = "iter"
    nmb_total_iter = compact_iter_array['all_signals'].shape[0]
    

    non_increasing_error_iter=np.arange(nmb_total_iter-1)
    nmb_iter = non_increasing_error_iter.size

    if get_real_meas:
        compact_seq_files = os.listdir(fullpath_seq)
        compact_seq_files = [filename for filename in compact_seq_files if (filename[-3:]=="seq" and filename[:4]=="iter")]
        non_increasing_error_iter = [''.join(i for i in s if i.isdigit()) for s in compact_seq_files]
        non_increasing_error_iter = [ int(x) for x in non_increasing_error_iter ]
        
    #non_increasing_error_iter = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,25,46,67,100,150,200,250,300,400,500,800])
        
    #non_increasing_error_iter = np.concatenate((non_increasing_error_iter[:5],non_increasing_error_iter[-5:]))
#    nmb_iter = non_increasing_error_iter.size
    
    print("exp = {} iteration number = {}".format(exp_current, nmb_iter))

    
    compact_sim_reco_adjoint = np.zeros([nmb_iter,sz[0],sz[1],2])
    compact_sim_reco_generalized_adjoint = np.zeros([nmb_iter,sz[0],sz[1],2])
    compact_sim_reco_ifft = np.zeros([nmb_iter,sz[0],sz[1],2])
    compact_sim_reco_nufft = np.zeros([nmb_iter,sz[0],sz[1],2])
    
    compact_sim_kspace = np.zeros([nmb_iter,sz[0],sz[1],2])
    compact_real_kspace = np.zeros([nmb_iter,sz[0],sz[1],2])
    
    compact_real_reco_adjoint = np.zeros([nmb_iter,sz[0],sz[1],2])
    compact_real_reco_generalized_adjoint = np.zeros([nmb_iter,sz[0],sz[1],2])
    compact_real_reco_ifft = np.zeros([nmb_iter,sz[0],sz[1],2])
    compact_real_reco_nufft = np.zeros([nmb_iter,sz[0],sz[1],2])
    
    lin_iter_counter = 0
    
    for c_iter in non_increasing_error_iter:
        print("Processing the iteration {}/{}  {}/{}".format(c_iter, nmb_total_iter, lin_iter_counter, nmb_iter))
        
        scanner.set_adc_mask(torch.from_numpy(compact_iter_array['all_adc_masks'][c_iter]))
        scanner.signal = setdevice(torch.from_numpy(compact_iter_array['all_signals'][c_iter+1])).unsqueeze(4)
        scanner.reco = setdevice(torch.from_numpy(compact_iter_array['reco_images'][c_iter+1]).reshape([NVox,2]))
        scanner.kspace_loc = setdevice(torch.from_numpy(compact_iter_array['all_kloc'][c_iter+1]))
        
        flips = setdevice(torch.from_numpy(compact_iter_array['flips'][c_iter]))
        event_time = setdevice(torch.from_numpy(compact_iter_array['event_times'][c_iter]))
        grad_moms = setdevice(torch.from_numpy(compact_iter_array['grad_moms'][c_iter]))
        
        scanner.init_flip_tensor_holder()
        scanner.set_flip_tensor_withB1plus(flips)
        
        # rotate ADC according to excitation phase
        if sequence_class.lower() == "gre" or sequence_class.lower() == "bssfp":
            scanner.set_ADC_rot_tensor(-flips[0,:,1] + np.pi/2) #GRE/FID specific #TODO
        elif (sequence_class.lower() == "rare" or sequence_class.lower() == "se"):
            scanner.set_ADC_rot_tensor(flips[0,:,1]*0) #GRE/FID specific #TODO
        else:
            print('dont know sequuence class dont know what to do with ADC rot')
            stop()

        TR=torch.sum(event_time[:,1])
        TE=torch.sum(event_time[:11,1])
        
        scanner.init_gradient_tensor_holder()
        scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes
        
        ###############################################################################
        ######### SIMULATION
        
        # simulation adjoint
        scanner.adjoint()
        
        fn_NN_paramlist = "compact_iter_NNparamlist_" + str(c_iter) + '.pt'
        if os.path.isfile(os.path.join(fullpath_seq, fn_NN_paramlist)):
            spectrum_max_norm = torch.max(scanner.signal).item()
            
            nmb_hidden_neurons_list = [2*sz[0],8,8,2]
            NN = core.nnreco.VoxelwiseNet(scanner.sz, nmb_hidden_neurons_list,use_gpu=use_gpu,gpu_device=gpu_dev)
            state_dict = torch.load(os.path.join(fullpath_seq, fn_NN_paramlist))
            NN.load_state_dict(state_dict)
            
            reco_sep = scanner.adjoint_separable()
            scanner.reco = NN(reco_sep.permute([1,0,2]).reshape([reco_sep.shape[1],reco_sep.shape[0]*reco_sep.shape[2]]))
        
        sim_reco_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        compact_sim_reco_adjoint[lin_iter_counter] = sim_reco_adjoint
        
        # simulation generalized adjoint
        if use_gen_adjoint:
            scanner.generalized_adjoint(alpha=adj_alpha, nmb_iter=adj_iter)
            sim_reco_generalized_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
            compact_sim_reco_generalized_adjoint[lin_iter_counter] = sim_reco_generalized_adjoint
        
        # simulation IFFT
        scanner.do_ifft_reco()
        sim_reco_ifft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        compact_sim_reco_ifft[lin_iter_counter] = sim_reco_ifft
        
        # simulation NUFFT
        scanner.do_nufft_reco()
        sim_reco_nufft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        compact_sim_reco_nufft[lin_iter_counter] = sim_reco_nufft
        
        coil_idx = 0
        adc_idx = np.where(scanner.adc_mask.cpu().numpy())[0]
        sim_kspace = scanner.signal[coil_idx,adc_idx,:,:2,0]
        sim_kspace = tonumpy(sim_kspace.detach()).reshape([sz[0],sz[1],2])
        compact_sim_kspace[lin_iter_counter] = sim_kspace
        
        # send to scanner
        iterfile = "iter" + str(c_iter).zfill(6)
        
        if (recreate_pulseq_files and do_real_meas) or recreate_pulseq_files_for_sim:
            fn_pulseq = "iter" + str(c_iter).zfill(6) + ".seq"
            iflips = compact_iter_array['flips'][c_iter]
            ivent = compact_iter_array['event_times'][c_iter]
            gmo = compact_iter_array['grad_moms'][c_iter]
            
#            enforce positivity on event times
            ivent = np.abs(ivent)
            
            # detect zero flip iteration
            if np.sum(np.abs(iflips)) < 1e-8:
                lin_iter_counter += 1
                continue
            
            seq_params = iflips, ivent, gmo
            
            today_datestr = date_str
            basepath_out = os.path.join(basepath, "seq" + today_datestr)
            basepath_out = os.path.join(basepath_out, experiment_id)
            
            if sequence_class.lower() == "gre":
                pulseq_write_GRE(seq_params, os.path.join(basepath_out, fn_pulseq), plot_seq=False)
            elif sequence_class.lower() == "rare":
                pulseq_write_RARE(seq_params, os.path.join(basepath_out, fn_pulseq), plot_seq=False)
            elif sequence_class.lower() == "se":
                pulseq_write_RARE(seq_params, os.path.join(basepath_out, fn_pulseq), plot_seq=False)                
            elif sequence_class.lower() == "bssfp":
                pulseq_write_BSSFP(seq_params, os.path.join(basepath_out, fn_pulseq), plot_seq=False)
            elif sequence_class.lower() == "epi":
                pulseq_write_EPI(seq_params, os.path.join(basepath_out, fn_pulseq), plot_seq=False)
                
        if do_real_meas:
            scanner.send_job_to_real_system(experiment_id, date_str, basepath_seq_override=fullpath_seq, jobtype=jobtype, iterfile=iterfile)
            scanner.get_signal_from_real_system(experiment_id, date_str, basepath_seq_override=fullpath_seq, jobtype=jobtype, iterfile=iterfile)               
        if get_real_meas:
            scanner.get_signal_from_real_system(experiment_id, date_str, basepath_seq_override=fullpath_seq, jobtype=jobtype, iterfile=iterfile)
            
        if 'extra_par_idx4' in compact_iter_array:
#            scanner.signal *= setdevice(torch.from_numpy(compact_iter_array['extra_par_idx4'][c_iter].reshape([1,1,NRep,1,1])))             
            scanner.signal *= compact_iter_array['extra_par_idx4'][c_iter]
        
        real_kspace = scanner.signal[coil_idx,adc_idx,:,:2,0]
        real_kspace = tonumpy(real_kspace.detach()).reshape([sz[0],sz[1],2])
        compact_real_kspace[lin_iter_counter] = real_kspace
        
        # real adjoint
        scanner.adjoint()
        
        fn_NN_paramlist = "compact_iter_NNparamlist_" + str(c_iter) + '.pt'
        if os.path.isfile(os.path.join(fullpath_seq, fn_NN_paramlist)):
            scanner.signal = 20.0*spectrum_max_norm * scanner.signal / torch.max(scanner.signal)
            
            nmb_hidden_neurons_list = [2*sz[0],8,8,2]
            NN = core.nnreco.VoxelwiseNet(scanner.sz, nmb_hidden_neurons_list,use_gpu=use_gpu,gpu_device=gpu_dev)
            state_dict = torch.load(os.path.join(fullpath_seq, fn_NN_paramlist))
            NN.load_state_dict(state_dict)
            
            reco_sep = scanner.adjoint_separable()
            scanner.reco = NN(reco_sep.permute([1,0,2]).reshape([reco_sep.shape[1],reco_sep.shape[0]*reco_sep.shape[2]]))
        
        real_reco_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        compact_real_reco_adjoint[lin_iter_counter] = real_reco_adjoint
        
        # real generalized adjoint
        if use_gen_adjoint:
            scanner.generalized_adjoint(alpha=adj_alpha, nmb_iter=adj_iter)
            real_reco_generalized_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
            compact_real_reco_generalized_adjoint[lin_iter_counter] = real_reco_generalized_adjoint
        
        # real IFFT
        scanner.do_ifft_reco()
        real_reco_ifft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        compact_real_reco_ifft[lin_iter_counter] = real_reco_ifft
        
        # real NUFFT
        scanner.do_nufft_reco()
        real_reco_nufft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        compact_real_reco_nufft[lin_iter_counter] = real_reco_nufft
        
        lin_iter_counter += 1
        
        if do_real_meas: # this is the web status update
            try:
	            scipy.misc.toimage(magimg(target_sim_reco_adjoint)).save(os.path.join(dp_control, "status_related", "target_sim_reco_adjoint.jpg"))
	            scipy.misc.toimage(magimg(target_real_reco_adjoint)).save(os.path.join(dp_control, "status_related", "target_real_reco_adjoint.jpg"))
	            scipy.misc.toimage(magimg(sim_reco_adjoint)).save(os.path.join(dp_control, "status_related", "sim_reco_adjoint.jpg"))
	            scipy.misc.toimage(magimg(real_reco_adjoint)).save(os.path.join(dp_control, "status_related", "real_reco_adjoint.jpg"))
	            scipy.misc.toimage(phaseimg(sim_reco_adjoint)).save(os.path.join(dp_control, "status_related", "sim_reco_adjoint_phase.jpg"))
	            scipy.misc.toimage(phaseimg(real_reco_adjoint)).save(os.path.join(dp_control, "status_related", "real_reco_adjoint_phase.jpg"))
	            scipy.misc.toimage((1e-8+magimg(sim_kspace))).save(os.path.join(dp_control, "status_related", "sim_kspace.jpg"))
	            scipy.misc.toimage((1e-8+magimg(real_kspace))).save(os.path.join(dp_control, "status_related", "real_kspace.jpg"))
            
	            # make some gifs            
	            gif_array = []
	            for i in range(compact_sim_reco_adjoint.shape[0]):
	                frame = magimg(compact_sim_reco_adjoint[i,:,:,:])
	                frame *= 255.0/np.max(frame)
	                gif_array.append(frame.astype(np.uint8))  
	            imageio.mimsave(os.path.join(dp_control, "status_related", "sim_reco_adjoint.gif"), gif_array)
            
	            gif_array = []
	            for i in range(compact_real_reco_adjoint.shape[0]):
	                frame = magimg(compact_real_reco_adjoint[i,:,:,:])
	                frame *= 255.0/np.max(frame)
	                gif_array.append(frame.astype(np.uint8))  
	            imageio.mimsave(os.path.join(dp_control, "status_related", "real_reco_adjoint.gif"), gif_array)
            
	            status_lines = []
	            status_lines.append("experiment id: " + experiment_id + "\n")
	            status_lines.append("processing iteration {} out of {} \n".format(lin_iter_counter, nmb_iter))
            
	            sim_error = e(magimg(target_sim_reco_adjoint), magimg(sim_reco_adjoint))
	            meas_error = e(magimg(target_real_reco_adjoint), magimg(real_reco_adjoint))
            
	            # compute error wrt target
	            status_lines.append("error = "+str(np.round(sim_error))+"%\n")
	            status_lines.append("error = "+str(np.round(meas_error))+"%\n")
            
	            with open(os.path.join(dp_control, "status_related", "status_lines.txt"),"w") as f:
	                f.writelines(status_lines)
            except:
                print('no status update possible, maybe server connectionto MPI down')
        
    compact_reco_dict = dict()
    
    # target
    compact_reco_dict['target_sim_reco_adjoint'] = target_sim_reco_adjoint
    compact_reco_dict['target_sim_reco_generalized_adjoint'] = target_sim_reco_generalized_adjoint
    compact_reco_dict['target_sim_reco_ifft'] = target_sim_reco_ifft
    compact_reco_dict['target_sim_reco_nufft'] = target_sim_reco_nufft
    compact_reco_dict['target_sim_kspace'] = target_sim_kspace
    
    if do_real_meas or get_real_meas:
        compact_reco_dict['target_real_kspace'] = target_real_kspace
        compact_reco_dict['target_real_reco_adjoint'] = target_real_reco_adjoint
        compact_reco_dict['target_real_reco_generalized_adjoint'] = target_real_reco_generalized_adjoint
        compact_reco_dict['target_real_reco_ifft'] = target_real_reco_ifft
        compact_reco_dict['target_real_reco_nufft'] = target_real_reco_nufft     
    
    # iterations
    compact_reco_dict['all_sim_reco_adjoint'] = compact_sim_reco_adjoint
    compact_reco_dict['all_sim_reco_generalized_adjoint'] = compact_sim_reco_generalized_adjoint
    compact_reco_dict['all_sim_reco_ifft'] = compact_sim_reco_ifft
    compact_reco_dict['all_sim_reco_nufft'] = compact_sim_reco_nufft
    compact_reco_dict['all_sim_kspace'] = compact_sim_kspace
    
    if do_real_meas or get_real_meas:
        compact_reco_dict['all_real_kspace'] = compact_real_kspace
        compact_reco_dict['all_real_reco_adjoint'] = compact_real_reco_adjoint
        compact_reco_dict['all_real_reco_generalized_adjoint'] = compact_real_reco_generalized_adjoint
        compact_reco_dict['all_real_reco_ifft'] = compact_real_reco_ifft
        compact_reco_dict['all_real_reco_nufft'] = compact_real_reco_nufft
        
    compact_reco_dict['target_adc_mask'] = input_array_target['adc_mask']
    compact_reco_dict['target_target_B1'] = input_array_target['B1']
    compact_reco_dict['target_signal'] = input_array_target['signal']
    compact_reco_dict['target_reco'] = input_array_target['reco']
    compact_reco_dict['target_kloc'] = input_array_target['kloc']
    compact_reco_dict['target_flips'] = input_array_target['flips']
    compact_reco_dict['target_event_times'] = input_array_target['event_times']
    compact_reco_dict['target_grad_moms'] = input_array_target['grad_moms']
    
    compact_reco_dict['all_adc_masks'] = compact_iter_array['all_adc_masks']
    
    compact_reco_dict['all_flips'] = compact_iter_array['flips']
    compact_reco_dict['all_event_times'] = compact_iter_array['event_times']
    compact_reco_dict['all_grad_moms'] = compact_iter_array['grad_moms']
    compact_reco_dict['all_kloc'] = compact_iter_array['all_kloc']
    compact_reco_dict['all_errors'] = compact_iter_array['all_errors']
    compact_reco_dict['sz'] = compact_iter_array['sz']
    compact_reco_dict['T'] = compact_iter_array['T']
    compact_reco_dict['NRep'] = compact_iter_array['NRep']
    compact_reco_dict['target'] = compact_iter_array['target']
    compact_reco_dict['sequence_class'] = compact_iter_array['sequence_class']
    compact_reco_dict['B1'] = compact_iter_array['B1']
    compact_reco_dict['iter_idx'] = non_increasing_error_iter
    
    savepath = os.path.join(basepath, "results", "seq" + date_str, experiment_id)
    try:
        os.makedirs(savepath)
    except:
        pass
    
    if do_real_meas or get_real_meas:
        #np.save(os.path.join(os.path.join(savepath, "compact_meas_reco_dict.npy")), compact_reco_dict)
        scipy.io.savemat(os.path.join(savepath,"compact_meas_reco_dict.mat"), compact_reco_dict)
    else:
        #np.save(os.path.join(os.path.join(savepath, "compact_sim_reco_dict.npy")), compact_reco_dict)
        scipy.io.savemat(os.path.join(savepath,"compact_sim_reco_dict.mat"), compact_reco_dict)
        

stop()
    
    
draw_iter = 1

# Visualize simulated images

plt.subplot(221)
plt.imshow(magimg(compact_sim_reco_adjoint[draw_iter]), interpolation='none')
plt.title("sim ADJOINT")
plt.subplot(222)
plt.imshow(magimg(compact_sim_reco_generalized_adjoint[draw_iter]), interpolation='none')
plt.title("sim GENERALIZED ADJOINT") 
plt.subplot(223)
plt.imshow(magimg(compact_sim_reco_ifft[draw_iter]), interpolation='none')
plt.title("sim IFFT")
plt.subplot(224)
plt.imshow(magimg(compact_sim_reco_nufft[draw_iter]), interpolation='none')
plt.title("sim NUFFT") 

plt.ion()
plt.show()

# Visualize kspace

plt.subplot(121)
plt.imshow(magimg(compact_sim_kspace[draw_iter]), interpolation='none')
plt.title("sim kspace pwr")
plt.subplot(122)
plt.imshow(magimg(compact_real_kspace[draw_iter]), interpolation='none')
plt.title("real kspace pwr") 

plt.ion()
plt.show()

# Visualize measured images

plt.subplot(221)
plt.imshow(magimg(compact_real_reco_adjoint[draw_iter]), interpolation='none')
plt.title("real ADJOINT")
plt.subplot(222)
plt.imshow(magimg(compact_real_reco_generalized_adjoint[draw_iter]), interpolation='none')
plt.title("real GENERALIZED ADJOINT") 
plt.subplot(223)
plt.imshow(magimg(compact_real_reco_ifft[draw_iter]), interpolation='none')
plt.title("real IFFT")
plt.subplot(224)
plt.imshow(magimg(compact_real_reco_nufft[draw_iter]), interpolation='none')
plt.title("real NUFFT") 

plt.ion()
plt.show()


        

    