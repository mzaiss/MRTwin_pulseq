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
from scipy import ndimage
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
import imageio  # pip install imageio

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

from core.pulseq_exporter import pulseq_write_GRE
from core.pulseq_exporter import pulseq_write_RARE
from core.pulseq_exporter import pulseq_write_BSSFP
from core.pulseq_exporter import pulseq_write_EPI

use_gpu = 0
gpu_dev = 0
recreate_pulseq_files = True
recreate_pulseq_files_for_sim = True
do_real_meas = True
get_real_meas = True  # this is to load existing seq.dat files when they were already measured completeley
use_custom_iter_sel_scheme = True  # if this is false search for sampling_of_optiters

max_nmb_iter = 30


# NRMSE error function
def e(gt, x):
    return 100 * np.linalg.norm((gt - x).ravel()) / np.linalg.norm(gt.ravel())


# torch to numpy
def tonumpy(x):
    return x.detach().cpu().numpy()


# get magnitude image
def magimg(x):
    return np.sqrt(np.sum(np.abs(x) ** 2, 2))


def magimg_torch(x):
    return torch.sqrt(torch.sum(torch.abs(x) ** 2, 1))


def phaseimg(x):
    return np.angle(1j * x[:, :, 1] + x[:, :, 0])


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
    if os.path.isfile(os.path.join('core', 'pathfile_local.txt')):
        pathfile = 'pathfile_local.txt'
    else:
        pathfile = 'pathfile.txt'
        print('You dont have a local pathfile in core/pathfile_local.txt, so we use standard file: pathfile.txt')

    with open(os.path.join('core', pathfile), "r") as f:
        path_from_file = f.readline()
    basepath = os.path.join(path_from_file, 'sequences')
    dp_control = os.path.join(path_from_file, 'control')

#    basepath = "//141.67.249.47/MRtransfer/pulseq_zero/sequences"
#    dp_control = "//141.67.249.47/MRtransfer/pulseq_zero/control"

experiment_list = []

experiment_list.append(["191121", "q12_invrec_seqNN_vivo"])

for exp_current in experiment_list:
    date_str = exp_current[0]
    experiment_id = exp_current[1]

    use_gen_adjoint = True

    fullpath_seq = os.path.join(basepath, "seq" + date_str, experiment_id)

    fn_alliter_array = "alliter_arr.npy"
    alliter_array = np.load(os.path.join(os.path.join(fullpath_seq, fn_alliter_array)), allow_pickle=True)

    try:
        fn_NN_paramlist = "alliter_NNparamlist.npy"
        all_NN_params = np.load(os.path.join(os.path.join(fullpath_seq, fn_NN_paramlist)), allow_pickle=True)
    except:
        all_NN_params = None

    alliter_array = alliter_array.item()

    # define setup
    sz = alliter_array['sz']
    NRep = alliter_array['NRep']
    T = sz[0] + 4
    NSpins = 2 ** 2
    NCoils = alliter_array['all_signals'].shape[1]
    noise_std = 0 * 1e0  # additive Gaussian noise std
    NVox = sz[0] * sz[1]
    extraRep = 3

    if use_gen_adjoint:
        scanner = core.scanner.Scanner_fast(sz, NVox, NSpins, NRep, T, NCoils, noise_std, use_gpu + gpu_dev)
    else:
        scanner = core.scanner.Scanner(sz, NVox, NSpins, NRep, T, NCoils, noise_std, use_gpu + gpu_dev)

    scanner.B1 = setdevice(torch.from_numpy(alliter_array['B1']))
    sequence_class = alliter_array['sequence_class']

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
    scanner.reco = setdevice(torch.from_numpy(input_array_target['reco']).reshape([NVox, 2]))
    scanner.kspace_loc = setdevice(torch.from_numpy(input_array_target['kloc']))
    sequence_class = input_array_target['sequence_class']

    flips = setdevice(torch.from_numpy(input_array_target['flips']))
    event_time = setdevice(torch.from_numpy(input_array_target['event_times']))
    grad_moms = setdevice(torch.from_numpy(input_array_target['grad_moms']))

    B1plus = torch.ones((NCoils, 1, NVox, 1, 1), dtype=torch.float32)
    # B1plus[:,0,:,0,0] = torch.from_numpy(real_phantom_resized[:,:,4].reshape([NCoils, NVox]))
    scanner.B1plus = setdevice(B1plus)

    scanner.init_flip_tensor_holder()
    scanner.set_flip_tensor_withB1plus(flips)

    # rotate ADC according to excitation phase
    scanner.set_ADC_rot_tensor(-flips[0, :, 1] + np.pi / 2)  # GRE/FID specific

    TR = torch.sum(event_time[:, 1])
    TE = torch.sum(event_time[:11, 1])

    scanner.init_gradient_tensor_holder()
    scanner.set_gradient_precession_tensor(grad_moms, sequence_class)

    ######### SIMULATION

    # simulation adjoint
    normsim = torch.from_numpy(np.load("auxutil/normsim.npy"))
    scanner.signal = scanner.signal / normsim / NVox
    reco_sep = scanner.adjoint_separable()

    measRepStep = NRep // extraRep
    meas_indices=np.zeros((extraRep,measRepStep))
    reco_all_rep=torch.zeros((extraRep,reco_sep.shape[1],2))
    for j in range(0,extraRep):
        reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)
        
    #Conventional Quantification of brain phantom    
    target_sim_reco_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0], sz[1], 2])
    
    # real measurement
    scanner.export_to_pulseq(experiment_id, date_str, sequence_class)
    scanner.send_job_to_real_system(experiment_id,date_str,jobtype="lastiter")
    scanner.get_signal_from_real_system(experiment_id,date_str,jobtype="lastiter")
    spin_db_input = np.zeros((1, sz[0], sz[1], 5), dtype=np.float32)
    core.FID_normscan.make_FID(spin_db_input[0,:,:,:],do_scanner_query = True)

    normmeas=torch.from_numpy(np.load("auxutil/normmeas.npy"))
    scanner.signal=scanner.signal/normmeas/NVox
    reco_sep_meas = scanner.adjoint_separable()

    
    reco_all_rep_meas=torch.zeros((extraRep,reco_sep.shape[1],2))
    for j in range(0,extraRep):
        reco_all_rep_meas[j,:,:] = reco_sep_meas[meas_indices[j,:],:,:].sum(0)
        
    #Conventional Quantification of real measurement
    target_real_reco_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])


    ###########################################################################    
    ###########################################################################
    ###########################################################################
    # process all optimization iterations
    jobtype = "iter"
    nmb_total_iter = alliter_array['all_signals'].shape[0]

    # autoiter metric error
    itt = alliter_array['all_errors']
    threshhold = 1
    lasterror = 1e10
    sign_fun = lambda x: np.abs(x)
        
    non_increasing_error_iter = []
    
    
    for c_iter in range(itt.size):
    
        if (sign_fun(itt[c_iter] - lasterror) > threshhold):
            non_increasing_error_iter.append(c_iter)
    
            lasterror =itt[c_iter]
            
    non_increasing_error_iter = np.array(non_increasing_error_iter)
    
    non_increasing_error_iter=non_increasing_error_iter[0:150]  # to be altered
    nmb_iter = non_increasing_error_iter.size
    max_nmb_iter = 50
       
    if nmb_iter > max_nmb_iter:
        non_increasing_error_iter = non_increasing_error_iter[(np.ceil(np.arange(0,nmb_iter,np.float(nmb_iter)/max_nmb_iter))).astype(np.int32)]
        plt.plot(non_increasing_error_iter,itt[non_increasing_error_iter],"d")

    # non_increasing_error_iter = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,25,46,67,100,150,200,250,300,400,500,800])

    # non_increasing_error_iter = np.concatenate((non_increasing_error_iter[:5],non_increasing_error_iter[-5:]))
    #    nmb_iter = non_increasing_error_iter.size

    print("exp = {} iteration number = {}".format(exp_current, nmb_iter))
    
    all_sim_reco_adjoint = np.zeros([nmb_iter,sz[0],sz[1]])   
    all_real_reco_adjoint = np.zeros([nmb_iter,sz[0],sz[1]])

    
    lin_iter_counter = 0

    for c_iter in non_increasing_error_iter:
        print("Processing the iteration {}/{}  {}/{}".format(c_iter, nmb_total_iter, lin_iter_counter, nmb_iter))

        ###############################################################################
        ######### SIMULATION

        # simulation adjoint
        fn_NN_paramlist = "alliter_NNparamlist_" + str(c_iter) + '.pt'
        nmb_hidden_neurons_list = [extraRep,16,32,16,1]
        NN = core.nnreco.VoxelwiseNet(scanner.sz, nmb_hidden_neurons_list, use_gpu=use_gpu, gpu_device=gpu_dev)
        state_dict = torch.load(os.path.join(fullpath_seq, fn_NN_paramlist))
        NN.load_state_dict(state_dict)

        sim_reco_adjoint = tonumpy(NN(reco_all_rep)).reshape([sz[0],sz[1]])
        all_sim_reco_adjoint[lin_iter_counter] = sim_reco_adjoint

        real_reco_adjoint = tonumpy(NN(reco_all_rep_meas)).reshape([sz[0],sz[1]])
        all_real_reco_adjoint[lin_iter_counter] = real_reco_adjoint


        lin_iter_counter += 1


    allreco_dict = dict()

    # target
    allreco_dict['target_sim_reco_adjoint'] = target_sim_reco_adjoint
    allreco_dict['target_real_reco_adjoint'] = target_real_reco_adjoint

    # iterations
    allreco_dict['all_sim_reco_adjoint'] = all_sim_reco_adjoint

    allreco_dict['target_adc_mask'] = input_array_target['adc_mask']
    allreco_dict['target_target_B1'] = input_array_target['B1']
    allreco_dict['target_signal'] = input_array_target['signal']
    allreco_dict['target_reco'] = input_array_target['reco']
    allreco_dict['target_kloc'] = input_array_target['kloc']
    allreco_dict['target_flips'] = input_array_target['flips']
    allreco_dict['target_event_times'] = input_array_target['event_times']
    allreco_dict['target_grad_moms'] = input_array_target['grad_moms']

    allreco_dict['all_adc_masks'] = alliter_array['all_adc_masks']

    allreco_dict['all_flips'] = alliter_array['flips']
    allreco_dict['all_event_times'] = alliter_array['event_times']
    allreco_dict['all_grad_moms'] = alliter_array['grad_moms']
    allreco_dict['all_kloc'] = alliter_array['all_kloc']
    allreco_dict['all_errors'] = alliter_array['all_errors']
    allreco_dict['sz'] = alliter_array['sz']
    allreco_dict['T'] = alliter_array['T']
    allreco_dict['NRep'] = alliter_array['NRep']
    allreco_dict['target'] = alliter_array['target']
    allreco_dict['sequence_class'] = alliter_array['sequence_class']
    allreco_dict['B1'] = alliter_array['B1']
    allreco_dict['iter_idx'] = non_increasing_error_iter

    savepath = os.path.join(basepath, "results", "seq" + date_str, experiment_id)
    try:
        os.makedirs(savepath)
    except:
        pass

    if do_real_meas or get_real_meas:
        # np.save(os.path.join(os.path.join(savepath, "all_meas_reco_dict.npy")), allreco_dict)
        scipy.io.savemat(os.path.join(savepath, "all_meas_reco_dict.mat"), allreco_dict)
    else:
        # np.save(os.path.join(os.path.join(savepath, "all_sim_reco_dict.npy")), allreco_dict)
        scipy.io.savemat(os.path.join(savepath, "all_sim_reco_dict.mat"), allreco_dict)

stop()

