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
do_real_meas = True
get_real_meas = True               # this is to load existing seq.dat files when they were already measured completeley
use_custom_iter_sel_scheme = True   # if this is false search for sampling_of_optiters


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
#experiment_list.append(["190601", "e25_opt_pitcher24_retry_fwd_fwd"])
#experiment_list.append(["190601", "e25_opt_pitcher24_retry_fwd_fwdfastsmem_genadj",True,[1e-4,10]])
#experiment_list.append(["190601", "e25_opt_pitcher24_retry_fwdfastsmem__fwdfastsmem_genadj",True,[1e-4,10]])
#experiment_list.append(["190601", "e25_opt_pitcher24_retry_fwd_fwd_discard"])
#experiment_list.append(["190601", "e25_opt_pitcher24_retry_fwd_fwdfast"])
#experiment_list.append(["190601", "e25_opt_pitcher24_retry_fwd_fwdfastsmem"])
#experiment_list.append(["190601", "t03_tgtRARE_tskRARE_128_linear_saropt_lbd4_smemfixed"])
#
#experiment_list.append(["190602", "e25_opt_pitcher24_retry_fwdfastsmem_kspaceloss"])
#experiment_list.append(["190602", "e25_opt_pitcher24_retry_fwdfastsmem_kspaceloss_ortho"])
#experiment_list.append(["190602", "e25_opt_pitcher24_retry_fwdfastsmem_kspaceloss_genadj",True,[1e-4,10]])    
#
#experiment_list.append(["190602", "e25_opt_pitcher48_retry_fwdfastsmem_kspaceloss"])
#experiment_list.append(["190602", "e25_opt_pitcher48_retry_fwdfastsmem_kspaceloss_ortho"])
#experiment_list.append(["190602", "e25_opt_pitcher48_retry_fwdfastsmem_kspaceloss_genadj",True,[1e-4,10]])    
#experiment_list.append(["190603", "e25_opt_pitcher48_onlysflips"])
#experiment_list.append(["190603", "e25_opt_pitcher48_all_longTR"])
#experiment_list.append(["190603", "e25_opt_pitcher48_onlyPE"])
#experiment_list.append(["190603", "e25_opt_pitcher48_onlyREAD"])

#experiment_list.append(["190603", "e25_opt_pitcher48_onlysflips"])
#experiment_list.append(["190603", "e25_opt_pitcher48_onlyPE"])
#experiment_list.append(["190603", "e25_opt_pitcher48_onlyREAD"])
#experiment_list.append(["190603", "e25_opt_pitcher48_onlyspoilers"])
#experiment_list.append(["190604", "e25_opt_pitcher48_onlysflips_redo_bad_flips"])
#experiment_list.append(["190604", "e25_opt_pitcher48_onlyPE_t2st"])
#experiment_list.append(["190604", "e25_opt_pitcher48_onlyREAD_t2st"])
#experiment_list.append(["190604", "e25_opt_pitcher48_onlyspoilers_t2st"])
#experiment_list.append(["190604", "e25_opt_pitcher48_allgrad_t2st"])
#experiment_list.append(["190602", "t03_tgtRARE_tskRARE_128_init"])
#experiment_list.append(["190604", "e25_opt_pitcher96_onlyflips"])
#experiment_list.append(["190605", "e26_tgtGRESP_tskGRESP_bigX"])
#experiment_list.append(["190605", "e26_tgtGRESP_tskGRESP_bigX_truelegs"]) 
#experiment_list.append(["190606", "e26_tgtGRESP_tskGRESP_bigX_truelegs",True,[1e-5,55]])
#experiment_list.append(["190604", "e25_opt_pitcher48_allparam_t2st_selective_multishape"])

#experiment_list.append(["190604", "e25_opt_pitcher48_allparam_t2st"])
#experiment_list.append(["190607", "e25_opt_pitcher64_allparamm_sar1x"])
#experiment_list.append(["190607", "e25_opt_pitcher64_allparamm_sar2x"])
#experiment_list.append(["190607", "e25_opt_pitcher64_allparamm_sar5x"])
#experiment_list.append(["190607", "e25_opt_pitcher64_allparamm_sar50x"])

#experiment_list.append(["190604", "e25_opt_pitcher96_onlyflips"])
#experiment_list.append(["190603", "e25_opt_pitcher48_onlysflips"])
#experiment_list.append(["190602", "t03_tgtRARE_tskRARE_128_init"])

#experiment_list.append(["190607", "e25_opt_pitcher64_allparamm_sar2x_sl"])
#experiment_list.append(["190607", "e25_opt_pitcher64_allparamm_sar5x_sl"])
#experiment_list.append(["190607", "e25_opt_pitcher64_allparamm_sar50x_sl"])
#
#experiment_list.append(["190604", "e25_opt_pitcher96_onlyflips_sl"])
#experiment_list.append(["190603", "e25_opt_pitcher48_onlysflips_sl"])

#experiment_list.append(["190618", "e25_opt_pitcher96_allparamm_sar5000x_adcrot"])
#experiment_list.append(["19061", "e25_opt_pitcher64_allparamm_sar5000x_adcrot"])
#experiment_list.append(["190618", "e25_opt_pitcher48_supervised_allparam_fwdsmemfix_noortho"])
#experiment_list.append(["190618", "e25_opt_pitcher48_supervised_allparam_smoothblock_fwdsmemfix_noortho"])
#experiment_list.append(["190618", "e25_opt_pitcher48_supervised_onlygrad_fwdsmemfix_noortho"])
#experiment_list.append(["190618", "e25_opt_pitcher48_supervised_onlygrad_smoothblock_fwdsmemfix_noortho"])


#experiment_list.append(["190619", "e25_opt_pitcher48_supervised_onlygrad_smoothblock_noortho_convergencefix"])
#experiment_list.append(["190619", "e25_opt_pitcher48_supervised_onlygrad_noortho_convergencefix"])
#experiment_list.append(["190619", "e25_opt_pitcher48_supervised_allparam_smoothblock_noortho_convergencefix"])
#experiment_list.append(["190619", "e25_opt_pitcher48_supervised_allparam_noortho_convergencefix"])
#experiment_list.append(["190620", "e25_opt_pitcher64_lowspin_onlygrad_noortho"])
#experiment_list.append(["190620", "e25_opt_pitcher64_lowspin_allparam_noortho"])
#experiment_list.append(["190618", "e25_opt_pitcher96_onlygrad_adcrot"])
#experiment_list.append(["190618", "e25_opt_pitcher96_allparamm_sar5000x_adcrot"])
#experiment_list.append(["190708", "e26_tgtGRESP_tskGRESP_bigX"])
#experiment_list.append(["190724", "p04_tgtGRESP_tskFLASH_FA_G_24_lowsar"])
#experiment_list.append(["190724", "p06_tgtGRESP_tskFLASH_FA_G_ET_24_lowsar"])
#experiment_list.append(["190724", "p05_tgtGRESP1s_tskFLASH_ET_24"])
#experiment_list.append(["190724", "p05_tgtGRESP1s_tskFLASH_ET_48"])
#experiment_list.append(["190724", "p07_tgtGRESP_tskFLASH_FA_G_NNscaler_24_lowsar"])
#experiment_list.append(["190724", "p06_tgtGRESP_tskFLASH_FA_G_ET_48_lowsar"])
#experiment_list.append(["190724", "p07_tgtGRESP_tskFLASH_FA_G_NNscaler_48_lowsar"])
#experiment_list.append(["190724", "p04_tgtGRESP_tskFLASH_FA_G_48_lowsar"])
#experiment_list.append(["190725", "p06_tgtGRESP_tskFLASH_FA_G_ET_24_lowsar_supervised"])
#experiment_list.append(["190725", "p06_tgtGRESP_tskFLASH_FA_G_ET_48_lowsar_supervised"])
#experiment_list.append(["190725", "p07_tgtGRESP_tskFLASH_FA_G_NNscaler_24_lowsar_supervised"])


#experiment_list.append(["190728", "p03_tgtGRESP_tskFLASH_FA_G_24_extendedLRprot"])
#experiment_list.append(["190728", "p03_tgtGRESP_tskFLASH_FA_G_32_extendedLRprot"])
#experiment_list.append(["190728", "p03_tgtGRESP_tskFLASH_FA_G_48_extendedLRprot"])
#experiment_list.append(["190728", "p03_tgtGRESP_tskFLASH_FA_G_64_extendedLRprot"])
#experiment_list.append(["190728", "p06_tgtGRESP_tskFLASH_FA_G_ET_24_lowsar_supervised2px"])
#experiment_list.append(["190728", "p06_tgtGRESP_tskFLASH_FA_G_ET_48_lowsar_supervised2px"])
#experiment_list.append(["190728", "p07_tgtGRESP_tskFLASH_FA_G_NNscaler_24_lowsar_supervised2px"])
#experiment_list.append(["190730", "p08_tgtFLAIR_RARE_tsklowSAR"])

#experiment_list.append(["190805", "p06_tgtGRESP_tskFLASH_FA_G_ET_48_lowsar_supervised2px_adaptive_frelax"])

#experiment_list.append(["190821", "t03_tgtRARE_tskRARE_96_inittotarget"])
#experiment_list.append(["190820", "e43_tgSErelaxed_tskSEshortETlastactALlFAScaler"])
#experiment_list.append(["190820", "e24_tgtRARE_tskRARE96_lowSAR_phantom_highpass_scaler"])
#experiment_list.append(["190820", "e24_tgtRARE_tskRARE96_lowSAR_brainphantom_highpass_scaler"])
#experiment_list.append(["190820", "e24_tgtRARE_tskRARE96_lowSAR_brainphantom_highpass"])
#experiment_list.append(["190820", "e24_tgtRARE_tskRARE96_lowSAR_brainphantom"])
#experiment_list.append(["190822", "p10_tgt_nonMR_nosar"])
#experiment_list.append(["190820", "e24_tgtRARE_tskRARE96_lowSAR_brainphantom_highpass"])
#experiment_list.append(["190925", "e24_tgtRARE_tskRARE64_lowSAR_phantom_supervised"])
experiment_list.append(["190927", "e24_tgtRARE_tskRARE96_lowSAR_highpass_scaler_brainphantom_sardiv100"])
#experiment_list.append(["191022", "FAU_t01_tgtGRESP_tsk_GRESP_no_grad_noflip_kspaceloss_new"])



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
    
    fn_alliter_array = "alliter_arr.npy"
    alliter_array = np.load(os.path.join(os.path.join(fullpath_seq, fn_alliter_array)), allow_pickle=True)
    
    alliter_array = alliter_array.item()
    
    # define setup
    sz = alliter_array['sz']
    NRep = alliter_array['NRep']
    T = sz[0] + 4
    NSpins = 2**2
    NCoils = alliter_array['all_signals'].shape[1]
    noise_std = 0*1e0                               # additive Gaussian noise std
    NVox = sz[0]*sz[1]
    
    if use_gen_adjoint:
        scanner = core.scanner.Scanner_fast(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev)
    else:
        scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,T,NCoils,noise_std,use_gpu+gpu_dev)
        
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
    nmb_total_iter = alliter_array['all_signals'].shape[0]
    

    sampling_of_optiters = 'SAR' # 'SAR' 'err'
         
    if sampling_of_optiters=='err':
        # autoiter metric error
        itt = alliter_array['all_errors']
        itt = alliter_array['all_errors']
        threshhold = 1
        lasterror = 1e10
        sign_fun = lambda x: np.abs(x)
        
    if sampling_of_optiters=='SAR':
        # autoiter metric SAR
        itt = alliter_array['flips']
        itt = np.sum(itt[:,:,:,0],axis=1)
        itt = np.sum(itt,axis=1)
        threshhold = 0.05
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

    
    
    if nmb_iter > max_nmb_iter:
        non_increasing_error_iter = non_increasing_error_iter[(np.ceil(np.arange(0,nmb_iter,np.float(nmb_iter)/max_nmb_iter))).astype(np.int32)]
        plt.plot(non_increasing_error_iter,itt[non_increasing_error_iter],"d")
       
   
            
#    stop()    
    if use_custom_iter_sel_scheme:

#        non_increasing_error_iter = np.arange(0,itt.size,itt.size//max_nmb_iter)
        non_increasing_error_iter = np.concatenate( (np.arange(0,max_nmb_iter-9,1), max_nmb_iter-9+np.array([10,20,50,100,200,300,400,500,800])))
        nmb_iter = non_increasing_error_iter.size
    
    plt.plot(itt)
    plt.plot(non_increasing_error_iter,itt[non_increasing_error_iter],"x")         
    plt.show()

    if get_real_meas:
        all_seq_files = os.listdir(fullpath_seq)
        all_seq_files = [filename for filename in all_seq_files if (filename[-3:]=="seq" and filename[:4]=="iter")]
        non_increasing_error_iter = [''.join(i for i in s if i.isdigit()) for s in all_seq_files]
        non_increasing_error_iter = [ int(x) for x in non_increasing_error_iter ]
        
#        nmb_iter = non_increasing_error_iter.size


    #non_increasing_error_iter = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,25,46,67,100,150,200,250,300,400,500,800])
        
    #non_increasing_error_iter = np.concatenate((non_increasing_error_iter[:5],non_increasing_error_iter[-5:]))
#    nmb_iter = non_increasing_error_iter.size
    
    print("exp = {} iteration number = {}".format(exp_current, nmb_iter))

    
    all_sim_reco_adjoint = np.zeros([nmb_iter,sz[0],sz[1],2])
    all_sim_reco_generalized_adjoint = np.zeros([nmb_iter,sz[0],sz[1],2])
    all_sim_reco_ifft = np.zeros([nmb_iter,sz[0],sz[1],2])
    all_sim_reco_nufft = np.zeros([nmb_iter,sz[0],sz[1],2])
    
    all_sim_kspace = np.zeros([nmb_iter,sz[0],sz[1],2])
    all_real_kspace = np.zeros([nmb_iter,sz[0],sz[1],2])
    
    all_real_reco_adjoint = np.zeros([nmb_iter,sz[0],sz[1],2])
    all_real_reco_generalized_adjoint = np.zeros([nmb_iter,sz[0],sz[1],2])
    all_real_reco_ifft = np.zeros([nmb_iter,sz[0],sz[1],2])
    all_real_reco_nufft = np.zeros([nmb_iter,sz[0],sz[1],2])
    
    lin_iter_counter = 0
    
    for c_iter in non_increasing_error_iter:
        print("Processing the iteration {}/{}  {}/{}".format(c_iter, nmb_total_iter, lin_iter_counter, nmb_iter))
        
        scanner.set_adc_mask(torch.from_numpy(alliter_array['all_adc_masks'][c_iter]))
        scanner.signal = setdevice(torch.from_numpy(alliter_array['all_signals'][c_iter+1])).unsqueeze(4)
        scanner.reco = setdevice(torch.from_numpy(alliter_array['reco_images'][c_iter+1]).reshape([NVox,2]))
        scanner.kspace_loc = setdevice(torch.from_numpy(alliter_array['all_kloc'][c_iter+1]))
        
        flips = setdevice(torch.from_numpy(alliter_array['flips'][c_iter]))
        event_time = setdevice(torch.from_numpy(alliter_array['event_times'][c_iter]))
        grad_moms = setdevice(torch.from_numpy(alliter_array['grad_moms'][c_iter]))
        
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
        sim_reco_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        all_sim_reco_adjoint[lin_iter_counter] = sim_reco_adjoint
        
        # simulation generalized adjoint
        if use_gen_adjoint:
            scanner.generalized_adjoint(alpha=adj_alpha, nmb_iter=adj_iter)
            sim_reco_generalized_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
            all_sim_reco_generalized_adjoint[lin_iter_counter] = sim_reco_generalized_adjoint
        
        # simulation IFFT
        scanner.do_ifft_reco()
        sim_reco_ifft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        all_sim_reco_ifft[lin_iter_counter] = sim_reco_ifft
        
        # simulation NUFFT
        scanner.do_nufft_reco()
        sim_reco_nufft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        all_sim_reco_nufft[lin_iter_counter] = sim_reco_nufft
        
        coil_idx = 0
        adc_idx = np.where(scanner.adc_mask.cpu().numpy())[0]
        sim_kspace = scanner.signal[coil_idx,adc_idx,:,:2,0]
        sim_kspace = tonumpy(sim_kspace.detach()).reshape([sz[0],sz[1],2])
        all_sim_kspace[lin_iter_counter] = sim_kspace
        
        # send to scanner
        iterfile = "iter" + str(c_iter).zfill(6)
        
        if (recreate_pulseq_files and do_real_meas) or recreate_pulseq_files_for_sim:
            fn_pulseq = "iter" + str(c_iter).zfill(6) + ".seq"
            iflips = alliter_array['flips'][c_iter]
            ivent = alliter_array['event_times'][c_iter]
            gmo = alliter_array['grad_moms'][c_iter]
            
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
        
        real_kspace = scanner.signal[coil_idx,adc_idx,:,:2,0]
        real_kspace = tonumpy(real_kspace.detach()).reshape([sz[0],sz[1],2])
        all_real_kspace[lin_iter_counter] = real_kspace
        
        # real adjoint
        scanner.adjoint()
        real_reco_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        all_real_reco_adjoint[lin_iter_counter] = real_reco_adjoint
        
        # real generalized adjoint
        if use_gen_adjoint:
            scanner.generalized_adjoint(alpha=adj_alpha, nmb_iter=adj_iter)
            real_reco_generalized_adjoint = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
            all_real_reco_generalized_adjoint[lin_iter_counter] = real_reco_generalized_adjoint
        
        # real IFFT
        scanner.do_ifft_reco()
        real_reco_ifft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        all_real_reco_ifft[lin_iter_counter] = real_reco_ifft
        
        # real NUFFT
        scanner.do_nufft_reco()
        real_reco_nufft = tonumpy(scanner.reco.detach()).reshape([sz[0],sz[1],2])
        all_real_reco_nufft[lin_iter_counter] = real_reco_nufft
        
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
	            for i in range(all_sim_reco_adjoint.shape[0]):
	                frame = magimg(all_sim_reco_adjoint[i,:,:,:])
	                frame *= 255.0/np.max(frame)
	                gif_array.append(frame.astype(np.uint8))  
	            imageio.mimsave(os.path.join(dp_control, "status_related", "sim_reco_adjoint.gif"), gif_array)
            
	            gif_array = []
	            for i in range(all_real_reco_adjoint.shape[0]):
	                frame = magimg(all_real_reco_adjoint[i,:,:,:])
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
        
    allreco_dict = dict()
    
    # target
    allreco_dict['target_sim_reco_adjoint'] = target_sim_reco_adjoint
    allreco_dict['target_sim_reco_generalized_adjoint'] = target_sim_reco_generalized_adjoint
    allreco_dict['target_sim_reco_ifft'] = target_sim_reco_ifft
    allreco_dict['target_sim_reco_nufft'] = target_sim_reco_nufft
    allreco_dict['target_sim_kspace'] = target_sim_kspace
    
    if do_real_meas or get_real_meas:
        allreco_dict['target_real_kspace'] = target_real_kspace
        allreco_dict['target_real_reco_adjoint'] = target_real_reco_adjoint
        allreco_dict['target_real_reco_generalized_adjoint'] = target_real_reco_generalized_adjoint
        allreco_dict['target_real_reco_ifft'] = target_real_reco_ifft
        allreco_dict['target_real_reco_nufft'] = target_real_reco_nufft     
    
    # iterations
    allreco_dict['all_sim_reco_adjoint'] = all_sim_reco_adjoint
    allreco_dict['all_sim_reco_generalized_adjoint'] = all_sim_reco_generalized_adjoint
    allreco_dict['all_sim_reco_ifft'] = all_sim_reco_ifft
    allreco_dict['all_sim_reco_nufft'] = all_sim_reco_nufft
    allreco_dict['all_sim_kspace'] = all_sim_kspace
    
    if do_real_meas or get_real_meas:
        allreco_dict['all_real_kspace'] = all_real_kspace
        allreco_dict['all_real_reco_adjoint'] = all_real_reco_adjoint
        allreco_dict['all_real_reco_generalized_adjoint'] = all_real_reco_generalized_adjoint
        allreco_dict['all_real_reco_ifft'] = all_real_reco_ifft
        allreco_dict['all_real_reco_nufft'] = all_real_reco_nufft
        
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
        #np.save(os.path.join(os.path.join(savepath, "all_meas_reco_dict.npy")), allreco_dict)
        scipy.io.savemat(os.path.join(savepath,"all_meas_reco_dict.mat"), allreco_dict)
    else:
        #np.save(os.path.join(os.path.join(savepath, "all_sim_reco_dict.npy")), allreco_dict)
        scipy.io.savemat(os.path.join(savepath,"all_sim_reco_dict.mat"), allreco_dict)
        

stop()
    
    
draw_iter = 1

# Visualize simulated images

plt.subplot(221)
plt.imshow(magimg(all_sim_reco_adjoint[draw_iter]), interpolation='none')
plt.title("sim ADJOINT")
plt.subplot(222)
plt.imshow(magimg(all_sim_reco_generalized_adjoint[draw_iter]), interpolation='none')
plt.title("sim GENERALIZED ADJOINT") 
plt.subplot(223)
plt.imshow(magimg(all_sim_reco_ifft[draw_iter]), interpolation='none')
plt.title("sim IFFT")
plt.subplot(224)
plt.imshow(magimg(all_sim_reco_nufft[draw_iter]), interpolation='none')
plt.title("sim NUFFT") 

plt.ion()
plt.show()

# Visualize kspace

plt.subplot(121)
plt.imshow(magimg(all_sim_kspace[draw_iter]), interpolation='none')
plt.title("sim kspace pwr")
plt.subplot(122)
plt.imshow(magimg(all_real_kspace[draw_iter]), interpolation='none')
plt.title("real kspace pwr") 

plt.ion()
plt.show()

# Visualize measured images

plt.subplot(221)
plt.imshow(magimg(all_real_reco_adjoint[draw_iter]), interpolation='none')
plt.title("real ADJOINT")
plt.subplot(222)
plt.imshow(magimg(all_real_reco_generalized_adjoint[draw_iter]), interpolation='none')
plt.title("real GENERALIZED ADJOINT") 
plt.subplot(223)
plt.imshow(magimg(all_real_reco_ifft[draw_iter]), interpolation='none')
plt.title("real IFFT")
plt.subplot(224)
plt.imshow(magimg(all_real_reco_nufft[draw_iter]), interpolation='none')
plt.title("real NUFFT") 

plt.ion()
plt.show()


        

    