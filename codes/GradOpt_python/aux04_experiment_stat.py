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
    basepath = 'K:\CEST_seq\pulseq_zero\sequences'
    dp_control = 'K:\CEST_seq\pulseq_zero\control'
    
experiment_list = []
#experiment_list.append(["190623", "e27_opt_pitcher48_sup_allparam_smblock_b1"])
#experiment_list.append(["190623", "e27_opt_pitcher48_sup_onlygrad_smblock_b1"])
experiment_list.append(["190623", "e27_opt_pitcher48_sup_onlyRF_smblock_b1"])


exp_current = experiment_list[0]

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

###########################################################################    
###########################################################################
###########################################################################
# process all optimization iterations
jobtype = "iter"
nmb_total_iter = alliter_array['all_signals'].shape[0]

# autoiter metric
itt = alliter_array['all_errors']
itt[itt > 100] = 100

plt.plot(itt)

