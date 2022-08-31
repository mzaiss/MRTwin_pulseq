#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:57:54 2019

@author: aloktyus
"""

# convert .dat files to .numpy arrays

import os,sys
import numpy as np
import scipy
import scipy.io

dp_dat_files = '/media/upload3t/CEST_seq/pulseq_zero/sequences/seq190805/p06_tgtGRESP_tskFLASH_FA_G_ET_48_lowsar_supervised2px_adaptive_frelax/data'
dp_output_npy_files = '/media/upload3t/CEST_seq/pulseq_zero/temp'

all_files = os.listdir(dp_dat_files)

file_idx = 0

# load file, and set shape
raw = np.loadtxt(os.path.join(dp_dat_files,all_files[file_idx]))

nmb_coils = 2
NCol = NLin = np.int(np.sqrt(raw.shape[0]//nmb_coils))

raw = raw.reshape([NLin,nmb_coils,NCol,2])
raw = raw[:,:,:,0] + 1j*raw[:,:,:,1]

raw = np.permute(raw,[1,0,2])

# raw: dims: nmb_chan x NLin x NCol
np.save(os.path.join(dp_output_npy_files, all_files[file_idx][:-4]+'.npy'),raw)

allreco_dict = dict()
allreco_dict['raw'] = raw
scipy.io.savemat(os.path.join(dp_output_npy_files,all_files[file_idx][:-4]+'.mat'), allreco_dict)
