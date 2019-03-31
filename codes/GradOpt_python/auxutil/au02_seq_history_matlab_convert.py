#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:38:26 2019

@author: mzaiss

experiment desciption:

2D imaging: GRE with spoilers and random phase cycling
# target is fully relaxed GRE (FA5), task is FLASH with TR>=12ms
"""

import os, sys
import numpy as np
import pickle
import scipy
import scipy.io


path = '../out'
experiment_id = 'FLASH_spoiled_lowSAR_test_sunday'

#path = 'K:\CEST_seq\pulseq_zero\sequences'
#experiment_id = 'FLASH_spoiled_lowSAR64_400spins_multistep'



with open(os.path.join(path,experiment_id,'param_reco_history.pdb'), 'rb') as handle:
    b = pickle.load(handle)

NIter = len(b[0])
sz = np.int(np.sqrt(b[0][0]['reco_image'].shape[0]))

NRep = sz
T = sz + 4

if sz == 64:
    event_time = 0.2*1e-3*np.ones((T,NRep))
    event_time[1,:] = 3*1e-3
    event_time[-2,:] = 3*1e-3    
else:
    event_time = 0.2*1e-3*np.ones((T,NRep))
    event_time[1,:] = 1e-3
    event_time[-2,:] = 1e-3

all_flips = np.zeros((NIter,T,NRep,2))
all_event_times = np.zeros((NIter,T,NRep))
all_grad_moms = np.zeros((NIter,T,NRep,2))
all_reco_images = np.zeros((NIter,sz,sz,2))

for ni in range(NIter):
    all_flips[ni] = b[0][ni]['flips_angles']
    all_event_times[ni] = event_time
    all_grad_moms[ni] = b[0][ni]['grad_moms']
    all_reco_images[ni] = b[0][ni]['reco_image'].reshape([sz,sz,2])
    

scanner_dict = dict()
scanner_dict['flips'] = all_flips
scanner_dict['event_times'] = all_event_times
scanner_dict['grad_moms'] = all_grad_moms
scanner_dict['reco_images'] = all_reco_images
scanner_dict['sz'] = np.array([sz,sz])
scanner_dict['T'] = T
scanner_dict['NRep'] = NRep

path=os.path.join('../out/',experiment_id)
try:
    os.mkdir(path)
except:
    print('export_to_matlab: directory already exists')
    
scipy.io.savemat(os.path.join(path,"all_iter.mat"), scanner_dict)











