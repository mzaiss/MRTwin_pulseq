"""3D snapshot GRE sequence."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pre_pass
import torch
from numpy import pi
from math import ceil
import imageio
from matplotlib.pyplot import cm
from termcolor import colored
import os, sys
from shutil import copyfile

from seq_builder.GRE3D_builder import GRE3D
from seq_builder.PREP_builder import PREPT1
from new_core import util
from new_core.pdg_main_pass import execute_graph
from new_core.pdg_FID_graph import FID_graph
from new_core.reconstruction import reconstruct, reconstruct_cartesian_fft,adaptive_combine
from new_core.bloch_sim import simulate
from new_core.sim_data import SimData, plot_sim_data, plot_sim_data_3D
from new_core.pulseq_exporter import pulseq_write_EPG_3D
import new_core.sequence as Seq
from new_core.grappa import create_grappa_weight_set, grappa_imspace
from new_core.reconstruction import get_kmatrix
import new_core.selective_pdg as sel_pdg

from auxutil.sensitivity_tools import load_external_coil_sensitivities3D, load_external_coil_sensitivities

experiment_id = '3DSnapshotGRE'
path = os.path.dirname(os.path.abspath(__file__))
checkin = None # File name which should be loaded - None

util.use_gpu = False

# %% ######## Save File Copy#################################################
#############################################################################    
# Save file in defined basepath
           
save_file_copy = 0
if save_file_copy:
    basepath = path
    # pathfile ='pathfile.txt'
    # with open(os.path.join('core',pathfile),"r") as f:
    #     basepath = f.readline()
    #     basepath = os.path.join(basepath, '3_B1T1Map') # project folder name
    #     basepath = os.path.join(basepath, today_datestr)
    try:
        os.makedirs(basepath)
    except:
        pass
    basepath = os.path.join(basepath, experiment_id+'.py')
    f = open(r'C:\Users\weinmusn\Documents\mrzero_src\\.git\\ORIG_HEAD')
    version = f.read()
    f.close()

    copyfile(os.path.realpath(__file__), basepath)
    f = open(basepath[:-3]+"_version_git.txt","w+")
    f.write(version)
    f.close()

# %% Loading of simulation data

# Sequence and reconstruction resolution
size = (64, 64, 4)
size_sim = (64, 64, 4)

data = SimData.load_npz(
    file_name="brainweb/output/subject20.npz"
).select_slice((50,60,70,80))
data = data.resize(*size_sim)
data.T2dash = torch.abs(data.T2dash)

## Plotting von phantom
# plot_sim_data(data)
# plot_sim_data_3D(data,0)

# Create target data for mapping
target_data = data

NCoils = 14
coil_sens = load_external_coil_sensitivities3D('../../data/B1minus/tueb/B1minus_14ch_simu_3D_Gaussians.mat', NCoils, size_sim)

## Coil sensitivities plotting
# plt.figure(figsize=(15,15))
# ip = 1
# for jj in range(NCoils):
#     for sli in range(0,size[-1],2):
#         plt.subplot(NCoils,size[2]//2,ip)
#         plt.imshow(torch.abs(coil_sens[jj,:,:,sli].abs()))
#         ip += 1
# plt.suptitle('coil sens')

data.set_coil_sens(coil_sens)
target_data.set_coil_sens(coil_sens)

pre_pass_settings = (
    target_data.shape.tolist(),
    float(torch.mean(data.T1)),
    float(torch.mean(data.T2)),
    float(torch.mean(data.T2dash)),
    1000,  # Number of states (+ and z) simulated in pre-pass
    1e-9,  # Minimum magnetisation of states in pre-pass
)

# %% Simulate target fully sampled

spiral_elongation = 0.5
R_accel = (1,1) # phase, partition
params_target = GRE3D(*size, R_accel)
params_target.spiralEncoding(spiral_elongation = spiral_elongation, alternating = True)
#params_target.centricEncoding(*size)

params_target.relaxation_time = torch.tensor(5.) # Add 5 s relaxation after one measurement

seq_full = params_target.generate_sequence()

# Add T1 Prep
params = (torch.tensor([2.0, 4.0]),torch.tensor([np.pi, np.pi])) # tuple of two tensors: TI, inversion angle
seq_full_combined = Seq.preparation(seq_full, adc_count = size[0], preptype = 'T1', params = params)

pulseq_write_EPG_3D(seq_full_combined,os.path.join(path,experiment_id+'.seq'),FOV=220,num_slices=size[-1])

graph = pre_pass.compute_graph(seq_full_combined, *pre_pass_settings)
#graph = FID_graph(seq_full)

## Alternative forward simulations
# bloch = simulate(seq_full,data,10**2)

target_signal_full = execute_graph(graph, seq_full_combined, target_data,return_mag_z=True)
target_mag_z = target_signal_full[1]
target_signal_full = target_signal_full[0]

# Plot z magnetization of image 1 and 2
target_mag_z_1 = util.to_full(target_mag_z[1][0],data.mask)
target_mag_z_1 *= util.to_full(data.PD,data.mask)

util.plot3D(torch.abs(target_mag_z_1))
plt.title(f'FULL, z mag - image 1')
plt.show()

# Select image which should be reconstruct
kspace_full = seq_full_combined.get_kspace()
kspace_full_grappa = get_kmatrix(seq_full_combined, target_signal_full, size)

num_image = len(params[0])
mask = torch.zeros([num_image,target_signal_full.shape[0]],dtype=torch.bool)
target_reco_full = torch.zeros([num_image,*size],dtype=torch.complex64)

for num in np.arange(num_image):
    mask[num] = seq_full_combined.get_contrast_mask(num+1)
    
    target_reco_full[num] = reconstruct(target_signal_full[mask[num]], kspace_full[mask[num]], size, return_multicoil=False)
    
    util.plot3D(torch.abs(target_reco_full[num]))
    plt.title(f'FULL, reconstruct function (=adjoint) : image ' + str(num))
    plt.show()

# %% Undersampled reco
R_accel = (2,1) # phase, partition
params_target = GRE3D(*size, R_accel)
params_target.spiralEncoding(spiral_elongation = spiral_elongation, alternating = True)
seq = params_target.generate_sequence()

seq_combined = Seq.preparation(seq, adc_count = size[0], preptype = 'T1', params = params)

pulseq_write_EPG_3D(seq_combined,os.path.join(path,experiment_id+'_alternating_grappa_21.seq'),FOV=220,num_slices=size[-1])

graph = pre_pass.compute_graph(seq_combined, *pre_pass_settings)
# graph = FID_graph(seq)

signal = execute_graph(graph, seq_combined, data)
#kspace_grappa = get_kmatrix(Seq.Sequence(seq_combined[:129]), signal[mask_grappa[0]], size)
kspace = seq_combined.get_kspace()

reco = torch.zeros([num_image,*size],dtype=torch.complex64)
mask_grappa = torch.zeros([num_image,signal.shape[0]],dtype=torch.bool)
kspace_grappa = torch.zeros([num_image,NCoils,*size],dtype=torch.complex64)

for num in np.arange(num_image):
    mask_grappa[num] = seq_combined.get_contrast_mask(num+1)
    
    num_rep_start = ((size[-1]*size[-2]//R_accel[0])*(num)+(num))
    num_rep_end = ((size[-1]*size[-2]//R_accel[0])*(num+1)+(num+1))
    kspace_grappa[num] = get_kmatrix(Seq.Sequence(seq_combined[num_rep_start:num_rep_end]), signal[mask_grappa[num]], size)
    
    reco[num] = reconstruct(signal[mask_grappa[num]], kspace[mask_grappa[num]], size, return_multicoil=False)
    
    util.plot3D(torch.abs(reco[num]))
    plt.title(f'UNDERSAMPLED, reconstruct function (=adjoint), Ry={R_accel[0]}, Rz={R_accel[1]} : image ' + str(num))
    plt.show()


# %% GRAPPA
NACS = size[0] # number of autocalibration lines
kernel_size = None # None=automatic (according to some heuristic)
lambd = 0.01 # Tikhonov reg for weight fitting
delta = 0 # CAIPI shift (probably not fully working yet)

# get grappa kernel
wsKernel, ws_imspace = create_grappa_weight_set(
    kspace_full_grappa, R_accel[0], R_accel[1], delta, size, lambd=lambd, kernel_size=kernel_size)

# apply grappa kernel


reco_grappa = torch.zeros([num_image,NCoils,*size],dtype=torch.complex64)
for num in np.arange(num_image):
    
    reco_grappa[num] = grappa_imspace(kspace_grappa[num], ws_imspace)
    
    util.plot3D(torch.abs(reco_grappa[num]),figsize=(24,4))
    plt.title(f'grappa reco, Ry={R_accel[0]}, Rz={R_accel[1]} : image ' + str(num))
