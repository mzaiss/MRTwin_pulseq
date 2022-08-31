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

seq_full = params_target.generate_sequence()
seq_full = Seq.chain(*seq_full)

pulseq_write_EPG_3D(seq_full,os.path.join(path,experiment_id+'.seq'),FOV=220,num_slices=size[-1])

graph = pre_pass.compute_graph(seq_full, *pre_pass_settings)
#graph = FID_graph(seq_full)

## Alternative forward simulations
# bloch = simulate(seq_full,data,10**2)

target_signal_full = execute_graph(graph, seq_full, target_data,return_mag_z=True)
target_mag_z = target_signal_full[1]
target_signal_full = target_signal_full[0]

kspace_full_grappa = get_kmatrix(seq_full, target_signal_full, size, kspace_scaling=torch.tensor([1.,1.,1.]))
kspace_full = seq_full.get_kspace()

target_mag_z = util.to_full(target_mag_z[0][0],data.mask)
target_mag_z *= util.to_full(data.PD,data.mask)

util.plot3D(torch.abs(target_mag_z))
plt.title(f'FULL, z mag')
plt.show()

util.plot3D(torch.abs(kspace_full_grappa))
plt.title(f'FULL, k space')
plt.show()

target_reco_full = reconstruct(target_signal_full, kspace_full, size, return_multicoil=False)

## Alternative reconstructions
# target_reco_full = reconstruct_cartesian_fft(seq_full,target_signal_full,size)
# target_reco_full_bloch = reconstruct(bloch, kspace_full, size, return_multicoil=False)

util.plot3D(torch.abs(target_reco_full))
plt.title(f'FULL, reconstruct function (=adjoint)')
plt.show()

# %% Undersampled reco
R_accel = (2,1) # phase, partition
params_target = GRE3D(*size, R_accel)
params_target.spiralEncoding(spiral_elongation = spiral_elongation, alternating = True)
seq = params_target.generate_sequence()
seq = Seq.chain(*seq)

pulseq_write_EPG_3D(seq,os.path.join(path,experiment_id+'_alternating_grappa_21.seq'),FOV=220,num_slices=size[-1])

graph = pre_pass.compute_graph(seq, *pre_pass_settings)
# graph = FID_graph(seq)

signal = execute_graph(graph, seq, data)
kspace_grappa = get_kmatrix(seq, signal, size, kspace_scaling=torch.tensor([1.,1.,1.]))
kspace = seq.get_kspace()
reco = reconstruct(signal, kspace, size, return_multicoil=False)

util.plot3D(torch.abs(reco))
plt.title(f'UNDERSAMPLED, reconstruct function (=adjoint), Ry={R_accel[0]}, Rz={R_accel[1]}')

# %% GRAPPA
NACS = size[0] # number of autocalibration lines
kernel_size = None # None=automatic (according to some heuristic)
lambd = 0.01 # Tikhonov reg for weight fitting
delta = 0 # CAIPI shift (probably not fully working yet)

# get grappa kernel
wsKernel, ws_imspace = create_grappa_weight_set(
    kspace_full_grappa, R_accel[0], R_accel[1], delta, size, lambd=lambd, kernel_size=kernel_size)

# apply grappa kernel
reco_grappa = grappa_imspace(kspace_grappa, ws_imspace)

reco_grappa_combined = torch.sqrt(torch.sum(torch.abs(reco_grappa)**2,0))
util.plot3D(torch.abs(reco_grappa),figsize=(24,4))
plt.title(f'grappa reco, Ry={R_accel[0]}, Rz={R_accel[1]}')

# %% Optimization
torch.cuda.empty_cache()
gif_array = []
loss_history_gauss = []
var = torch.ones_like(reco_grappa_combined,dtype=float)
loss_gauss = torch.nn.GaussianNLLLoss()
class opt_history:
    def __init__(self):
        self.loss_history = []
        self.FA = []
opt_history = opt_history()

def calc_loss(params: GRE3D, iteration: int):
    seq = params.generate_sequence()
    seq = Seq.chain(*seq)
    global graph  # just to analyze it in ipython after the script ran
        
    # Trainings data
    signal = execute_graph(graph, seq, data)
    kspace_grappa = get_kmatrix(seq, signal, size, kspace_scaling=torch.tensor([1.,1.,1.]))
    reco_grappa = grappa_imspace(kspace_grappa, ws_imspace)
    reco_grappa_combined = torch.sqrt(torch.sum(reco_grappa**2,0))
    
    if iteration % 10 == 0:
        pulseq_write_EPG_3D(seq,os.path.join(path,experiment_id+'_opt_'+str(iteration)+'.seq'),FOV=220,num_slices=size[-1])
        
        plt.figure(figsize=(10, 15))
        plt.subplot(4, 2, (1,2))
        plt.imshow(util.to_numpy(torch.abs((reco_grappa_combined)).transpose(2,1).reshape(size[0],size[1]*size[2])))
        plt.colorbar()
        plt.clim(0,1.2)
        plt.title("Reco Grappa")
        plt.subplot(4, 2, (3,4))
        plt.imshow(util.to_numpy((torch.abs(target_mag_z)).transpose(2,1).reshape(size[0],size[1]*size[2])))
        plt.colorbar()
        plt.clim(0,1.2)
        plt.title("Target")
        plt.subplot(4, 2, (5,6))
        quotients = [number / opt_history.loss_history[0] for number in opt_history.loss_history]
        plt.plot(quotients)
        plt.grid()
        plt.title("Loss Curve")
        plt.subplot(4, 2, (7,8))
        plt.plot(util.to_numpy(params.pulse_angles) * 180 / pi)
        plt.plot(util.to_numpy(params_target.pulse_angles * 180 / pi),color='r')
        plt.grid()
        plt.title("FA")

        gif_array.append(util.current_fig_as_img())
        plt.show()

    # Losses
    loss_spoiler = torch.tensor(0.0, device=util.get_device())
    loss_kspace = torch.tensor(0.0, device=util.get_device())
    loss_image = torch.tensor(0.0, device=util.get_device())
    loss_sar = torch.tensor(0.0, device=util.get_device())
    loss_time = torch.tensor(0.0, device=util.get_device())
    
    loss_image = torch.tensor(0.0, device=util.get_device())
    
    loss_image = util.MSR(reco_grappa_combined,target_mag_z,root=True)
    
    # Lamda
    lbd_spoiler = 0.0
    lbd_kspace = 0.0
    lbd_image = 1.0
    lbd_sar = 0.0
    lbd_time = 0.0
    
    loss = lbd_spoiler*loss_spoiler + lbd_kspace*loss_kspace + lbd_image*loss_image + lbd_sar*loss_sar + lbd_time*loss_time
    opt_history.loss_history.append(loss.detach().cpu())
    opt_history.FA.append(params.pulse_angles.detach().cpu())

    print(
        "% 4d | spoiler %s, kspace %s, image %s, sar %s, time %s | loss %s | c1 %s, c2 %s"
        % (
            iteration,
            colored(f"{lbd_spoiler*loss_spoiler.detach():.3e}", 'red'),
            colored(f"{lbd_kspace*loss_kspace.detach():.3e}", 'red'),
            colored(f"{lbd_image*loss_image.detach():.3e}", 'red'),
            colored(f"{lbd_sar*loss_sar.detach():.3e}", 'red'),
            colored(f"{lbd_time*loss_time.detach():.3e}", 'red'),
            colored(f"{loss.detach():.3e}", 'yellow'),
            colored(f"{c1.detach():.3e}", 'green'),
            colored(f"{c2.detach():.3e}", 'green')
        )
    )
    return loss

# Define the starting parameters for the optimisation process
params = params_target.clone()
# Replace parameters that should be optimized
params.pulse_angles = 1e-3 * torch.rand_like(params.pulse_angles)
params.pulse_angles.requires_grad = True

# Add constant for scaling
c1 = torch.tensor(1.)
c1.requires_grad = True
c2 = torch.tensor(0.01)
c2.requires_grad = True

optimizable_params = [
    {'params': params.pulse_angles, 'lr': 0.03},
    {'params': c1, 'lr': 0.01},
    {'params': c2, 'lr': 0.01},
]

# # Peparation of nn
# nmb_hidden_neurons_list = [meas_count,32,64,32,1]
# NN = nn.VoxelwiseNet(nmb_hidden_neurons_list)

# optimizable_params.extend([
#     {'params': NN.parameters(), 'lr': 2e-3}
# ])
t0 = time.time()

iteration = 0
for restart in range(1):
    optimizer = torch.optim.Adam(optimizable_params, lr=0.03, betas = [0.9, 0.999])
    
    # if checkin != None and restart == 0:
    #     optimizer, _, _, _, NN = util.load_optimizer(optimizer, os.path.join(path, checkin+'.pth'), NN)
        
    for i in range((restart + 1) * 1000):
        iteration += 1

        if i % 5 == 0:
            graph = pre_pass.compute_graph(seq_full, *pre_pass_settings)
        
        t1 = time.time()
        print(t1-t0)
        t0 = time.time()
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        loss = calc_loss(params, iteration)
        loss.backward()
        optimizer.step()

imageio.mimsave(os.path.join(path, experiment_id+'.gif'), gif_array, format='GIF', duration=0.2)
seq = params.generate_sequence()
seq_combined = Seq.preparation(seq, adc_count = size[0], preptype = 'T1', params = (torch.tensor([2.0]),torch.tensor([np.pi])))
pulseq_write_EPG_3D(seq_combined,os.path.join(path,experiment_id+'_final.seq'),FOV=220,num_slices=size[-1])

checkout = {
    'optimizer': optimizer.state_dict(),
    'optimizer_params': optimizer.param_groups,
    'loss_history': opt_history.loss_history,
    'params_target': params_target.clone(),
    'target_reco': target_mag_z,
    'reco': reco_grappa_combined,
    'FA_history': opt_history.FA
#    'NN': NN.state_dict()
}
torch.save(checkout, os.path.join(path, experiment_id+'.pth'))
