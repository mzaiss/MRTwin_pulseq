"""Gradient Echo sequence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pre_pass
import torch

from seq_builder.GRE3D_builder import GRE3D
from seq_builder.TSE2D_builder import TSE2D
from new_core import util
from new_core.pdg_main_pass import execute_graph
from new_core.pdg_FID_graph import FID_graph
from new_core.reconstruction import reconstruct
from new_core.sim_data import SimData, plot_sim_data
from new_core.reconstruction import get_kmatrix, reconstruct_cartesian_fft, adaptive_combine
from new_core.grappa import create_grappa_weight_set, grappa_imspace

from auxutil.sensitivity_tools import  load_external_coil_sensitivities


util.use_gpu = True


# %% Loading of simulation data

data = SimData.load_npz(
    file_name="brainweb/output/subject04.npz"
).select_slice(64)
datasize = (128, 128, 1)
data = data.resize(*datasize)
data.T2dash = torch.abs(data.T2dash)
plot_sim_data(data)

NCoils = 20
# coil_sens = load_external_coil_sensitivities3D('../../data/B1minus/tueb/B1minus_14ch_simu_3D_Gaussians.mat', NCoils, datasize)
coil_sens = load_external_coil_sensitivities('../../data/B1minus/tueb/B1minus_20ch_espirit_sz64_complex.mat', NCoils, (datasize[0],datasize[1])).unsqueeze(3)
# coil_sens = load_external_coil_sensitivities('../../data/B1minus/tueb/B1minus_20ch_simu.mat', NCoils, (128,128)).unsqueeze(3)

data.set_coil_sens(coil_sens)

# Sequence and reconstruction resolution
size = datasize

pre_pass_settings = (
    data.shape.tolist(),
    float(torch.mean(data.T1)),
    float(torch.mean(data.T2)),
    float(torch.mean(data.T2dash)),
    1000,  # Number of states (+ and z) simulated in pre-pass
    1e-9,  # Minimum magnetisation of states in pre-pass
)

def transform_phantom(phantom):
    return phantom.flip(0, 1).permute(1, 0)
# %% Simulate ACS Scan

params_target = GRE3D(*size)
seq_full = params_target.generate_sequence()

# graph = pre_pass.compute_graph(seq_full, *pre_pass_settings)
graph = FID_graph(seq_full)

signal_full = execute_graph(graph, seq_full, data)
reco_full = reconstruct(signal_full, seq_full.get_kspace(), size, return_multicoil=True)

plt.figure(figsize=(12,12))
plt.imshow(torch.abs(reco_full[0,:,:,0]))


# get kspace coordinates at ADC sampling locations
traj = seq_full.get_kspace()
plt.figure(figsize=(12,9))
plt.subplot(1,2,1)
plt.plot(traj[:,0], traj[:,1], '.', ms=1), plt.title('trajectory')
plt.xlabel('kx'), plt.ylabel('ky')
plt.axis('equal')

ksp_full = get_kmatrix(seq_full, signal_full, size)
reco_fft_full = reconstruct_cartesian_fft(seq_full, signal_full, size)

fft_reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(ksp_full,dim=(1, 2)),dim=(1, 2)),dim=(1, 2))
plt.figure(figsize=(12,12))
plt.imshow(torch.abs(fft_reco[0,:,:,0]))


# plt.figure(figsize=(15,1))
# ip = 1
# for cc in range(NCoils):
#     plt.subplot(1,NCoils,ip)
#     plt.imshow(torch.abs(reco_fft_full[cc,:,:,0]))
#     ip += 1
#     plt.suptitle('FULL FFT reco')

# %% Simulate undersampled
# subjects = [
#     4, 5, 6, 18, 20, 38, 41, 42, 43, 44,
#     45, 46, 47, 48, 49, 50, 51, 52, 53, 54
# ]
subjects = [
    4
]


for i in subjects:
    data = SimData.load_npz(
        file_name=f"brainweb/output/subject{i:02d}.npz"
    ).select_slice(64)
    datasize = datasize
    data = data.resize(*datasize)
    data.T2dash = torch.abs(data.T2dash)
    # plot_sim_data(data)
    
    NCoils = 20
    # coil_sens = load_external_coil_sensitivities3D('../../data/B1minus/tueb/B1minus_14ch_simu_3D_Gaussians.mat', NCoils, datasize)
    coil_sens = load_external_coil_sensitivities('../../data/B1minus/tueb/B1minus_20ch_espirit_sz64_complex.mat', NCoils, (size[0],size[1])).unsqueeze(3)
    data.set_coil_sens(coil_sens)
    
    params_target = TSE2D(size[0],size[1])
    seq_full = params_target.generate_sequence(encoding_scheme='centric',remove_p_enc=False)
    
    graph = pre_pass.compute_graph(seq_full, *pre_pass_settings)
    
    signal_full = execute_graph(graph, seq_full, data)
    reco_full = reconstruct(signal_full, seq_full.get_kspace(), size, return_multicoil=True)
    
    R_accel = (3,1) # phase, partition
    params_target = TSE2D(size[0],size[1], R_accel)
    seq = params_target.generate_sequence(encoding_scheme='linear',remove_p_enc=False)
    
    graph = pre_pass.compute_graph(seq, *pre_pass_settings)
    
    signal = execute_graph(graph, seq, data)
    reco = reconstruct(signal, seq.get_kspace(), size, return_multicoil=True)
    
    
    # plt.figure(figsize=(12,12))
    # plt.imshow(torch.abs(reco[0,:,:,0]))
    # plt.suptitle(f'UNDERSAMPLED, reconstruct function (=adjoint), Ry={R_accel[0]}, Rz={R_accel[1]}')
    
    # get kspace coordinates at ADC sampling locations
    traj = seq.get_kspace()
    plt.figure(figsize=(12,9))
    plt.subplot(1,2,1)
    plt.plot(traj[:,0], traj[:,1], '.', ms=1), plt.title('trajectory')
    plt.xlabel('kx'), plt.ylabel('ky')
    plt.axis('equal')
    
    ksp = get_kmatrix(seq, signal, size)
    reco_fft = reconstruct_cartesian_fft(seq, signal, size)
    
    plt.figure(figsize=(15,1))
    ip = 1
    for cc in range(NCoils):
        plt.subplot(1,NCoils,ip)
        plt.imshow(torch.abs(reco_fft[cc,:,:,0]))
        ip += 1
        plt.suptitle('FULL FFT reco')
        
    # %% GRAPPA
    
    NACS = size[0] # number of autocalibration lines
    kernel_size = None # None=automatic (according to some heuristic)
    lambd = 0.01 # Tikhonov reg for weight fitting
    delta = 0 # CAIPI shift (probably not fully working yet)
    
    # get grappa kernel
    wsKernel, ws_imspace = create_grappa_weight_set(
        ksp_full, R_accel[0], R_accel[1], delta, size, lambd=lambd, kernel_size=kernel_size)
    
    # apply grappa kernel
    reco_grappa = grappa_imspace(ksp, ws_imspace)
    
    reco_grappa_sos = torch.sqrt(torch.sum(torch.abs(reco_grappa)**2,0)).cpu().detach().transpose(1,0)
                            
                        
    reco_full_sos = torch.sqrt(torch.sum(torch.abs(reco_full)**2,0))
    reco_us_sos = torch.sqrt(torch.sum(torch.abs(reco)**2,0))

    
    # %% adaptive combine test
    
    recon, weights = adaptive_combine(reco_grappa)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow((torch.abs(recon[:,:,0])))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow((torch.abs(reco_full_sos[:,:,0])))
    plt.axis('off')
    plt.show()