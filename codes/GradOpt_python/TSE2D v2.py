"""Turbo Spin Echo sequence."""

# %% Imports
from __future__ import annotations

import matplotlib.pyplot as plt
import pre_pass
import torch
import numpy as np
import imageio
from termcolor import colored
from time import time

from seq_builder.TSE2D_builder import TSE2D, plot_optimization_progress
from new_core import util
from new_core.pdg_main_pass import execute_graph
import new_core.selective_pdg as sel_pdg
from new_core import bloch_sim
from new_core.pdg_FID_graph import FID_graph
from new_core.reconstruction import reconstruct
from new_core.sim_data import SimData, plot_sim_data

util.use_gpu = True


# %% Loading of simulation data

data = SimData.load('../../data/numerical_brain_cropped.mat')
# data = SimData.load_npz("brainweb/output/subject05.npz").select_slice(64)
data = data.resize(64, 64, 1)
data.T2dash = torch.abs(data.T2dash)
plot_sim_data(data)

# Sequence and reconstruction resolution
size = (64, 64, 1)

pre_pass_settings = (
    data.shape.tolist(),
    float(torch.mean(data.T1)),
    float(torch.mean(data.T2)),
    float(torch.mean(data.T2dash)),
    1000,  # Number of states (+ and z) simulated in pre-pass
    1e-9,  # Minimum magnetisation of states in pre-pass
)


# %% Simulate target
params_target = TSE2D(size[0], size[1])

params_target.excit_pulse_angle = torch.tensor(90 * np.pi/180)
params_target.refoc_pulse_angles[:] = 120 * np.pi/180
# params_target.spoiler[:] = 0

seq = params_target.generate_sequence(encoding_scheme="linear")
for rep in seq:
    rep.gradm *= 0.5
# util.plot_kspace_trajectory(seq, plot_timeline=False)

graph = pre_pass.compute_graph(seq, *pre_pass_settings)
# graph = FID_graph(seq)

target_signal = execute_graph(graph, seq, data)
# target_signal = sel_pdg.execute_graph_selective(
#     graph, seq, data, sel_pdg.is_spin_echo, "ee"
# )
# target_signal = bloch_sim.simulate(seq, data, 10)
target_reco = reconstruct(target_signal, seq.get_kspace())

plt.figure(figsize=(5, 5))
plt.imshow(torch.abs(target_reco[:, :, 0]), vmin=0)
plt.colorbar()
plt.show()


# %% Optimization
gif_array = []
loss_history = []
TEd_hist = []


def calc_loss(params: TSE2D, iteration: int):
    global graph

    TEd_hist.append(params.TEd.clone().detach())

    seq = params.generate_sequence()
    if iteration % 5 == 0:
        graph = pre_pass.compute_graph(seq, *pre_pass_settings)

    signal = execute_graph(graph, seq, data)
    reco = reconstruct(signal, seq.get_kspace(), size)

    loss_spoiler = torch.sum(params.spoiler**2)
    reco_diff = (reco - target_reco)
    loss_image = torch.mean(reco_diff.real**2 + reco_diff.imag**2)
    loss_sar = params.refoc_pulse_angles.abs().sum()
    loss_timing = params.TEd.abs().sum()

    loss = 0.0*loss_spoiler + 1.0*loss_image + 0.0*loss_sar + 1e-2*loss_timing
    loss_history.append(loss.detach().cpu())

    print(
        "\r% 4d | spoiler %s, image %s, sar %s, timing %s | loss %s"
        % (
            iteration,
            colored(f"{loss_spoiler.detach():.3e}", 'red'),
            colored(f"{loss_image.detach():.3e}", 'red'),
            colored(f"{loss_sar.detach():.3e}", 'red'),
            colored(f"{loss.detach():.3e}", 'red'),
            colored(f"{loss_timing.detach():.3e}", 'yellow')
        )
    )

    if iteration % 3 == 0:
        # trajectory = seq.get_kspace_trajectory()
        # gif_array.append(plot_optimization_progress(
        #     reco, target_reco,
        #     params, params_target,
        #     trajectory, loss_history
        # ))
        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        plt.title("Reco")
        plt.imshow(util.to_numpy(reco.abs()))
        plt.colorbar()
        plt.subplot(222)
        plt.title("Loss")
        plt.plot(loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss [arbitrary units]")
        plt.grid()
        plt.subplot(212)
        for r in range(len(seq)):
            mask = util.to_numpy(seq[r].adc_usage != 0)
            time = util.to_numpy(seq[r].event_time.cumsum(0) * 1000)
            rep = r * np.ones(seq[r].event_count)

            plt.plot(time[~mask], rep[~mask], 'k.')
            plt.plot(time[mask], rep[mask], 'r.')

        plt.grid()
        plt.gca().invert_yaxis()
        plt.gca().set_yticks([0, 32])
        plt.xlabel("Time [ms]")
        plt.ylabel("Repetition")
        plt.xlim(-10, 110)
        gif_array.append(util.current_fig_as_img())
        plt.close()

    return loss


# Define the starting parameters for the optimisation process
params = params_target.clone()
# Replace parameters that should be optimized
# params.refoc_pulse_angles = 1e-6 * torch.rand_like(params.refoc_pulse_angles)
# params.refoc_pulse_angles.requires_grad = True
# params.refoc_pulse_phases = 1e-6 * torch.rand_like(params.refoc_pulse_phases)
# params.refoc_pulse_phases.requires_grad = True
params.TEd = 50e-3 * torch.rand_like(params.TEd)
params.TEd.requires_grad = True

optimizable_params = [
    {'params': params.refoc_pulse_angles, 'lr': 0.03},
    {'params': params.refoc_pulse_phases, 'lr': 0.03},
    {'params': params.excit_pulse_angle, 'lr': 0.03},
    {'params': params.excit_pulse_phase, 'lr': 0.03},
    {'params': params.spoiler, 'lr': 4},
    # {'params': params.TEd, 'lr': 0.03},
    {'params': params.TEd, 'lr': 0.001},
    {'params': params.adc_time, 'lr': 0.03},

    # {'params': params.TEd_excit, 'lr': 0.03},
]

iteration = 0
for restart in range(1):
    optimizer = torch.optim.Adam(optimizable_params)
    # optimizer = torch.optim.SGD(optimizable_params, lr=0.03)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4.0, total_steps=1000)

    for i in range((restart + 1) * 301):
        iteration += 1

        optimizer.zero_grad()
        loss = calc_loss(params, iteration)
        loss.backward()
        optimizer.step()
        # scheduler.step()

imageio.mimsave(r"./out/TSE.gif", gif_array, format='GIF', duration=0.04)

# %% Plotting optimization
cmap = plt.get_cmap('viridis')
hist = torch.stack(TEd_hist).abs()

plt.figure(figsize=(10, 10))
plt.subplot(211)
for i in range(hist.shape[1]):
	plt.plot(hist[:, i] * 1000, c=cmap(i / 65))
plt.axhline(3.8, c='k', label="Target (3.8 ms)")
plt.xlabel("Iteration")
plt.ylabel("Timing [ms]")
plt.grid()
plt.legend()
plt.subplot(212)
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss [arbitrary units]")
plt.grid()
plt.show()
