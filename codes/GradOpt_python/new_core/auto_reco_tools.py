"""The contents of this file are subject to change."""
import torch
import numpy as np
from . import util


def create_encoded_signal(seq, dist_signals):
    # Masks by adc and removes phase info, for use in multi_trajectory_adjoint
    encoded_signal = []

    for seq_rep, sig_rep in zip(seq, dist_signals):
        encoded_signal_rep = []
        encoded_signal.append(encoded_signal_rep)

        for dist_sig, kspace, phase in sig_rep:
            dist_sig = dist_sig * np.exp(1j*(np.pi/2 - phase))
            dist_sig[seq_rep.adc_usage == 0] = 0
            encoded_signal_rep.append((dist_sig, kspace))

    return encoded_signal


def generate_signal_estimator(dist_signals, measured_signal):
    # Both arguments are lists of repetitions
    # dist_signals contains signal trajectory pairs
    estimation = []

    for rep_dist_sigs, rep_measured_sig in zip(dist_signals, measured_signal):
        rep_estimation = []
        estimation.append(rep_estimation)

        for dist_signal, trajectory, phase in rep_dist_sigs:
            rel_sig = dist_signal / rep_measured_sig
            mean_rel_sig = rel_sig[rep_measured_sig != 0].mean().item()

            rep_estimation.append((
                mean_rel_sig * np.exp(1j*(np.pi/2 - phase)),
                trajectory
            ))

    def estimator(measured_signal):
        encoded_signal = []

        for est_rep, sig_rep in zip(estimation, measured_signal):
            encoded_signal_rep = []
            encoded_signal.append(encoded_signal_rep)

            for factor, kspace in est_rep:
                encoded_signal_rep.append((factor * sig_rep, kspace))

        return encoded_signal

    return estimator


def multi_trajectory_adjoint(encoded_signal, resolution):
    """Reconstructs the image based on all trajectories signal pairs provided.

    ``encoded_signal`` is expected to be a list of repetition:
    every repetition is a list of (signal, kspace trajectory) tuples
    """
    # TODO: This should really include a kernel density estimator to compensate
    # k-space sample inhomogeneities

    # calculate voxel size
    vs_x = 1 / float(resolution[0])
    vs_y = 1 / float(resolution[1])
    vs_z = 1 / float(resolution[2])
    # Same grid as FFT expects, e.g. positions at a resolution of 4 are
    # [-0.25, 0.0, 0.25, 0.5]
    pos_x = torch.linspace(-0.5 + vs_x, +0.5, resolution[0])
    pos_y = torch.linspace(-0.5 + vs_y, +0.5, resolution[1])
    pos_z = torch.linspace(-0.5 + vs_z, +0.5, resolution[2])
    pos_x, pos_y, pos_z = torch.meshgrid(pos_x, pos_y, pos_z)

    voxel_pos = util.set_device(torch.stack([
        pos_x.flatten(),
        pos_y.flatten(),
        pos_z.flatten()
    ], dim=1)).t()

    image = torch.zeros(resolution,
                        dtype=torch.cfloat, device=util.get_device())

    for rep in encoded_signal:
        for signal, kspace in rep:
            # (Events, 3) x (3, Voxels)
            phase = kspace @ voxel_pos
            # (Events, Voxels): Inverse Phase of voxels given by gradients
            inv_rot = torch.exp(-1j * phase)
            # (Coils x Events) x (Events, Voxels)
            image += (signal.t() @ inv_rot).sum(0).view(resolution)

    return image.cpu()
