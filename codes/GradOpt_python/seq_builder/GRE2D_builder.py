from __future__ import annotations
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import new_core.sequence as Seq
from new_core import util


class GRE2D:
    """Stores all parameters needed to create a 2D GRE sequence."""

    def __init__(self, adc_count: int, rep_count: int):
        """Initialize parameters with default values."""

        # Centric reordering
        self.gradm_phase = torch.tensor(
            [(i//2) if i % 2 == 0 else (-1 - i//2) for i in range(rep_count)])
        # Linear reordering
        # self.gradm_phase = torch.arange(-rep_count/2, rep_count/2)

        self.adc_count = adc_count
        self.event_count = adc_count + 4
        self.rep_count = rep_count

        self.spoiler_time = torch.full((rep_count, ), 2e-3)

        self.pulse_angles = torch.full((rep_count, ), 5 * np.pi / 180)
        self.pulse_phases = torch.tensor(
            [util.phase_cycler(r) for r in range(rep_count)])
        self.gradm_rewinder = torch.full((rep_count, ), -adc_count/2)
        self.gradm_adc = torch.full((rep_count, ), 1.0)
        self.gradm_spoiler = torch.full((rep_count, ), 0.5 * adc_count)
        self.gradm_spoiler_phase = -self.gradm_phase

    def clone(self) -> GRE2D:
        """Create a copy with cloned tensors."""
        clone = GRE2D(self.adc_count, self.rep_count)

        clone.pulse_angles = self.pulse_angles.clone()
        clone.pulse_phases = self.pulse_phases.clone()
        clone.gradm_rewinder = self.gradm_rewinder.clone()
        clone.gradm_phase = self.gradm_phase.clone()
        clone.gradm_adc = self.gradm_adc.clone()
        clone.gradm_spoiler = self.gradm_spoiler.clone()
        clone.gradm_spoiler_phase = self.gradm_spoiler_phase.clone()

        return clone

    def generate_sequence(self) -> Seq.Sequence:
        """Generate a GRE sequence based on the given parameters."""
        seq = Seq.Sequence()

        for r in range(self.rep_count):
            # extra events: pulse + winder + rewinder
            rep = seq.new_rep(self.event_count)

            rep.pulse.angle = self.pulse_angles[r]
            rep.pulse.phase = self.pulse_phases[r]
            rep.pulse.usage = Seq.PulseUsage.EXCIT

            rep.event_time[0] = 2e-3  # Pulse
            rep.event_time[1] = 2e-3  # Winder
            rep.event_time[2:-2] = 0.08e-3  # Readout

            rep.gradm[1, 0] = self.gradm_rewinder[r]
            rep.gradm[1, 1] = self.gradm_phase[r]
            rep.gradm[2:-2, 0] = self.gradm_adc[r]

            # Rewinder / Spoiler, centers readout in rep
            rep.event_time[-1] = 2e-3
            rep.event_time[-2] = self.spoiler_time[r]

            rep.gradm[-2, 0] = self.gradm_spoiler[r]
            rep.gradm[-2, 1] = self.gradm_spoiler_phase[r]

            rep.adc_usage[2:-2] = 1
            rep.adc_phase[2:-2] = np.pi/2 - rep.pulse.phase

        return seq

    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name) -> GRE2D:
        with open(file_name, 'rb') as file:
            return pickle.load(file)


def plot_optimization_progress(
    reco: torch.Tensor, reco_target: torch.Tensor,
    params: GRE2D, params_target: GRE2D,
    kspace_trajectory: list[torch.Tensor], loss_history: list[float],
    figsize: tuple[float, float] = (10, 10), dpi: float = 180
) -> np.ndarray:
    """
    Plot a picture containing the most important sequence properties.

    This function also returns the plotted image as array for gif creation.
    """
    plt.figure(figsize=figsize)
    reco_max = max(np.abs(util.to_numpy(reco[:, :, 0])).max(),
                   np.abs(util.to_numpy(reco_target[:, :, 0])).max())
    plt.subplot(3, 2, 1)
    plt.imshow(np.abs(util.to_numpy(reco[:, :, 0])), vmin=0, vmax=reco_max)
    plt.colorbar()
    plt.title("Reco")
    plt.subplot(3, 2, 3)
    plt.imshow(np.abs(util.to_numpy(reco_target[:, :, 0])), vmin=0, vmax=reco_max)
    plt.colorbar()
    plt.title("Target")

    plt.subplot(3, 2, 2)
    plt.plot(np.abs(util.to_numpy(params.pulse_angles)) * 180 / np.pi, '.')
    plt.plot(util.to_numpy(params_target.pulse_angles) * 180 / np.pi, '.', color='r')
    plt.title("Flip Angles")
    plt.ylim(bottom=0)
    plt.subplot(3, 2, 4)
    plt.plot(np.mod(np.abs(util.to_numpy(params.pulse_phases)) * 180 / np.pi, 360), '.')
    plt.plot(np.mod(np.abs(util.to_numpy(params_target.pulse_phases)) * 180 / np.pi, 360), '.', color='r')
    plt.title("Phase")

    plt.subplot(3, 2, 5)
    plt.plot(loss_history)
    plt.yscale('log')
    plt.grid()
    plt.title("Loss Curve")

    plt.subplot(3, 2, 6)
    for i, rep_traj in enumerate(kspace_trajectory):
        kx = util.to_numpy(rep_traj[:, 0]) / (2*np.pi)
        ky = util.to_numpy(rep_traj[:, 1]) / (2*np.pi)
        plt.plot(kx, ky, c=cm.rainbow(i / len(kspace_trajectory)))
        plt.plot(kx, ky, 'k.')
    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")
    plt.grid()

    img = util.current_fig_as_img(dpi)
    plt.show()
    return img
