from __future__ import annotations
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import new_core.sequence as Seq
from new_core import util


class TSE2D_seg:
    """Stores all parameters needed to create a 2D TSE sequence."""

    def __init__(self, adc_count: int, rep_count: int, R_accel: (int,int) = (1,1)):
        """Initialize parameters with default values."""
        self.R_accel = R_accel
        self.adc_count = adc_count
        self.event_count = adc_count + 5
        self.rep_count = rep_count // R_accel[0]

        self.excit_pulse_angle = torch.tensor(90 * np.pi / 180)
        self.excit_pulse_phase = torch.tensor(90 * np.pi / 180)

        self.refoc_pulse_angles = torch.full((rep_count, ), 180 * np.pi / 180)
        self.refoc_pulse_phases = torch.zeros((rep_count, ))

        self.TEd = torch.tensor(3.8e-3)
        self.TR = torch.tensor(1)
        self.adc_time = torch.tensor(0.06e-3)
        self.spoiler = torch.full((rep_count + 1, ), float(adc_count))

        self.time_scale = torch.tensor(1.0)

    def clone(self) -> TSE2D_seg:
        """Create a copy with cloned tensors."""
        clone = TSE2D_seg(self.adc_count, self.rep_count)

        clone.excit_pulse_angle = self.excit_pulse_angle.clone()
        clone.excit_pulse_phase = self.excit_pulse_phase.clone()
        clone.refoc_pulse_angles = self.refoc_pulse_angles.clone()
        clone.refoc_pulse_phases = self.refoc_pulse_phases.clone()
        clone.TEd = self.TEd.clone()
        clone.TR = self.TR.clone()
        clone.adc_time = self.adc_time.clone()
        clone.spoiler = self.spoiler.clone()

        return clone

    def generate_sequence(self, encoding_scheme='centric', turbo_factor = 1,
                          remove_p_enc=False, remove_f_enc=False) -> Seq.Sequence:
        """Generate a GRE sequence based on the given parameters."""
        seq = Seq.Sequence()
        n_seg = self.rep_count // turbo_factor
        # Excitation - no readout in this repetition
        for n in range(n_seg):
            rep = Seq.Repetition.zero(2)
            seq.append(rep)
    
            rep.pulse.angle = self.excit_pulse_angle
            rep.pulse.phase = self.excit_pulse_phase
            rep.pulse.usage = Seq.PulseUsage.EXCIT
    
            rep.event_time[0] = 2.0e-3  # Pulse
            rep.event_time[1] = self.TEd + 1.5e-3 + (self.adc_count * self.adc_time) / 2
            rep.gradm[1, 0] = 0.5 * self.adc_count + self.spoiler[0]
            rep.event_time *= self.time_scale
            
            if remove_f_enc:
                rep.gradm[:, 0] = 0        
    
            for r in range(turbo_factor*n,turbo_factor*(n+1)*self.R_accel[0],self.R_accel[0]):
                rep = Seq.Repetition.zero(self.event_count)
                seq.append(rep)
    
                rep.pulse.angle = self.refoc_pulse_angles[r]
                rep.pulse.phase = self.refoc_pulse_phases[r]
                rep.pulse.usage = Seq.PulseUsage.REFOC
    
                rep.event_time[0] = 1.5e-3  # Pulse
                rep.event_time[1] = self.TEd
                rep.event_time[2] = 2e-3  # Winder
                rep.event_time[3:-2] = self.adc_time  # Readout
                rep.event_time[-1] = self.TEd
                rep.event_time[-2] = 3.50e-3
    
                rep.event_time *= self.time_scale
    
    
                rep.gradm[2, 0] = self.spoiler[r] - 1
                if encoding_scheme == 'linear':
                    rep.gradm[2, 1] = r - self.rep_count*self.R_accel[0]/2
                elif encoding_scheme == 'centric':
                    if (r//self.R_accel[0]) % 2 == 0:
                        rep.gradm[2, 1] = r/2
                    else:
                        rep.gradm[2, 1] = -(r+1*self.R_accel[0])/2
    
                rep.gradm[3:-2, 0] = 1
                rep.gradm[-2, 0] = self.spoiler[r+1] + 1
                rep.gradm[-2, 1] = -rep.gradm[2, 1]
    
                if remove_p_enc:
                    rep.gradm[:, 1] = 0
    
                if remove_f_enc:
                    rep.gradm[:, 0] = 0
    
                rep.adc_usage[3:-2] = 1
                rep.gradm = rep.gradm
            rep.event_time[-1] = self.TR
        return seq

    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name) -> TSE2D_seg:
        with open(file_name, 'rb') as file:
            return pickle.load(file)


def plot_optimization_progress(
    reco: torch.Tensor, reco_target: torch.Tensor,
    params: TSE2D_seg, params_target: TSE2D_seg,
    kspace_trajectory: list[torch.Tensor], loss_history: list[float],
    loss_axis: str = 'log',  # 'log' or 'linear'
    figsize: tuple[float, float] = (10, 10), dpi: float = 180
) -> np.ndarray:
    """
    Plot a picture containing the most important sequence properties.

    This function also returns the plotted image as array for gif creation.
    """
    plt.figure(figsize=figsize)
    reco_max = max(np.abs(util.to_numpy(reco.squeeze())).max(),
                   np.abs(util.to_numpy(reco_target.squeeze())).max())
    plt.subplot(3, 2, 1)
    plt.imshow(np.abs(util.to_numpy(reco.squeeze())), vmin=0, vmax=reco_max)
    plt.colorbar()
    plt.title("Reco")
    plt.subplot(3, 2, 3)
    plt.imshow(np.abs(util.to_numpy(reco_target.squeeze())), vmin=0, vmax=reco_max)
    plt.colorbar()
    plt.title("Target")

    def convert(x: torch.Tensor):
        return np.mod(np.abs(util.to_numpy(x)) * 180 / np.pi, 360)

    refoc_rep = range(1, 1 + params.refoc_pulse_angles.numel())

    plt.subplot(3, 2, 2)
    plt.title("Flip Angles")
    plt.plot([0], convert(params.excit_pulse_angle), 'Db')
    plt.plot([0], convert(params_target.excit_pulse_angle), 'Dr')
    plt.plot(refoc_rep, convert(params.refoc_pulse_angles), '.b')
    plt.plot(refoc_rep, convert(params_target.refoc_pulse_angles), '.r')
    plt.ylim(bottom=0)
    plt.subplot(3, 2, 4)
    plt.title("Phase")
    plt.plot([0], convert(params.excit_pulse_phase), 'Db')
    plt.plot([0], convert(params_target.excit_pulse_phase), 'Dr')
    plt.plot(refoc_rep, convert(params.refoc_pulse_phases), '.b')
    plt.plot(refoc_rep, convert(params_target.refoc_pulse_phases), '.r')

    plt.subplot(3, 2, 5)
    plt.plot(loss_history)
    if loss_axis == 'log':
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
