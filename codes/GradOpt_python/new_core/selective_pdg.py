from __future__ import annotations

from typing import Callable, Any, Optional

import numpy as np
import torch

from . import util
from .sequence import Sequence
from .sim_data import SimData


# Expected signature for all functions used for selection
SelectorFunc = Callable[[Any, Optional[torch.Tensor]], bool]


def is_any(_state, _trajectory: Optional[torch.Tensor]) -> bool:
    return True


def is_trans_mag(state, _trajectory: Optional[torch.Tensor]) -> bool:
    return state.dist_type == '+'


def is_long_mag(state, _trajectory: Optional[torch.Tensor]) -> bool:
    return state.dist_type in ['z', 'z0']


def is_z0(state, _trajectory: Optional[torch.Tensor]) -> bool:
    return state.dist_type == 'z0'


def is_spin_echo(state, trajectory: Optional[torch.Tensor]) -> bool:
    if state.dist_type != '+' or trajectory is None:
        return False
    kt_t = state.kt_offset_vec[3]
    time = trajectory[:, 3].sum().item()
    return kt_t < 0 and kt_t + time > 0


# This class is used to simulate the remaining path after state selection
class State:
    def __init__(self, dist_type: str, mag: torch.Tensor,
                 kt_offset_vec: torch.Tensor) -> None:
        self.mag = mag.clone()
        self.kt_offset_vec = kt_offset_vec.clone()
        self.dist_type = dist_type

    def measure(self, trajectory: torch.Tensor, data: SimData,
                coil_sensitivity: torch.Tensor) -> torch.Tensor:
        dist_traj = self.kt_offset_vec + trajectory
        # shape: events x voxels
        transverse_mag = (
            self.mag.unsqueeze(0)
            * torch.exp(-trajectory[:, 3:] / data.T2)
            * torch.exp(-torch.abs(dist_traj[:, 3:]) / data.T2dash)
            * torch.exp(1j * (
                2*np.pi * dist_traj[:, 3:] * data.B0
                + -2*np.pi * dist_traj[:, :3] @ data.voxel_pos.t()))
        )
        transverse_mag *= (1.41421356237 * torch.prod(torch.sinc(
            dist_traj[:, :3] / data.shape
        ), dim=1)).unsqueeze(1)  # same for all voxels

        # (events x voxels) @ (voxels x coils) = (events x coils)
        return transverse_mag @ coil_sensitivity


# Recursive function that simulates path before measuring the state
# The state could split up, either by using 'e' on a + state (produces a
# z state pair), or if less strict paths are allowed in the future
def simulate_path(state: State, path: str, seq: Sequence,
                  coil_sensitivity: torch.Tensor, coil_count: int,
                  data: SimData) -> torch.Tensor:
    if len(path) > 0:
        # Update state (relaxation and dephasing)
        total_time = seq[0].event_time.sum()
        if state.dist_type == '+':
            state.mag = state.mag * torch.exp(-total_time / data.T2)
            state.kt_offset_vec[:3] += seq[0].gradm.sum(0)
            state.kt_offset_vec[3] += seq[0].event_time.sum()
        elif state.dist_type == 'z':
            state.mag = state.mag * torch.exp(-total_time / data.T1)
        elif state.dist_type == 'z0':
            state.mag = 1 + (state.mag - 1)*torch.exp(-total_time / data.T1)

        # Apply pulse (in its selected usage) and return simulate_path result
        # of the remaining path on the resulting state(s)

        # The pulse belongs to the next repetition
        angle = seq[1].pulse.angle * data.B1[0, :]
        phase = torch.as_tensor(seq[1].pulse.phase)

        z_to_z = util.set_device(torch.cos(angle))
        p_to_p = util.set_device(torch.cos(angle/2)**2)
        z_to_p = util.set_device(
            -0.70710678118j * torch.sin(angle) * torch.exp(1j*phase)
        )
        p_to_z = -z_to_p.conj()
        m_to_z = -z_to_p
        m_to_p = (1 - p_to_p) * torch.exp(2j*phase)

        params = [path[1:], seq[1:], coil_sensitivity, coil_count, data]

        if state.dist_type == '+':
            if path[0] == 'e':
                return (
                    simulate_path(State(
                        'z', state.mag * p_to_z, state.kt_offset_vec
                    ), *params)
                    + simulate_path(State(
                        'z', state.mag.conj() * m_to_z, -state.kt_offset_vec
                    ), *params)
                )
            elif path[0] == 'r':
                return simulate_path(State(
                    '+', state.mag.conj() * m_to_p, -state.kt_offset_vec
                ), *params)
            else:  # path[0] == 'u'
                return simulate_path(State(
                    '+', state.mag * p_to_p, state.kt_offset_vec
                ), *params)
        else:  # dist_type == 'z' or 'z0'
            if path[0] == 'e':
                return simulate_path(State(
                    '+', state.mag * z_to_p, state.kt_offset_vec
                ), *params)
            elif path[0] == 'r':
                # Can't refocus z magnetisation
                return torch.zeros(
                    seq[1].event_count, coil_count,
                    dtype=torch.cfloat, device=util.get_device())
            else:  # path[0] == 'u'
                return simulate_path(State(
                    'z', state.mag * z_to_z, state.kt_offset_vec
                ), *params)
    else:
        trajectory = util.set_device(torch.cat([
            torch.cumsum(seq[0].gradm, 0),
            torch.cumsum(seq[0].event_time, 0).unsqueeze(1)], 1))
        # We're done simulating the path and can return
        return state.measure(trajectory, data, coil_sensitivity)


def execute_graph_selective(graph: list[list],
                            seq: Sequence, data: SimData,
                            selection: SelectorFunc, path: str,
                            min_influence: float = 1e-3) -> torch.Tensor:
    """Calculate the signal of selected states only.

    The selection works by describing what previous pulses did to a specified
    type of state. Therefore ``selection`` is a function that determines what
    states to use and ``path`` then describes what transformations modified
    these states before they are measured.

    Possible values for specifying a path:
     - ``"e"``: Excitation, can stand for both ``z -> +`` and ``+ -> z``
     - ``"r"``: Refocusing, also works on both ``+`` and ``z`` states
     - ``"u"``: Unmodified, dominant at low flip angles

    Example selections:
     - ``is_spin_echo + ""``: Spin Echos
     - ``is_spin_echo + "ee"``: Stimulated echos
     - ``is_spin_echo + "uru"``: 2nd order echos
     - ``is_z0 + "e"``: Freshly excited magnetisation

    Parameters
    ----------
    graph : list[list[Distribution]]
        Distribution graph that will be executed.
    seq : Sequence
        Sequence that will be simulated and was used to create ``graph``.
    data : SimData
        Physical properties of phantom and scanner.
    selection : (State, Tensor (optional)) -> bool
        Function determining the measured states
    path : str
        Path these selected states follow before measurement
    min_influence : float
        Minimum influence of a state for it to be measured.

    Returns
    -------
    signal : torch.Tensor
        The simulated signal of the selected distributions.
    """
    # Proton density can be baked into coil sensitivity. shape: voxels x coils
    coil_sensitivity = (
        data.coil_sens.t().to(torch.cfloat)
        * data.PD.unsqueeze(1)/data.PD.sum()
    )
    coil_count = data.coil_count
    # Zero initialize signal
    signal: list[torch.Tensor] = []
    for rep in seq:
        signal.append(
            torch.zeros(rep.event_count, coil_count,
                        dtype=torch.cfloat, device=util.get_device())
        )

    # The first repetition contains only one element: A fully relaxed z0
    graph[0][0].mag = torch.ones(
        data.voxel_count, dtype=torch.cfloat, device=util.get_device())
    # Calculate kt_offset_vec ourselves for autograd
    graph[0][0].kt_offset_vec = torch.zeros(4, device=util.get_device())

    # Selected state might already apply to the z0 state before the first pulse
    if selection(graph[0][0], None):
        signal[len(path)] += simulate_path(
            State(graph[0][0].dist_type, graph[0][0].mag,
                  graph[0][0].kt_offset_vec),
            path, seq, coil_sensitivity, coil_count, data
        )

    for i, (dists, rep) in enumerate(zip(graph[1:len(graph)-len(path)], seq)):
        print(f"\rCalculating repetition {i+1} / {len(seq)}", end='')
        # Apply the pulse
        # Necessary elements of the pulse rotation matrix
        angle = rep.pulse.angle * data.B1[0, :]
        phase = torch.as_tensor(rep.pulse.phase)
        # Unaffected magnetisation
        z_to_z = util.set_device(torch.cos(angle))
        p_to_p = util.set_device(torch.cos(angle/2)**2)
        # Excited magnetisation
        z_to_p = util.set_device(
            -0.70710678118j * torch.sin(angle) * torch.exp(1j*phase)
        )
        p_to_z = -z_to_p.conj()
        m_to_z = -z_to_p
        # Refocussed magnetisation
        m_to_p = (1 - p_to_p) * torch.exp(2j*phase)

        def calc_mag(ancestor: tuple) -> torch.Tensor:
            if ancestor[0] == 'zz':
                return ancestor[1].mag * z_to_z
            elif ancestor[0] == '++':
                return ancestor[1].mag * p_to_p
            elif ancestor[0] == 'z+':
                return ancestor[1].mag * z_to_p
            elif ancestor[0] == '+z':
                return ancestor[1].mag * p_to_z
            elif ancestor[0] == '-z':
                return ancestor[1].mag.conj() * m_to_z
            elif ancestor[0] == '-+':
                return ancestor[1].mag.conj() * m_to_p
            else:
                raise ValueError(f"Unknown transform {ancestor[0]}")

        # shape: events x 4
        trajectory = torch.cat([
            torch.cumsum(util.set_device(rep.gradm), 0),
            torch.cumsum(util.set_device(rep.event_time), 0).unsqueeze(1)], 1)

        total_time = rep.event_time.sum()
        r1 = torch.exp(-total_time / data.T1)
        r2 = torch.exp(-total_time / data.T2)

        for dist in dists:
            # Create a list only containing ancestors that were simulated
            ancestors = list(filter(
                lambda edge: edge[1].mag is not None, dist.ancestors
            ))

            if dist.dist_type != 'z0' and dist.rel_influence < min_influence:
                continue  # skip unimportant distributions
            if dist.dist_type != 'z0' and len(ancestors) == 0:
                continue  # skip dists for which no ancestors were simulated

            dist.mag = sum([calc_mag(ancestor) for ancestor in ancestors])
            # The pre_pass already calculates kt_offset_vec, but that does not
            # work with autograd -> we need to calculate it with torch
            if dist.dist_type == 'z0':
                dist.kt_offset_vec = torch.zeros(4, device=util.get_device())
            elif ancestors[0][0] in ['-+', '-z']:
                dist.kt_offset_vec = -1.0 * ancestors[0][1].kt_offset_vec
            else:
                dist.kt_offset_vec = ancestors[0][1].kt_offset_vec.clone()

            if selection(dist, trajectory):
                signal[i+len(path)] += simulate_path(
                    State(dist.dist_type, dist.mag, dist.kt_offset_vec),
                    path, seq[i:], coil_sensitivity, coil_count, data
                )

            if dist.dist_type == '+':
                dist.mag = dist.mag * r2
            else:  # z or z0
                dist.mag = dist.mag * r1
            if dist.dist_type == 'z0':
                dist.mag = dist.mag + 1 - r1

            if dist.dist_type == '+':
                dist.kt_offset_vec += trajectory[-1]

        # Remove ancestors to save memory as we don't need them anymore.
        # When running with autograd gradients this doesn't change memory
        # consumption bc. the values are still stored in the computation graph.
        for dist in dists:
            for ancestor in dist.ancestors:
                ancestor[1].mag = None

    print(" - done", end='')
    # Only return measured samples
    return torch.cat([
        (
            sig * torch.exp(1j * rep.adc_phase).unsqueeze(1)
        )[rep.adc_usage > 0, :] for sig, rep in zip(signal, seq)
    ])
