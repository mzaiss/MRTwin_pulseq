from __future__ import annotations
import torch
import matplotlib.pyplot as plt

from .sequence import Sequence
from .sim_data import SimData, RawSimData
import numpy as np
from . import util


def execute_graph(graph: list[list],
                  seq: Sequence,
                  data: SimData | RawSimData,
                  min_signal: float = 1e-2,
                  min_weight: float = 1e-3,
                  return_mag_p: int | bool | None = None,
                  return_mag_z: int | bool | None = None,
                  return_encoding: bool = False
                  ) -> torch.Tensor | list:
    """Calculate the signal of the sequence by computing the graph.

    This function can optionally return the + or z magnetisation and the
    encoding of all states, if requested. The encoding consists of the signal
    of a distribution and its k-t space trajectory.

    Warnings
    --------
    If ``return_mag_(p/z)`` or ``return_encoding`` are set, this
    simulation returns additional information that a real measurement cannot
    provide. The optimized / trained sequence and reconstruction should not
    depend on this data for a specific phantom.

    Parameters
    ----------
    graph : list[list[Distribution]]
        Distribution graph that will be executed.
    seq : Sequence
        Sequence that will be simulated and was used to create ``graph``.
    data : SimData
        Physical properties of phantom and scanner.
    min_signal : float
        Minimum relative signal of a state for it to be measured.
    min_weight : float
        Minimum "weight" metric of a state for it to be simulated. Should be
        less than min_signal.
    return_mag_p : int or bool, optional
        If set, returns the transverse magnetisation of either the given
        repetition (int) or all repetitions (``True``).
    return_mag_z : int or bool, optional
        If set, returns the longitudinal magnetisation of either the given
        repetition (int) or all repetitions (``True``).
    return_encoding : bool
        If set, returns the encoding of all measured states.

    Returns
    -------
    signal : torch.Tensor
        The simulated signal of the sequence.
    mag_p : torch.Tensor | list[torch.Tensor]
        The transverse magnetisation of the specified or all repetition(s).
    mag_z : torch.Tensor | list[torch.Tensor]
        The longitudinal magnetisation of the specified or all repetition(s).
    encoding : list[list[tuple[torch.Tensor, torch.Tensor, float]]]
        (signal, trajectory, phase) of all distributions of all repetitions.
    """
    k_to_si = 2*np.pi / data.fov
    signal: list[torch.Tensor] = []
    # Only filled and returned if requested
    mag_p = []
    mag_z = []
    encoding = []

    # Proton density can be baked into coil sensitivity. shape: voxels x coils
    coil_sensitivity = (
        data.coil_sens.t().to(torch.cfloat)
        * torch.abs(data.PD).unsqueeze(1)#/torch.abs(data.PD).sum()
    )
    coil_count = int(coil_sensitivity.shape[1])
    voxel_count = data.PD.numel()

    # The first repetition contains only one element: A fully relaxed z0
    graph[0][0].mag = torch.ones(
        voxel_count, dtype=torch.cfloat, device=util.get_device()
    )
    # Calculate kt_vec ourselves for autograd
    graph[0][0].kt_vec = torch.zeros(4, device=util.get_device())

    for i, (dists, rep) in enumerate(zip(graph[1:], seq)):
        print(f"\rCalculating repetition {i+1} / {len(seq)}", end='')
        # Apply the pulse
        # Necessary elements of the pulse rotation matrix
        angle = rep.pulse.angle * torch.abs(data.B1[0, :])
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

        # shape: events x coils
        rep_sig = torch.zeros(rep.event_count, coil_count,
                              dtype=torch.cfloat, device=util.get_device())

        # shape: events x 4
        trajectory = util.set_device(torch.cumsum(torch.cat([
            rep.gradm, rep.event_time[:, None]
        ], 1), 0))
        dt = util.set_device(rep.event_time)

        total_time = rep.event_time.sum()
        r1 = torch.exp(-total_time / torch.abs(data.T1))
        r2 = torch.exp(-total_time / torch.abs(data.T2))

        mag_p_rep = []
        mag_z_rep = []
        encoding_rep = []
        # Use the same adc phase for all coils
        adc_rot = torch.exp(1j * rep.adc_phase).unsqueeze(1)

        for dist in dists:
            # Create a list only containing ancestors that were simulated
            ancestors = list(filter(
                lambda edge: edge[1].mag is not None, dist.ancestors
            ))

            if dist.dist_type != 'z0' and dist.weight < min_weight:
                continue  # skip unimportant distributions
            if dist.dist_type != 'z0' and len(ancestors) == 0:
                continue  # skip dists for which no ancestors were simulated

            dist.mag = sum([calc_mag(ancestor) for ancestor in ancestors])
            if dist.dist_type in ['z0', 'z'] and return_mag_z in [i, True]:
                mag_z_rep.append(dist.mag)
            # The pre_pass already calculates kt_vec, but that does not
            # work with autograd -> we need to calculate it with torch
            if dist.dist_type == 'z0':
                dist.kt_vec = torch.zeros(4, device=util.get_device())
            elif ancestors[0][0] in ['-+', '-z']:
                dist.kt_vec = -1.0 * ancestors[0][1].kt_vec
            else:
                dist.kt_vec = ancestors[0][1].kt_vec.clone()

            # shape: events x 4
            dist_traj = dist.kt_vec + trajectory

            # Diffusion
            k2 = dist_traj[:, :3] * k_to_si
            k1 = torch.empty_like(k2)  # Calculate k-space at start of event
            k1[0, :] = dist.kt_vec[:3] * k_to_si
            k1[1:, :] = k2[:-1, :]
            # Integrate over each event to get b factor (lin. interp. grad)
            b = 1/3 * dt * (k1**2 + k1*k2 + k2**2).sum(1)
            # shape: events x voxels
            diffusion = torch.exp(-1e-9 * data.D * torch.cumsum(b, 0)[:, None])

            # NOTE: We are calculating the signal for samples that are not
            # measured (adc_usage == 0), which is, depending on the sequence,
            # produces an overhead of ca. 5 %. On the other hand, this makes
            # the code much simpler bc. we only have to apply the adc mask
            # once at the end instead of for every input. Change only if this
            # performance improvement is worth it. Repetitions without any adc
            # are already skipped because the pre-pass returns no signal.

            if dist.dist_type == '+' and dist.rel_signal >= min_signal:
                # shape: events x voxels
                transverse_mag = (
                    dist.mag.unsqueeze(0)
                    * torch.exp(-trajectory[:, 3:] / torch.abs(data.T2))
                    * torch.exp(-torch.abs(dist_traj[:, 3:]) / torch.abs(data.T2dash))
                    * torch.exp(1j * (
                        2*np.pi * dist_traj[:, 3:] * data.B0
                        # This gradient definition matches DFT
                        + -2*np.pi * dist_traj[:, :3] @ data.voxel_pos.t()))
                    * diffusion
                )

                transverse_mag *= (
                    1.41421356237 * data.dephasing_func(dist_traj[:, :3])
                ).unsqueeze(1)

                # PD is encoded in coil_sensitivity which we don't include in
                # the transverse magnetisation, so apply it separately here
                if return_mag_p in [i, True]:
                    mag_p_rep.append(adc_rot * transverse_mag * torch.abs(data.PD))

                # (events x voxels) @ (voxels x coils) = (events x coils)
                dist_signal = transverse_mag @ coil_sensitivity
                rep_sig += dist_signal

                if return_encoding:
                    phase = np.angle(dist.prepass_mag)
                    encoding_rep.append([dist_signal, dist_traj, phase])

            if dist.dist_type == '+':
                # Diffusion for whole trajectory + T2 relaxation
                dist.mag = dist.mag * r2 * diffusion[-1, :]
                dist.kt_vec = dist_traj[-1]
            else:  # z or z0
                k = torch.linalg.vector_norm(dist.kt_vec[:3] * k_to_si)
                diffusion = torch.exp(-1e-9 * data.D * total_time * k**2)
                dist.mag = dist.mag * r1 * diffusion
            if dist.dist_type == 'z0':
                dist.mag = dist.mag + 1 - r1

        rep_sig *= adc_rot

        mag_p.append(mag_p_rep)
        mag_z.append(mag_z_rep)
        encoding.append(encoding_rep)

        # Remove ancestors to save memory as we don't need them anymore.
        # When running with autograd this doesn't change memory consumption
        # bc. the values are still stored in the computation graph.
        for dist in dists:
            for ancestor in dist.ancestors:
                ancestor[1].mag = None

        signal.append(rep_sig)

    print(" - done")

    # Only return measured samples
    measurement = torch.cat([
        sig[rep.adc_usage > 0, :] for sig, rep in zip(signal, seq)
    ])

    if return_mag_p is None and return_mag_z is None and not return_encoding:
        return measurement
    else:
        tmp = [measurement]

        if return_mag_p is not None:
            tmp.append(mag_p[0] if len(mag_p) == 1 else mag_p)
        if return_mag_z is not None:
            tmp.append(mag_z[0] if len(mag_z) == 1 else mag_z)
        if return_encoding:
            tmp.append((signal, encoding))

        return tmp


def signal_hist_sim(
    graph: list[list],
    seq: Sequence,
    T1: float,
    T2: float,
    T2dash: float,
    min_signal: float = 1e-3,
    min_influence: float = 1e-3,
):
    """Execute the graph but calculate history information instead of a signal.

    This function works similar to ``execute_graph`` but instead of simulating
    a spatially resolved magnetisation and returning a signal that can be used
    for reconstruction, a dummy spatial distribution is used and the signal is
    summed per repetition. This allows to instead store history information
    about the magnetisation of every state: what percentage of the mag was
    refocused or excited 0, 1, 2, ... times. This is then summed over all
    states and returned per repetition. The resulting tensor tells, if the
    signal originates from FID, higher order echoes, stimulated echoes etc...

    The way this works is by starting with all magnetisation at [0, 0] in the
    per-state magnetisation tensor. When applying a pulse, the newly created
    excited state has its magnetisation shifted 1 along the first index. New
    refocused states are shifted 1 along the second index, the third unchanged
    states is not shifted. During the simulation, states can be combined as
    usual and the information about what happened to the magnetisation is still
    available. The signal is calculated as usual per-state, but instead of
    summing over the magnetisation, the full history information is returned.

    Parameters
    ----------
    graph : list[list[Distribution]]
        Distribution graph that will be executed.
    seq : Sequence
        Sequence that will be simulated and was used to create ``graph``.
    T1 : float
        Average T1 relaxation time (seconds)
    T2 : float
        Average T2 relaxation time (seconds)
    T2dash : float
        Average T2' dephasing time (seconds)
    min_signal : float
        Minimum signal of a state for it to be measured.
    min_influence : float
        Minimum influence of a state for it to be measured.

    Returns
    -------
    torch.Tensor
        A tensor of shape len(seq)^3 that contains the sum of the magnitude
        of the signal of a repetition. Indices are [repetition, # of
        excitations, # refocusings]
    """

    print("!!! This simulation currently ignores diffusion !!!")
    print("Also, update for weight metric & renamed properties")

    size = len(seq)
    dev = util.get_device()
    # repetition, # of excits, # of refocs
    signal = torch.zeros(size, size, size, device=dev)

    graph[0][0].mag = torch.zeros(size, size, dtype=torch.cfloat, device=dev)
    graph[0][0].mag[0, 0] = 1
    graph[0][0].kt_offset_vec = torch.zeros(4, device=dev)

    for i, (dists, rep) in enumerate(zip(graph[1:], seq)):
        print(f"\rCalculating repetition {i+1} / {len(seq)}", end='')
        # Apply the pulse
        # Necessary elements of the pulse rotation matrix
        angle = torch.as_tensor(rep.pulse.angle)
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
            # This function shifts along 1st / 2nd axis at every excite / refoc
            # such that the magnetisation tensors store the history of the mag
            # instead of spatial information
            if ancestor[0] == 'zz':
                return ancestor[1].mag * z_to_z
            elif ancestor[0] == '++':
                return ancestor[1].mag * p_to_p
            elif ancestor[0] == 'z+':
                return torch.roll(ancestor[1].mag * z_to_p, 1, 0)
            elif ancestor[0] == '+z':
                return torch.roll(ancestor[1].mag * p_to_z, 1, 0)
            elif ancestor[0] == '-z':
                return torch.roll(ancestor[1].mag.conj() * m_to_z, 1, 0)
            elif ancestor[0] == '-+':
                return torch.roll(ancestor[1].mag.conj() * m_to_p, 1, 1)
            else:
                raise ValueError(f"Unknown transform {ancestor[0]}")

        # shape: events x 4
        trajectory = torch.cat([
            torch.cumsum(util.set_device(rep.gradm), 0),
            torch.cumsum(util.set_device(rep.event_time), 0).unsqueeze(1)], 1)

        total_time = rep.event_time.sum()
        r1 = torch.exp(-total_time / T1)
        r2 = torch.exp(-total_time / T2)

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

            if dist.dist_type == 'z0':
                dist.kt_offset_vec = torch.zeros(4, device=util.get_device())
            elif ancestors[0][0] in ['-+', '-z']:
                dist.kt_offset_vec = -1.0 * ancestors[0][1].kt_offset_vec
            else:
                dist.kt_offset_vec = ancestors[0][1].kt_offset_vec.clone()

            if dist.dist_type == '+' and dist.prepass_rel_signal >= min_signal:
                # shape: events x 3
                dist_traj = dist.kt_offset_vec + trajectory

                # For spatial information we use a cauchy distribution
                norm = torch.linalg.norm(dist_traj[:, :3], dim=1)
                attenuation = (
                    1.41421356237
                    * torch.exp(-trajectory[:, 3] / T2)
                    * torch.exp(-dist_traj[:, 3].abs() / T2dash)
                    * torch.exp(-0.1 * norm)  # FT of Cauchy
                )[rep.adc_usage > 0].sum()
                signal[i, :, :] += attenuation * dist.mag.abs()

            if dist.dist_type == '+':
                dist.mag = dist.mag * r2
            else:  # z or z0
                dist.mag = dist.mag * r1
            if dist.dist_type == 'z0':
                dist.mag[0, 0] += 1 - r1

            if dist.dist_type == '+':
                dist.kt_offset_vec += trajectory[-1]

        # Remove the calculated magnetisations, otherwise there are conflicts
        # executing the normal and this simulation on the same graph
        for dist in dists:
            for ancestor in dist.ancestors:
                ancestor[1].mag = None

    print(" - done")
    return signal


def plot_states(
    graph: list[list],
    seq: Sequence,
    typ: str = "+",
    prop: str = "mag",
    comp: int = 3,
    log: bool = False,
    figsize: tuple[int, int] = (7, 6)
):
    """Plot the specified property of all + or z states in a single image.

    Parameters
    ----------
    graph : list[list[Distribution]]
        Graph containing the distributions that will be plotted
    seq : Sequence
        Sequence that produced the graph (needed because the graph stores the
        dephasing at the end of the repetition but we want at the beginning)
    typ : str
        Type of distributions that will be plotted, can be + or z
    prop : str
        Distribution property that will be plotted as color. can be one of
        ``mag``, ``signal``, ``rel_signal``, ``influence`` or ``rel_influence``
    comp : int
        kt vector component that is plotted along the y-axis, can be 0, 1, 2
        for x, y, z or 3 for time (T2' dephasing)
    log : bool
        If set, the selected property will be plotted logarithmic
    figsize : (int, int)
        Figure size used for matplotlib plot
    """
    # Labels for plotting
    titles = {
        "mag": "Magnetisation",
        "signal": "Signal",
        "rel_signal": "Relative Signal",
        "influence": "Influence",
        "rel_influence": "Relative Influence"
    }
    ylabels = [
        "$k_x$ Dephasing [1/FOV]",
        "$k_y$ Dephasing [1/FOV]",
        "$k_z$ Dephasing [1/FOV]",
        "$T_2'$ Dephasing [s]"
    ]
    prefix = "Logarithmic " if log else ""

    def dist_prop(d):
        return {
            "mag": np.abs(d.prepass_mag),
            "signal": d.prepass_signal,
            "rel_signal": d.prepass_rel_signal,  # Relative to max of rep
            "influence": d.influence,
            "rel_influence": d.rel_influence,  # Relative to max of whole sim
        }[prop]

    x_list = []
    y_list = []
    c_list = []
    for r, rep in enumerate(graph[1:]):
        rep_kt = [
            *seq[r].gradm.sum(0).cpu().tolist(),
            seq[r].event_time.sum().cpu()
        ]
        for dist in rep:
            dtyp = dist.dist_type
            if dtyp == typ or (typ == "z" and dtyp == "z0"):
                kt = dist.prepass_kt_vec[comp]
                if typ == "+":
                    kt -= rep_kt[comp]
                x_list.append(r)
                y_list.append(float(kt))
                c_list.append(dist_prop(dist))

    # Convert y to bins / indices
    reps = len(graph) - 1
    bins = (len(graph) - 1) // 2 * 2 + 1  # make it odd so 0.0 is in the center
    y_min = min(y_list)
    y_max = max(y_list)
    y_ext = max(abs(y_min), abs(y_max)) + 1e-6  # epsilon for exclusive range
    # Map from range (-y_ext, y_ext) to (0, bins-1)
    y_list = [round((y+y_ext) / (2*y_ext) * (bins - 1)) for y in y_list]

    # Plot
    img = np.zeros((bins, reps))
    for x, y, c in zip(x_list, y_list, c_list):
        img[y, x] = c
    if log:
        img = np.log(img)

    plt.figure(figsize=figsize)
    plt.imshow(
        img,
        extent=[1, reps, -y_ext, y_ext],
        aspect="auto",
        origin="lower"
    )
    plt.colorbar()
    plt.title(f"{prefix}{titles[prop]} of {typ} States")
    plt.xlabel("Repetition")
    plt.ylabel(ylabels[comp])
    plt.show()


def reset_kt_offset_vec(graph: list[list]):
    """
    Set all kt_offset_vec's of the graph to None.

    This allows determining which distributions were used during simulation.
    """
    for rep in graph:
        for dist in rep:
            dist.kt_offset_vec = None


def plot_graph(graph: list[list]):
    """
    Plot distributions which have a kt_offset_vector unequal None.

    If this attribute is reset before simulation (e.g. by using
    :func:`reset_kt_offset_vec`), only distributions that were simulated
    afterwards have this attribute set and will be plotted.
    """
    def to_list(x):
        if x is None:
            return None
        elif torch.is_tensor(x):
            return x.tolist()
        else:
            return x

    reps = len(graph)
    plt.figure(figsize=(10, 10))

    # Plot + distributions
    plt.subplot(211)
    plt.title("+ Distributions")
    plt.xlabel("Repetition")
    plt.ylabel("$T_2'$ Dephasing [s]")
    plt.grid()
    plt.xlim([-0.1 * reps, 1.1 * reps])

    for r, rep in enumerate(graph):
        for dist in rep:
            if dist.dist_type == '+':
                kt_vec = to_list(dist.kt_offset_vec)
                if kt_vec is not None:
                    plt.plot([r], kt_vec[3], 'k.')

                    for a in dist.ancestors:
                        anc_kt_vec = to_list(a[1].kt_offset_vec)
                        if anc_kt_vec is not None:
                            plt.plot([r-1, r], [anc_kt_vec[3], kt_vec[3]],
                                     {'z+': 'r-', '-+': 'g:', '++': 'b-'}[a[0]])

    # Plot + distributions
    plt.subplot(212)
    plt.title("z Distributions")
    plt.xlabel("Repetition")
    plt.ylabel("$T_2'$ Dephasing [s]")
    plt.grid()
    plt.xlim([-0.1 * reps, 1.1 * reps])

    for r, rep in enumerate(graph):
        for dist in rep:
            if dist.dist_type in ['z', 'z0']:
                kt_vec = to_list(dist.kt_offset_vec)
                if kt_vec is not None:
                    plt.plot([r], kt_vec[3], 'k.')

                    for a in dist.ancestors:
                        anc_kt_vec = to_list(a[1].kt_offset_vec)
                        if anc_kt_vec is not None:
                            plt.plot([r-1, r], [anc_kt_vec[3], kt_vec[3]],
                                     {'+z': 'r-', '-z': 'g:', 'zz': 'b-'}[a[0]])

    plt.show()
