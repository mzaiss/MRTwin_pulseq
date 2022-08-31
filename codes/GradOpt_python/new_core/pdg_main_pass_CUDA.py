"""This file requires pycuda and the nVidia CUDA toolkit to be installed."""

from __future__ import annotations

import pycuda.autoinit
from pycuda.driver import PointerHolderBase
from pycuda.compiler import SourceModule
import numpy as np
import torch

from .sequence import Sequence
from .sim_data import SimData
from . import util

# this seems to be necessary for torch and pycuda to use the same context
x = torch.cuda.FloatTensor(8)


class Holder(PointerHolderBase):
    """This allows using a CUDA pyTorch tensor as a pyCuda GPUArray."""

    def __init__(self, tensor: torch.Tensor):
        """Create an instance that holds the passed CUDA tensor."""
        if not tensor.is_cuda:
            raise TypeError("Holder can't take non-CUDA tensors")

        super(Holder, self).__init__()
        self.tensor = tensor
        self.gpudata = tensor.data_ptr()


class CudaDist:
    """A distribution that is adapted to the usage in the CUDA kernel."""

    TYPES = {'+!': 0, '+': 1, 'z': 2, 'z0': 3}

    def __init__(self, type: str):
        """Create a new instance of given type."""
        self.ancestors = [None, None, None]
        self.attenuation = None
        self.dephasing_time = None
        self.kt_vec = None
        self.type = type
        self.index = None

    def __repr__(self) -> str:
        """Convert this instance into a compact printable form."""
        def anc(n: int) -> str:
            if self.ancestors[n] == -1:
                return ' '
            else:
                return str(self.ancestors[n])

        if self.type in ['+!', '+']:
            anc_list = f"{{++ {anc(0)}, -+ {anc(1)}, z+ {anc(2)}}}"
        else:
            anc_list = f"{{zz {anc(0)}, -z {anc(1)}, +z {anc(2)}}}"
        return f"D({self.index}, {self.type}, {anc_list}"


def convert_graph(graph: list[list], seq: Sequence, data_shape: torch.Tensor,
                  min_influence: float = 1e-3,
                  min_signal: float = 1e-3) -> list[list[CudaDist]]:
    """Convert a normal graph into a Cuda Graph.

    Strip distributions that are not simulated and convert into a form that
    does not rely on pointers but uses indices. Additionally precompute data
    that doesn't rely on the phantom itself.
    """
    assert util.use_gpu, "Only use with util.use_gpu = True"
    data_shape = util.set_device(data_shape)
    mapping = {}
    new_graph = []

    dist = CudaDist('z0')
    mapping[graph[0][0]] = dist
    new_graph.append([dist])
    # Use mag as marker for determining which ancestors were simulated
    graph[0][0].mag = True

    # Copy graph (only simulated distributions)
    for rep in graph[1:]:
        new_rep = []
        for dist in rep:
            # Ancestors that actually were simulated
            ancestors = list(filter(lambda edge: edge[1].mag, dist.ancestors))

            # Mark distributions that are not simulated
            if dist.dist_type != 'z0' and (
                    len(ancestors) == 0 or dist.rel_influence < min_influence):
                dist.mag = None
            else:
                dist.mag = True

                dist_type = dist.dist_type
                if dist_type == '+' and dist.prepass_rel_signal >= min_signal:
                    dist_type = '+!'

                new_dist = CudaDist(dist_type)

                for edge in ancestors:
                    index = {
                        '++': 0, 'zz': 0,
                        '-+': 1, '-z': 1,
                        'z+': 2, '+z': 2
                    }[edge[0]]
                    new_dist.ancestors[index] = mapping[edge[1]]

                # Insert distribution into mapping and graph
                mapping[dist] = new_dist
                new_rep.append(new_dist)
        new_graph.append(new_rep)

    # Calculate dephasing_time and attenuation
    new_graph[0][0].kt_vec = torch.zeros(4, device=util.get_device())

    for dists, rep in zip(new_graph[1:], seq):
        trajectory = torch.cat([
            torch.cumsum(util.set_device(rep.gradm), 0),
            torch.cumsum(util.set_device(rep.event_time), 0).unsqueeze(1)], 1)

        for dist in dists:
            # Calculate kt_vec after pulse
            if dist.type == 'z0':
                dist.kt_vec = torch.zeros(4, device=util.get_device())
            elif dist.ancestors[0]:  # ++ or zz
                dist.kt_vec = dist.ancestors[0].kt_vec.clone()
            elif dist.ancestors[1]:  # -+ or -z
                dist.kt_vec = -1.0 * dist.ancestors[1].kt_vec
            elif dist.ancestors[2]:  # z+ or +z
                dist.kt_vec = dist.ancestors[2].kt_vec.clone()

            # Calculate attenuation and time dephasing
            if dist.type == '+!':
                dist.dephasing_time = (dist.kt_vec[3] + trajectory[:, 3]).abs()
                dist.attenuation = torch.prod(
                    torch.sinc(
                        0.5 * (dist.kt_vec[:3] + trajectory[:, :3])
                        / data_shape / np.pi
                    ), dim=1
                ) * rep.adc * 1.41421356237
            else:  # not measured = not needed
                dist.dephasing_time = torch.zeros(rep.event_count)
                dist.attenuation = torch.zeros(rep.event_count)

            # Update kt_vec
            if dist.type in ['+!', '+']:
                dist.kt_vec += trajectory[-1]

    # Sort repetitions so that measured distributions come first
    for rep in new_graph:
        rep.sort(key=lambda x: x.type == '+!', reverse=True)

    # Replace references by indices
    for rep in new_graph:
        for index, dist in enumerate(rep):
            dist.index = index
            index += 1
            for i in range(3):
                if dist.ancestors[i]:
                    dist.ancestors[i] = dist.ancestors[i].index
                else:
                    dist.ancestors[i] = -1

    return new_graph


class CudaData:
    """Stores a sequence and its graph in a GPU-friendly way.

    The graph contained in a :class:`CudaData` instance is converted to
    multiple linear arrays that don't rely on pointers but indices.

    Attributes
    ----------
    rep_count: int
        Number of repetitions
    total_event_count: int
        Total number of events
    event_count: torch.Tensor
        int32 tensor containing the number of events for every rep
    pulse_angle: torch.Tensor
        float32 tensor containing the pulse flip angles for every rep
    pulse_phase: torch.Tensor
        float32 tensor containing the pulse phase for every rep
    dist_count: torch.Tensor
        int32 tensor containing the number of distributions for every rep
    event_time: torch.Tensor
        float32 tensor containing the duration of every event
    gradm_event: torch.Tensor
        float32 tensor, size ``event_count * 3``, containing gradient moments
    attenuation: torch.Tensor
        cfloat tensor, size ``sum(dist_count[i] * event_count[i]``, containing
        the gradient moment induced attenuation for every distribution & event
    dephasing_time: torch.Tensor
        float tensor, size ``sum(dist_count[i] * event_count[i]``, containing
        the dephasing time for every distribution & event (for T2')
    dists: torch.Tensor
        int tensor, size ``sum(dist_count) * 4``, contains for every dist its
        up to 4 ancestors via their index in the previous repetition
    """

    def __init__(self, graph: list[list], seq: Sequence,
                 data_shape: torch.Tensor) -> None:
        """Create a ``CudaData`` instance by converting the passed args."""
        assert util.use_gpu, "Only use with util.use_gpu = True"
        dev = util.get_device()
        graph = convert_graph(graph, seq, data_shape)

        self.rep_count = len(seq)
        self.total_event_count = sum([rep.event_count for rep in seq])

        # One element per repetition
        self.event_count = torch.tensor([rep.event_count for rep in seq],
                                        dtype=torch.int32, device=dev)
        self.pulse_angle = torch.tensor([rep.pulse.angle for rep in seq],
                                        dtype=torch.float32, device=dev)
        self.pulse_phase = torch.tensor([rep.pulse.phase for rep in seq],
                                        dtype=torch.float32, device=dev)
        self.dist_count = torch.tensor([len(rep) for rep in graph[1:]],
                                       dtype=torch.int32, device=dev)

        # One element per event
        self.event_time = util.set_device(torch.cat(
            [rep.event_time for rep in seq], dim=0).type(torch.float32))
        self.gradm_event = util.set_device(torch.cat(
            [rep.gradm for rep in seq], dim=0))

        event_dist_size = sum([
            len(graph_rep) * seq_rep.event_count
            for graph_rep, seq_rep in zip(graph[1:], seq)
        ])

        # One element per event per distribution
        self.attenuation = torch.zeros(event_dist_size, dtype=torch.cfloat,
                                       device=dev)
        self.dephasing_time = torch.zeros(event_dist_size, device=dev)

        # One element per distribution
        self.dists = torch.zeros(int(self.dist_count.sum()), 4,
                                 dtype=torch.int32, device=dev)

        i = 0
        for rep in graph[1:]:
            for dist in rep:
                self.dists[i, 0] = dist.ancestors[0]
                self.dists[i, 1] = dist.ancestors[1]
                self.dists[i, 2] = dist.ancestors[2]
                self.dists[i, 3] = CudaDist.TYPES[dist.type]
                i += 1

        i = 0
        for graph_rep, seq_rep in zip(graph[1:], seq):
            for event in range(seq_rep.event_count):
                for dist in graph_rep:
                    self.attenuation[i] = dist.attenuation[event]
                    self.dephasing_time[i] = dist.dephasing_time[event]
                    i += 1

        self.compile()

    def compile(self):
        """Compile the CUDA kernel."""
        with open('new_core/execute_graph_kernel.cu') as file:
            kernel_source = file.read()
            print("Compiling CUDA kernel...")
            self.cuda_func = SourceModule(kernel_source, options=[
                "--use_fast_math",
                f"-DMAX_DIST_COUNT={max(self.dist_count)}"
            ], no_extern_c=True).get_function("execute_graph")


def execute_graph_CUDA(seq_data: CudaData, sim_data: SimData
                       ) -> list[torch.Tensor]:
    """Simulate the sequence stored in ``seq_data`` on ``sim_data``.

    Arguments
    ---------
    seq_data: CudaData
        The sequence and its graph, prepared for usage in the CUDA kernel
    sim_data: SimData
        The phantom data which is used by the simulation

    Returns
    -------
    list[torch.Tensor]
        The signal returnd by the simulation, split into the repetitions
    """
    assert util.use_gpu, "This function can only run on a GPU"

    signal = torch.zeros((seq_data.total_event_count, sim_data.coil_count),
                         dtype=torch.cfloat, device=util.get_device())

    coil_sensitivity = (
        sim_data.coil_sens.t()
        * sim_data.PD.unsqueeze(1) / sim_data.PD.sum()
    )

    seq_data.cuda_func(
        np.int32(sim_data.voxel_count), np.int32(sim_data.coil_count),
        Holder(sim_data.T1), Holder(sim_data.T2), Holder(sim_data.T2dash),
        Holder(sim_data.B0), Holder(sim_data.B1),
        Holder(sim_data.voxel_pos), Holder(coil_sensitivity),
        np.int32(seq_data.rep_count), Holder(seq_data.event_count),
        Holder(seq_data.pulse_angle), Holder(seq_data.pulse_phase),
        Holder(seq_data.event_time), Holder(seq_data.gradm_event),
        Holder(seq_data.dists), Holder(seq_data.dist_count),
        Holder(seq_data.attenuation), Holder(seq_data.dephasing_time),
        Holder(signal),
        block=(1024, 1, 1),
        grid=(int(np.ceil(sim_data.voxel_count / 1024)), 1)
    )

    # Concurrent execution possible:
    #
    # For each invocation create a new CUDA stream and pass it as stream=...
    # Return a wrapper for that stream that also holds the signal tensor
    # When the signal is needed, synchronize the stream and return the signal
    #
    # NOTE: Profile if it makes sense / is done right, first tests did not show
    # any performance gains. (Maybe streams were prematurely synchronized?)

    return signal.split(seq_data.event_count.tolist())


def print_dist_tensor(flat_dists: torch.Tensor, dist_count: torch.Tensor):
    rep_lengths = dist_count.cpu().tolist()
    flat_dists = flat_dists.cpu().tolist()
    dists = []
    for count in rep_lengths:
        dists.append(flat_dists[:count])
        flat_dists = flat_dists[count:]

    print("        ++ -+ z+")
    print("        zz -z +z")

    def fmt(x: int):
        if x < 0:
            return "--"
        else:
            return f"{x:2d}"

    for r, rep in enumerate(dists):
        print()
        print(f"# {r}")
        for d, dist in enumerate(rep):
            print(f"{d:2d}: {['+!', ' +', ' z', 'z0'][dist[3]]}  ", end="")
            print(f"{fmt(dist[0])} {fmt(dist[1])} {fmt(dist[2])}")
