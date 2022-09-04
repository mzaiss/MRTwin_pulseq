# %% Imports
from __future__ import annotations
import torch
import numpy as np
from pulseq_loader import intermediate, PulseqFile, plot_file, Adc, Spoiler
from new_core.sequence import Sequence, PulseUsage
from new_core.util import plot_kspace_trajectory
from new_core import util
import matplotlib.pyplot as plt
from new_core.sim_data import SimData
import torchvision
import pre_pass
from new_core.pdg_main_pass import execute_graph
from new_core.reconstruction import reconstruct

util.use_gpu = False


def sim_external(object_sz=32,reco_sz=0, plot_seq_k=(0,0), obj=0,dB0=0,M_threshold=1e-3):   
    #  Load a pulseq file
    
    # NOTE
    # .seq files exported by the official exporter put the adc sample at the
    # beginning of the event but MRzero has them at the end - we need to shift
    # adc by 1 in our exporters
    #
    # We interpret gradients & pulses with time shapes as being constant until the
    # next time point, but linear interp. might be assumed - the spec says nothing
    #
    # Adc phases in the pulseq files seem to be defined such that e.g. in a GRE,
    # adc phase should equal pulse phase. In MRzeros coordinate space, adc phase
    # should be 90° - pulse phase, which means we should export 90° - adc phase,
    # but currently use adc phase - 45°
    
    # Import a pulseq file (supported: 1.2.0, 1.2.1, 1.3.0, 1.3.1, 1.4.0)
    pulseq = PulseqFile("out/external.seq")
    # Can also save it again as 1.4.0 file (for converting older files)
    # pulseq.save("pulseq_loader/tests/out.seq")
    # Plot the full sequence stored in the file
    if plot_seq_k[0]:
        plot_file(pulseq, figsize=(10, 6))
    # Convert seqence into a intermediate form only containing what's simulated
    tmp_seq = intermediate(pulseq)
    
    # Convert into a MRzero sequence
    seq = Sequence()
    rep = None
    for tmp_rep in tmp_seq:
        rep = seq.new_rep(tmp_rep[0])
        rep.pulse.angle = torch.tensor(tmp_rep[1].angle, dtype=torch.float)
        rep.pulse.phase = torch.tensor(tmp_rep[1].phase, dtype=torch.float)
        is_refoc = abs(tmp_rep[1].angle) > 1.6  # ~91°
        rep.pulse.usage = PulseUsage.REFOC if is_refoc else PulseUsage.EXCIT
    
        offset = 0
        for block in tmp_rep[2]:
            if isinstance(block, Spoiler):
                rep.event_time[offset] = block.duration
                rep.gradm[offset, :] = torch.tensor(block.gradm)
                offset += 1
            else:
                assert isinstance(block, Adc)
                num = len(block.event_time)
                rep.event_time[offset:offset+num] = torch.tensor(block.event_time)
                rep.gradm[offset:offset+num, :] = torch.tensor(block.gradm)
                rep.adc_phase[offset:offset+num] = np.pi/2 - block.phase
                rep.adc_usage[offset:offset+num] = 1
                offset += num
        assert offset == tmp_rep[0]
    
    # Trajectory calculation and plotting
    
    if plot_seq_k[1]:
        plot_kspace_trajectory(seq, plot_timeline=False)
        
        # time_axis = np.arange(1, ktraj.shape[1] + 1) * system.grad_raster_time
        # plt.figure()
        # plt.plot(time_axis, ktraj.T)  # Plot the entire k-space trajectory
        # plt.plot(t_adc, ktraj_adc[0], '.')  # Plot sampling points on the kx-axis
        # plt.figure()
        # plt.plot(ktraj[0], ktraj[1], 'b')  # 2D plot
        # plt.axis('equal')  # Enforce aspect ratio for the correct trajectory display
        # plt.plot(ktraj_adc[0], ktraj_adc[1], 'r.')  # Plot  sampling points
        # plt.show()

    
    # Convert sequence to gpu if needed
    for rep in seq:
        rep.pulse.angle = util.set_device(rep.pulse.angle)
        rep.pulse.phase = util.set_device(rep.pulse.phase)
        rep.adc_phase = util.set_device(rep.adc_phase)
        rep.gradm = util.set_device(rep.gradm)
        rep.event_time = util.set_device(rep.event_time)
    
    
    # Simulate imported sequence
    reco_size = (reco_sz, reco_sz, 1)
    
    # data = SimData.load_npz(
    #     file_name="brainweb/output/subject20.npz"
    # ).select_slice(64)

    data=obj        
       
    
    pre_pass_settings = (
        data.shape.tolist(),
        float(torch.mean(data.T1)),
        float(torch.mean(data.T2)),
        float(torch.mean(data.T2dash)),
        1000,  # Number of states (+ and z) simulated in pre-pass
        M_threshold,  # Minimum magnetisation of states in pre-pass
    )
    
    graph = pre_pass.compute_graph(seq, *pre_pass_settings)
    signal = execute_graph(graph, seq, data)
    kspace = seq.get_kspace()
    
    # Spiral sample density compensation
    # dist = (np.sqrt(kspace[:, 0]**2 + kspace[:, 1]**2))[:, None]
    # signal *= dist
    
    if reco_sz>0:
    
        reco = reconstruct(
            signal, kspace,
            resolution=reco_size, FOV=(1, 1, 1)
        )
        
        plt.figure(figsize=(7, 5))
        plt.subplot(211),
        plt.imshow(reco.abs(), vmin=0)
        plt.colorbar()
        plt.subplot(212),
        plt.imshow(reco.angle())
        plt.show()
    
    return signal, kspace
