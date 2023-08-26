# %% S0. SETUP env
import MRzeroCore as mr0
import pypulseq as pp
import numpy as np
import matplotlib.pyplot as plt
import util

# makes the ex folder your working directory
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

experiment_id = 'exA03_gradient_echo'


# %% S1. SETUP sys

# choose the scanner limits
system = pp.Opts(
    max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6,
    adc_dead_time=20e-6, grad_raster_time=50 * 10e-6
)


# %% S2. DEFINE the sequence
seq = pp.Sequence()

# Define FOV and resolution
fov = 1000e-3
Nread = 128
Nphase = 1
slice_thickness = 8e-3  # slice

# Define rf events
rf1, _, _ = pp.make_sinc_pulse(
    flip_angle=90 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)
# rf1 = pp.make_block_pulse(flip_angle=90 * np.pi / 180, duration=1e-3, system=system)

# Define other gradients and ADC events

gx = pp.make_trapezoid(channel='x', flat_area=Nread, flat_time=20e-3, system=system)
gx_ = pp.make_trapezoid(channel='x', flat_area=-Nread, flat_time=20e-3, system=system)

gx_pre = pp.make_trapezoid(channel='x', area=-Nread / 2, duration=2e-3, system=system)

adc = pp.make_adc(num_samples=Nread, duration=20e-3, phase_offset=0 * np.pi / 180, delay=gx.rise_time, system=system)


# ======
# CONSTRUCT SEQUENCE
# ======
seq.add_block(pp.make_delay(0.011 - rf1.delay))
seq.add_block(rf1)
seq.add_block(gx_pre)
seq.add_block(adc, gx)
seq.add_block(adc, gx_)
seq.add_block(adc, gx)
seq.add_block(adc, gx_)

# seq.add_block(adc)


# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc, t_adc = util.pulseq_plot(seq, clear=False, figid=(11,12))

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id + '.seq')


# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
sz = [64, 64]

if 0:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('../data/numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)
    # Manipulate loaded data
    obj_p.B0 *= 1    # alter the B0 inhomogeneity
    obj_p.D *= 0 
else:
    # or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
    obj_p = mr0.CustomVoxelPhantom(
        pos=[[-0.25, -0.25, 0]],
        PD=[1.0],
        T1=[3.0],
        T2=[0.5],
        T2dash=[30e-3],
        D=[0.0],
        B0=0,
        voxel_size=0.1,
        voxel_shape="box"
    )

obj_p.plot()
# Convert Phantom into simulation data
obj_p = obj_p.build()


# %% S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot

# Read in the sequence
seq0 = mr0.Sequence.from_seq_file("out/external.seq")
# seq0.plot_kspace_trajectory()
# Simulate the sequence
graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0, obj_p)

# PLOT sequence with signal in the ADC subplot
plt.close(11);plt.close(12)
sp_adc, t_adc = util.pulseq_plot(seq, clear=False, signal=signal.numpy())
 
 
