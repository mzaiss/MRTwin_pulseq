# %% S0. SETUP env
import MRzeroCore as mr0
import pypulseq as pp
import numpy as np
import torch
from matplotlib import pyplot as plt

# makes the ex folder your working directory
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

experiment_id = 'exA05_incoherent_echo_train'


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
slice_thickness = 8e-3

Nread = 128    # frequency encoding steps/samples
Nphase = 32    # phase encoding steps/samples

# Define rf events
rf1, _, _ = pp.make_sinc_pulse(
    flip_angle=60 * np.pi / 180, phase_offset=90 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)
rf2, _, _ = pp.make_sinc_pulse(
    flip_angle=120 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)


# Define other gradients and ADC events
adc = pp.make_adc(num_samples=Nread, duration=20e-3, phase_offset=90*np.pi/180, delay=5*1e-4, system=system)
adc0 = pp.make_adc(num_samples=Nread, duration=8e-3, phase_offset=90*np.pi/180, delay=5*1e-4, system=system)

# ======
# CONSTRUCT SEQUENCE
# ======

seq.add_block(rf1)
seq.add_block(adc0)

for ii in range(-Nphase // 2, Nphase // 2 - 1):  # e.g. -64:63
    seq.add_block(rf2)
    seq.add_block(pp.make_delay(5e-3))
    seq.add_block(adc)
    seq.add_block(pp.make_delay(1e-3))


# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc, t_adc = mr0.pulseq_plot(seq, clear=False)

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id + '.seq')

# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
sz = [64, 64]

if 0:
    # (i) load a phantom object from file
    # obj_p = VoxelGridPhantom.load('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('../data/numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)
    # Manipulate loaded data
    # obj_p.B0 *= Z1
    # obj_p.T2dash *= 0.05
    # obj_p.T2 *= 100
    # obj_p.T1 *= 100
    # obj_p.B1[:]=1
    obj_p.D *= 0
    # Store PD for comparison
    PD = obj_p.PD
    B0 = obj_p.B0
else:
    # or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
    obj_p = mr0.CustomVoxelPhantom(
        pos=[[-0.4, -0.4, 0], [-0.4, -0.2, 0], [-0.3, -0.2, 0], [-0.2, -0.2, 0], [-0.1, -0.2, 0]],
        PD=[1.0, 1.0, 0.5, 0.5, 0.5],
        T1=1.0,
        T2=0.1,
        T2dash=0.1,
        D=0.0,
        voxel_size=0.1,
        voxel_shape="box"
    )
    # Store PD for comparison
    PD = obj_p.generate_PD_map()
    B0 = torch.zeros_like(PD)

obj_p.plot()
# Convert Phantom into simulation data
obj_p = obj_p.build()


# %% S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot

# Read in the sequence
seq_file = mr0.PulseqFile("out/external.seq")
# seq_file.plot()
seq = mr0.Sequence.from_seq_file(seq_file)
seq.plot_kspace_trajectory()
# Simulate the sequence
graph = mr0.compute_graph(seq, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq, obj_p)

# plot the result into the ADC subplot
sp_adc.plot(t_adc, np.real(signal.numpy()), t_adc, np.imag(signal.numpy()))
sp_adc.plot(t_adc, np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())

# %% S6: MR IMAGE RECON of signal ::: #####################################
fig = plt.figure()  # fig.clf()
plt.subplot(411)
plt.title('ADC signal')
spectrum = torch.reshape((signal), (Nphase, Nread)).clone().t()
kspace = spectrum
plt.plot(torch.real(signal), label='real')
plt.plot(torch.imag(signal), label='imag')


# this adds ticks at the correct position szread
major_ticks = np.arange(0, Nphase * Nread, Nread)
ax = plt.gca()
ax.set_xticks(major_ticks)
ax.grid()

space = torch.zeros_like(spectrum)

# fftshift
spectrum = torch.fft.fftshift(spectrum, 0)
spectrum = torch.fft.fftshift(spectrum, 1)
# FFT
space = torch.fft.ifft2(spectrum)
# fftshift
space = torch.fft.ifftshift(space, 0)
space = torch.fft.ifftshift(space, 1)


plt.subplot(345)
plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349)
plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346)
plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy()))
plt.colorbar()
plt.subplot(3, 4, 10)
plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()), vmin=-np.pi, vmax=np.pi)
plt.colorbar()

# % compare with original phantom obj_p.PD
plt.subplot(348)
plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3, 4, 12)
plt.title('phantom B0')
plt.imshow(B0)
