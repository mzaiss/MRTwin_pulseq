# %% S0. SETUP env
import MRzeroCore as mr0
import pypulseq as pp
import numpy as np
import torch
from matplotlib import pyplot as plt
import util

# makes the ex folder your working directory
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

experiment_id = 'exD01_bSSFP_2D'


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
sz = (128, 128)   # spin system size / resolution
Nread = 128    # frequency encoding steps/samples
Nphase = 61    # phase encoding steps/samples - number of spokes

# Define rf events
rf1, _, _ = pp.make_sinc_pulse(
    flip_angle=6 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)
# rf1 = pp.make_block_pulse(flip_angle=90 * np.pi / 180, duration=1e-3, system=system)
rf0, _, _ = pp.make_sinc_pulse(
    flip_angle=6/2 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)

# Define other gradients and ADC events
gx = pp.make_trapezoid(channel='x', flat_area=Nread, flat_time=5e-3, system=system)
gy = pp.make_trapezoid(channel='y', flat_area=Nread, flat_time=5e-3, system=system)

gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
gy_pre = pp.make_trapezoid(channel='y', area=-gx.area / 2, duration=1e-3, system=system)

adc = pp.make_adc(num_samples=Nread, duration=5e-3, phase_offset=0 * np.pi / 180, delay=gx.rise_time, system=system)

rf_phase = 180
rf_inc = 180

# ======
# CONSTRUCT SEQUENCE
# ======
sdel = 1e-0

seq.add_block(rf0)
seq.add_block(pp.make_delay(3e-3))

for ii in range(-Nphase // 2, Nphase // 2):  # e.g. -64:63

    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase

    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC
    # increment additional pahse
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

    seq.add_block(rf1)

    gx = pp.make_trapezoid(channel='x', flat_area=-Nread * np.sin(ii / Nphase * np.pi) + 1e-7, flat_time=5e-3, system=system)
    gy = pp.make_trapezoid(channel='y', flat_area=Nread * np.cos(ii / Nphase * np.pi) + 1e-7, flat_time=5e-3, system=system)

    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
    gy_pre = pp.make_trapezoid(channel='y', area=-gy.area / 2, duration=1e-3, system=system)

    seq.add_block(gx_pre, gy_pre)
    seq.add_block(adc, gx, gy)
    # seq.add_block(adc,gx,gy)
    seq.add_block(gx_pre, gy_pre)
    # seq.add_block(make_delay(10))


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

if 1:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('../data/numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)
    # Manipulate loaded data
    obj_p.T2dash[:] = 30e-3
    obj_p.D *= 0
    obj_p.B0 *= 1    # alter the B0 inhomogeneity
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
        B0=0,
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
seq0 = mr0.Sequence.from_seq_file("out/external.seq")
seq0.plot_kspace_trajectory()
kspace_loc = seq0.get_kspace()
# Simulate the sequence
graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0, obj_p)

# PLOT sequence with signal in the ADC subplot
plt.close(11);plt.close(12)
sp_adc, t_adc = util.pulseq_plot(seq, clear=False, signal=signal.numpy())


# %% S6: MR IMAGE RECON of signal ::: #####################################
fig = plt.figure()  # fig.clf()
plt.subplot(411)
plt.title('ADC signal')
kspace_adc = torch.reshape((signal), (Nphase, Nread)).clone().t()
plt.plot(torch.real(signal), label='real')
plt.plot(torch.imag(signal), label='imag')

# this adds ticks at the correct position szread
major_ticks = np.arange(0, Nphase * Nread, Nread)
ax = plt.gca()
ax.set_xticks(major_ticks)
ax.grid()

if 0:  # FFT
    # fftshift
    spectrum = torch.fft.fftshift(kspace_adc)
    # FFT
    space = torch.fft.ifft2(spectrum)
    # fftshift
    space = torch.fft.ifftshift(space)


if 1:  # NUFFT
    import scipy.interpolate
    grid = kspace_loc[:, :2]
    Nx = 64
    Ny = 64

    X, Y = np.meshgrid(np.linspace(0, Nx - 1, Nx) - Nx / 2,
                       np.linspace(0, Ny - 1, Ny) - Ny / 2)
    grid = np.double(grid.numpy())
    grid[np.abs(grid) < 1e-3] = 0

    plt.subplot(347)
    plt.plot(grid[:, 0].ravel(), grid[:, 1].ravel(), 'rx', markersize=3)
    plt.plot(X, Y, 'k.', markersize=2)
    plt.show()

    spectrum_resampled_x = scipy.interpolate.griddata(
        (grid[:, 0].ravel(), grid[:, 1].ravel()),
        np.real(signal.ravel()), (X, Y), method='cubic'
    )
    spectrum_resampled_y = scipy.interpolate.griddata(
        (grid[:, 0].ravel(), grid[:, 1].ravel()),
        np.imag(signal.ravel()), (X, Y), method='cubic'
    )

    kspace_r = spectrum_resampled_x + 1j * spectrum_resampled_y
    kspace_r[np.isnan(kspace_r)] = 0

    # fftshift
    # kspace_r = np.roll(kspace_r,Nx//2,axis=0)
    # kspace_r = np.roll(kspace_r,Ny//2,axis=1)
    kspace_r_shifted = np.fft.ifftshift(kspace_r, 0)
    kspace_r_shifted = np.fft.ifftshift(kspace_r_shifted, 1)

    space = np.fft.ifft2(kspace_r_shifted)
    space = np.fft.ifftshift(space, 0)
    space = np.fft.ifftshift(space, 1)

space = np.transpose(space)
plt.subplot(345)
plt.title('k-space')
plt.imshow(np.abs(kspace_adc))
plt.subplot(349)
plt.title('k-space_r')
plt.imshow(np.abs(kspace_r))

plt.subplot(346)
plt.title('FFT-magnitude')
plt.imshow(np.abs(space))
plt.colorbar()
plt.subplot(3, 4, 10)
plt.title('FFT-phase')
plt.imshow(np.angle(space), vmin=-np.pi, vmax=np.pi)
plt.colorbar()

# % compare with original phantom obj_p.PD
plt.subplot(348)
plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3, 4, 12)
plt.title('phantom B0')
plt.imshow(B0)
