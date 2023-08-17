# %% S0. SETUP env
from skimage.restoration import denoise_tv_chambolle
import pywt
import MRzeroCore as mr0
import numpy as np
from matplotlib import pyplot as plt
import pypulseq as pp
import torch
# makes the ex folder your working directory
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import util

experiment_id = 'exD01_bSSFP_2D'

# %% S1. SETUP sys

# choose the scanner limits
system = pp.Opts(
    max_grad=28,
    grad_unit='mT/m',
    max_slew=150,
    slew_unit='T/m/s',
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=20e-6,
    grad_raster_time=50*10e-6
)

# %% S2. DEFINE the sequence 
seq = pp.Sequence()

# Define FOV and resolution
fov = 1000e-3 
slice_thickness=8e-3
sz=(64,64)   # spin system size / resolution
Nread = 64    # frequency encoding steps/samples
Nphase = 64    # phase encoding steps/samples

# Define rf events
rf1 = pp.make_sinc_pulse(flip_angle=5 * np.pi / 180, duration=1e-3,slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4, system=system)
# rf1, _= pp.make_block_pulse(flip_angle=90 * np.pi / 180, duration=1e-3, system=system)

# Define other gradients and ADC events
gx = pp.make_trapezoid(channel='x', flat_area=Nread, flat_time=10e-3, system=system)
adc = pp.make_adc(num_samples=Nread, duration=10e-3, phase_offset=0*np.pi/180,delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=5e-3, system=system)
gx_spoil = pp.make_trapezoid(channel='x', area=1.5*gx.area, duration=2e-3, system=system)

rf_phase = 0
rf_inc = 0
rf_spoiling_inc=117

phase_enc__gradmoms = torch.arange(0,Nphase,1)-Nphase//2

# ======
# CONSTRUCT SEQUENCE
# ======
#idx  = [1,5,15,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,45,50,55]
#idx = np.linspace(0,63,64)
idx  = [5,15,20,27,28,29,30,31,32,33,34,35,36,37,38,39,40,45,50,55]
# idx = np.random.poisson(Nread/2,Nread*8)
idx = np.unique(idx)
for ii in range(0, len(idx)):  # e.g. -64:63

    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC
    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]   # increase increment
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]        # increment additional pahse

    seq.add_block(rf1)
    gp= pp.make_trapezoid(channel='y', area=phase_enc__gradmoms[int(idx[ii])], duration=5e-3, system=system)
    seq.add_block(gx_pre,gp)
    seq.add_block(adc,gx)
    gp= pp.make_trapezoid(channel='y', area=-phase_enc__gradmoms[int(idx[ii])], duration=5e-3, system=system)
    seq.add_block(gx_spoil,gp)
    if ii<Nphase-1:
        seq.add_block(pp.make_delay(0.001))
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

if 1:
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
seq0.plot_kspace_trajectory()
# Simulate the sequence
graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0, obj_p)

# PLOT sequence with signal in the ADC subplot
plt.close(11);plt.close(12)
sp_adc, t_adc = util.pulseq_plot(seq, clear=False, signal=signal.numpy())

kspace_adc=torch.reshape((signal),(len(idx),Nread)).clone().t()

kspace = torch.zeros((Nread,Nread),dtype = torch.complex64)
kspace[:,idx] = kspace_adc
pattern = torch.zeros((Nread,Nread))
pattern[:,idx] = torch.ones(Nread,len(idx))

# high  frequencies centered as kspace and as FFT needs it
pattern = np.fft.fftshift(pattern.numpy())
kspace = np.fft.ifftshift(kspace.numpy())

# kspace = kspace_full * pattern  # apply the undersampling pattern

# calculate the actually measured data in percent
actual_measured_percent = np.count_nonzero(pattern) / pattern.size * 100

# Plotting

pattern_vis = np.fft.fftshift(pattern.copy())
kspace_vis = np.log(1+abs(np.fft.fftshift((kspace.copy()))))
fig = plt.figure(dpi=90)
plt.subplot(321)
plt.set_cmap(plt.gray())
# plt.imshow(abs(recon_nufft))
plt.ylabel('recon_full')
plt.subplot(322)
plt.set_cmap(plt.gray())
plt.imshow(abs(pattern_vis))
plt.ylabel("pattern_vis")
plt.title("{:.1f} % sampled".format(actual_measured_percent))

plt.subplot(324)
plt.set_cmap(plt.gray())
plt.imshow(np.log(1+abs(kspace_adc.numpy().copy())))
plt.ylabel('kspace_adc')
plt.subplot(326)
plt.set_cmap(plt.gray())
plt.imshow(kspace_vis)
plt.ylabel('kspace*pattern')
plt.show()

# %% ##########################################################################
# S6: compressed sensing MR reconstruction of undersampled signal
# S6.1: function definitions

def shrink(coeff, epsilon):
    shrink_values = (abs(coeff) < epsilon)
    high_values = coeff >= epsilon
    low_values = coeff <= -epsilon
    coeff[shrink_values] = 0
    coeff[high_values] -= epsilon
    coeff[low_values] += epsilon


# help?
#  https://www2.isye.gatech.edu/~brani/wp/kidsA.pdf
for family in pywt.families():
    print("%s family: " % family + ', '.join(pywt.wavelist(family)))

print(pywt.Wavelet('haar'))


def waveletShrinkage(current, epsilon):
    # Compute Wavelet decomposition
    cA, (cH, cV, cD) = pywt.dwt2(current, 'haar')
    # Shrink
    shrink(cA, epsilon)
    shrink(cH, epsilon)
    shrink(cV, epsilon)
    shrink(cD, epsilon)
    wavelet = cA, (cH, cV, cD)
    # return inverse WT
    return pywt.idwt2(wavelet, 'haar')


def updateData(k_space, pattern, current, step, i):
    # go to k-space
    update = np.fft.ifft2(np.fft.fftshift(current))
    # compute difference
    update = k_space - (update * pattern)
    print("i: {}, consistency RMSEpc: {:3.6f}".format(
        i, np.abs(update[:]).sum() * 100))
    # return to image space
    update = np.fft.fftshift(np.fft.fft2(update))
    # improve current estimation by consitency
    update = current + (step * update)
    return update


# S6.2: preparation and conventional fully sampled reconstruction

# high  frequencies centered as FFT needs it
kspace_full = np.fft.ifftshift(kspace_adc)

# fully sampled recon
recon_nufft = (np.fft.fftshift(np.fft.fft2(kspace_full)))


# %% S6.3 undersampling and undersampled reconstruction
# space= space/ np.linalg.norm(space[:])   # normalization of the data somethimes helps

# parameters of iterative reconstructio using total variation denoising
denoising_strength = 10e-6
number_of_iterations = 8000


# actual iterative reconstruction algorithm
current = np.zeros(kspace.shape)
current_shrink = np.zeros(kspace.size).reshape(kspace.shape)
first = updateData(kspace, pattern, current, 1, 0)
current_shrink = first
all_iter = np.zeros((kspace.shape[0], kspace.shape[1], number_of_iterations))

for i in range(number_of_iterations):
    current = updateData(kspace, pattern, current_shrink, 0.1, i)

    current_shrink = denoise_tv_chambolle(abs(current), denoising_strength)
    # current_shrink = waveletShrinkage(abs(current), denoising_strength)

    all_iter[:, :, i] = current




plt.subplot(323)
plt.set_cmap(plt.gray())
plt.imshow(abs(first))
plt.ylabel('first iter (=NUFFT)')
plt.subplot(325)
plt.set_cmap(plt.gray())
plt.imshow(abs(current_shrink))
plt.ylabel('final recon')



# %% Plot all iter
# make 25 example iterations
idx = np.linspace(1, all_iter.shape[2], 25) - 1
# choose them from all iters
red_iter = all_iter[:, :, tuple(idx.astype(int))]
Tot = red_iter.shape[2]
Rows = Tot // 5
if Tot % 5 != 0:
    Rows += 1
Position = range(1, Tot + 1)  # Position index

fig = plt.figure()
for k in range(Tot):
    ax = fig.add_subplot(Rows, 5, Position[k])
    ax.imshow((abs((red_iter[:, :, k]))))
    plt.title('iter {}'.format(idx[k].astype(int)))
    print(k)
plt.show()
