experiment_id = 'exB02_SE_to_RARE_2D'

# %% S0. SETUP env
import sys,os
os.chdir(os.path.abspath(os.path.dirname(__file__)))  #  makes the ex folder your working directory
sys.path.append(os.path.dirname(os.getcwd()))         #  add required folders to path
mpath=os.path.dirname(os.getcwd())
c1=r'codes'; c2=r'codes\GradOpt_python'; c3=r'codes\scannerloop_libs' #  add required folders to path
sys.path += [rf'{mpath}\{c1}',rf'{mpath}\{c2}',rf'{mpath}\{c3}']

## imports for simulation
from GradOpt_python.pulseq_sim_external import sim_external
from GradOpt_python.new_core.util import plot_kspace_trajectory
import math
import numpy as np
import torch
from matplotlib import pyplot as plt

## imports for pypulseq
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts


# %% S1. SETUP sys

## choose the scanner limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', rf_ringdown_time=20e-6,
                 rf_dead_time=100e-6, adc_dead_time=20e-6,grad_raster_time=50*10e-6)

# %% S2. DEFINE the sequence 
seq = Sequence()

# Define FOV and resolution
fov = 1000e-3 
slice_thickness=8e-3

Nread = 64    # frequency encoding steps/samples
Nphase = 64    # phase encoding steps/samples

# Define rf events
rf1, _,_ = make_sinc_pulse(flip_angle=90* math.pi / 180, phase_offset=90* math.pi / 180,duration=1e-3,slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4, system=system)
rf2, _,_ = make_sinc_pulse(flip_angle=180* math.pi / 180, duration=1e-3,slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4, system=system)
# rf1, _= make_block_pulse(flip_angle=90 * math.pi / 180, duration=1e-3, system=system)

rf2.delay=0  
#rf2.delay=5*1e-4    # wrong timing! strong artifacts

# Define other gradients and ADC events
gx = make_trapezoid(channel='x', flat_area=Nread, flat_time=2e-3, system=system)
adc = make_adc(num_samples=Nread, duration=2e-3, phase_offset=90*np.pi/180,delay=gx.rise_time, system=system)
gx_pre0 = make_trapezoid(channel='x', area=+(1.0 +0.0 )*gx.area / 2, duration=1e-3, system=system)
gx_prewinder = make_trapezoid(channel='x', area= +0.0 *gx.area / 2, duration=1e-3, system=system)

# ======
# CONSTRUCT SEQUENCE
# ======

rf_prep, _= make_block_pulse(flip_angle=180 * math.pi / 180, duration=1e-3, system=system)
#FLAIR
# seq.add_block(rf_prep)
# seq.add_block(make_delay(2.7))
# seq.add_block(gx_pre0)

# seq.add_block(make_delay(0.00031))
seq.add_block(make_delay(0.0009))

seq.add_block(rf1)

calc_duration(rf1)
calc_duration(gx_pre0)

seq.add_block(gx_pre0,make_delay(0.0041-calc_duration(rf1)-rf2.delay - rf2.t[-1]/2+ rf2.ringdown_time/2))

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63
    seq.add_block(rf2)
    seq.add_block(make_delay(0.0001))
    gp= make_trapezoid(channel='y', area=ii, duration=1e-3, system=system)
    gp_= make_trapezoid(channel='y', area=-ii, duration=1e-3, system=system)
    seq.add_block(gx_prewinder,gp)
    seq.add_block(adc,gx)
    seq.add_block(gx_prewinder,gp_)
    seq.add_block(make_delay(0.00008))



# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc,t_adc =seq.plot(clear=False)
#   
if 0:
    sp_adc,t_adc =seq.plot(clear=True)

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id +'.seq')

# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
from new_core.sim_data import VoxelGridPhantom, CustomVoxelPhantom
sz = [64, 64]

if 1:
    # (i) load a phantom object from file
    # obj_p = VoxelGridPhantom.load('../data/phantom2D.mat')
    obj_p = VoxelGridPhantom.load('../data/numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)
    # Manipulate loaded data
    obj_p.T2dash[:] = 30e-3
    # Store PD for comparison
    PD = obj_p.PD
    B0 = obj_p.B0
else:
    # or (ii) set phantom  manually to a pixel phantom
    obj_p = CustomVoxelPhantom(
        pos=[[-0.4, -0.4, 0], [-0.4, -0.2, 0], [-0.3, -0.2, 0], [-0.2, -0.2, 0], [-0.1, -0.2, 0]],
        PD=[1.0, 1.0, 0.5, 0.5, 0.5],
        T1=1.0,
        T2=0.1,
        T2dash=0.1,
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
signal, _= sim_external(obj=obj_p,plot_seq_k=[0,1], M_threshold=-1e-3)
# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())

# %% S6: MR IMAGE RECON of signal ::: #####################################
fig=plt.figure(); # fig.clf()
plt.subplot(411); plt.title('ADC signal')
spectrum=torch.reshape((signal),(Nphase,Nread)).clone().t()
kspace=spectrum
plt.plot(torch.real(signal),label='real')
plt.plot(torch.imag(signal),label='imag')


major_ticks = np.arange(0, Nphase*Nread, Nread) # this adds ticks at the correct position szread
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()

space = torch.zeros_like(spectrum)

# fftshift
spectrum=torch.fft.fftshift(spectrum,0); spectrum=torch.fft.fftshift(spectrum,1)
#FFT
space = torch.fft.ifft2(spectrum)
# fftshift
space=torch.fft.ifftshift(space,0); space=torch.fft.ifftshift(space,1)


plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace.numpy()))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.log(np.abs(kspace.numpy())))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space.numpy())); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space.numpy()),vmin=-np.pi,vmax=np.pi); plt.colorbar()

# % compare with original phantom obj_p.PD
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)

