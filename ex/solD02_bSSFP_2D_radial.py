experiment_id = 'exD01_bSSFP_2D'

# %% S0. SETUP env
import sys,os
os.chdir(os.path.abspath(os.path.dirname(__file__)))  #  makes the ex folder your working directory
sys.path.append(os.path.dirname(os.getcwd()))         #  add required folders to path
mpath=os.path.dirname(os.getcwd())
c1=r'codes'; c2=r'codes\GradOpt_python'; c3=r'codes\scannerloop_libs' #  add required folders to path
sys.path += [rf'{mpath}\{c1}',rf'{mpath}\{c2}',rf'{mpath}\{c3}']

## imports for simulation
from GradOpt_python.pulseq_sim_external import sim_external
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
sz=(64,64)   # spin system size / resolution
Nread = 64    # frequency encoding steps/samples
Nphase = 64    # phase encoding steps/samples

# Define rf events
rf1, _,_ = make_sinc_pulse(flip_angle=6 * math.pi / 180, duration=1e-3,slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4, system=system)
# rf1, _= make_block_pulse(flip_angle=90 * math.pi / 180, duration=1e-3, system=system)

# Define other gradients and ADC events
gx = make_trapezoid(channel='x', flat_area=Nread, flat_time=5e-3, system=system)
gy = make_trapezoid(channel='y', flat_area=Nread, flat_time=5e-3, system=system)

gx_pre = make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
gy_pre = make_trapezoid(channel='y', area=-gx.area / 2, duration=1e-3, system=system)

adc = make_adc(num_samples=Nread, duration=5e-3, phase_offset=0*np.pi/180,delay=gx.rise_time, system=system)

rf_phase = 180
rf_inc = 180

# ======
# CONSTRUCT SEQUENCE
# ======
sdel=1e-0

rf0, _,_ = make_sinc_pulse(flip_angle=6/2 * math.pi / 180, duration=1e-3,slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4, system=system)
seq.add_block(rf0)
seq.add_block(make_delay(3e-3))

for ii in range(-Nphase//2, Nphase//2):  # e.g. -64:63

    rf1.phase_offset = rf_phase / 180 * np.pi   # set current rf phase
    
    adc.phase_offset = rf_phase / 180 * np.pi  # follow with ADC
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]        # increment additional pahse

    seq.add_block(rf1)
    
    
    gx = make_trapezoid(channel='x', flat_area=-Nread*np.sin(ii/Nphase*np.pi), flat_time=5e-3, system=system)
    gy = make_trapezoid(channel='y', flat_area=Nread*np.cos(ii/Nphase*np.pi), flat_time=5e-3, system=system)
    
    gx_pre = make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
    gy_pre = make_trapezoid(channel='y', area=-gy.area / 2, duration=1e-3, system=system)
    
    
    seq.add_block(gx_pre,gy_pre)
    seq.add_block(adc,gx,gy)
    # seq.add_block(adc,gx,gy)
    seq.add_block(gx_pre,gy_pre)
    # seq.add_block(make_delay(10))

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
seq.set_definition('FOV', [fov, fov, slice_thickness]*1000)
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
    obj_p.D *= 0
    # Store PD for comparison
    PD = obj_p.PD
    B0 = obj_p.B0
else:
    # or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
    obj_p = CustomVoxelPhantom(
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
signal, kspace_loc,= sim_external(obj=obj_p,plot_seq_k=[0,1])   
# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())

# %% S6: MR IMAGE RECON of signal ::: #####################################
fig=plt.figure(); # fig.clf()
plt.subplot(411); plt.title('ADC signal')
spectrum=torch.reshape((signal),(Nphase,Nread)).clone().t()
kspace=spectrum
kspace_adc=spectrum
plt.plot(torch.real(signal),label='real')
plt.plot(torch.imag(signal),label='imag')

major_ticks = np.arange(0, Nphase*Nread, Nread) # this adds ticks at the correct position szread
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()

space = torch.zeros_like(spectrum)

if 0: # FFT
    # fftshift
    spectrum=torch.fft.fftshift(spectrum,0); spectrum=torch.fft.fftshift(spectrum,1)
    #FFT
    space = torch.fft.ifft2(spectrum)
    # fftshift
    space=torch.fft.ifftshift(space,0); space=torch.fft.ifftshift(space,1)


if 1: # NUFFT
    import scipy.interpolate
    grid = kspace_loc[:,:2]
    Nx=64
    Ny=64
    
    X, Y = np.meshgrid(np.linspace(0,Nx-1,Nx) - Nx / 2, np.linspace(0,Ny-1,Ny) - Ny/2)
    grid = np.double(grid.numpy())
    grid[np.abs(grid) < 1e-3] = 0
    
    plt.subplot(347); plt.plot(grid[:,0].ravel(),grid[:,1].ravel(),'rx',markersize=3);  plt.plot(X,Y,'k.',markersize=2);
    plt.show()
    
    spectrum_resampled_x = scipy.interpolate.griddata((grid[:,0].ravel(), grid[:,1].ravel()), np.real(signal.ravel()), (X, Y), method='cubic')
    spectrum_resampled_y = scipy.interpolate.griddata((grid[:,0].ravel(), grid[:,1].ravel()), np.imag(signal.ravel()), (X, Y), method='cubic')

    kspace_r=spectrum_resampled_x+1j*spectrum_resampled_y
    kspace_r[np.isnan(kspace_r)] = 0
    
    # fftshift
    # kspace_r = np.roll(kspace_r,Nx//2,axis=0)
    # kspace_r = np.roll(kspace_r,Ny//2,axis=1)
    kspace_r_shifted=np.fft.ifftshift(kspace_r,0); kspace_r_shifted=np.fft.ifftshift(kspace_r_shifted,1)
             
    space = np.fft.ifft2(kspace_r_shifted)
    space=np.fft.ifftshift(space,0); space=np.fft.ifftshift(space,1)

space=np.transpose(space)
plt.subplot(345); plt.title('k-space')
plt.imshow(np.abs(kspace))
plt.subplot(349); plt.title('k-space_r')
plt.imshow(np.abs(kspace_r))

plt.subplot(346); plt.title('FFT-magnitude')
plt.imshow(np.abs(space)); plt.colorbar()
plt.subplot(3,4,10); plt.title('FFT-phase')
plt.imshow(np.angle(space),vmin=-np.pi,vmax=np.pi); plt.colorbar()

# % compare with original phantom obj_p.PD
plt.subplot(348); plt.title('phantom PD')
plt.imshow(PD)
plt.subplot(3,4,12); plt.title('phantom B0')
plt.imshow(B0)

