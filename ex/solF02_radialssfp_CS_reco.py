experiment_id = 'exF02_undersampled_radial'

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
Nphase = 32    # phase encoding steps/samples

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

obj_p.plot()
# Convert Phantom into simulation data
obj_p = obj_p.build()


# %% S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot
signal, kspace_loc,= sim_external(obj=obj_p,plot_seq_k=[0,1])   
# plot the result into the ADC subplot
sp_adc.plot(t_adc,np.real(signal.numpy()),t_adc,np.imag(signal.numpy()))
sp_adc.plot(t_adc,np.abs(signal.numpy()))
# seq.plot(signal=signal.numpy())
          
kspace_adc=torch.reshape((signal),(Nphase,Nread)).clone().t()
spectrum=kspace_adc

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
    
    
    
    # k-space sampling pattern needed for the CS algorithms
    pattern_resampled=np.zeros([sz[0],sz[1]])
    gridx=grid[:,0].ravel()
    gridy=grid[:,1].ravel()
    for ii in range(len(gridx)):
        pattern_resampled[int(gridx[ii]),int(gridy[ii])]=1
    plt.imshow(pattern_resampled)
    plt.show()
    # end sampling pattern
    
             

#%% ############################################################################
## S6: compressed sensing MR reconstruction of undersampled signal ::: #####################################
## S6.1: function definitions
import pywt
from skimage.restoration import denoise_tv_chambolle


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
	cA, (cH, cV, cD)  = pywt.dwt2(current, 'haar')
	#Shrink
	shrink(cA, epsilon)
	shrink(cH, epsilon)
	shrink(cV, epsilon)
	shrink(cD, epsilon)
	wavelet = cA, (cH, cV, cD)
	# return inverse WT
	return pywt.idwt2(wavelet, 'haar')
	

def updateData(k_space, pattern, current, step,i):
    # go to k-space
    update = np.fft.ifft2(np.fft.fftshift(current))
    # compute difference
    update = k_space - (update * pattern)
    print("i: {}, consistency RMSEpc: {:3.6f}".format(i,np.abs(update[:]).sum()*100))
    # return to image space
    update = np.fft.fftshift(np.fft.fft2(update))
    update = current + (step * update)  # improve current estimation by consitency
    return update


## S6.2: preparation and conventional fully sampled reconstruction

kspace_full = np.fft.ifftshift(kspace_r)  # high  frequencies centered as FFT needs it

kspace=kspace_full
recon_nufft = (np.fft.fftshift(np.fft.fft2(kspace_full))) # fully sampled recon


#%% S6.3 undersampling and undersampled reconstruction
# kspace_full= kspace_full/ np.linalg.norm(kspace_full[:])   # normalization of the data somethimes helps

# parameters of iterative reconstructio using total variation denoising  
denoising_strength = 5e-5
number_of_iterations = 8000

# parameters of random subsampling pattern
# percent = 0.25        # this is the amount of data that is randomly measured
# square_size = 16      # size of square in center of k-space 


# # generate a random subsampling pattern
# np.random.seed(np.random.randint(100))
# pattern = np.random.random_sample(kspace.shape)
# pattern=pattern<percent  # random data

# pattern[sz[0]//2-square_size//2:sz[0]//2+square_size//2,sz[0]//2-square_size//2:sz[0]//2+square_size//2] = 1   # square in center of k-space 
# pattern = np.fft.fftshift(pattern) # high  frequencies centered as kspace and as FFT needs it

pattern= pattern_resampled

kspace = kspace_full *pattern  # apply the undersampling pattern

actual_measured_percent =np.count_nonzero(pattern) / pattern.size *100  #  calculate the actually measured data in percent

## actual iterative reconstruction algorithm 
current = np.zeros(kspace.size).reshape(kspace.shape)
current_shrink = np.zeros(kspace.size).reshape(kspace.shape)
first = updateData(kspace, pattern, current, 1,0)
current_shrink=first
all_iter = np.zeros((kspace.shape[0],kspace.shape[1],number_of_iterations))

i = 0
while i < number_of_iterations:
    current = updateData(kspace, pattern, current_shrink, 0.1,i)
   
    current_shrink = denoise_tv_chambolle(abs(current), denoising_strength)
    # current_shrink = waveletShrinkage(abs(current), denoising_strength)
    
    all_iter[:,:,i]=current
    i = i + 1; 
		
## Plotting

pattern_vis = np.fft.fftshift(pattern * 256)

fig=plt.figure(dpi=90)
plt.subplot(321)
plt.set_cmap(plt.gray())
plt.imshow(abs(recon_nufft)); plt.ylabel('recon_full')
plt.subplot(322)
plt.set_cmap(plt.gray())
plt.imshow(abs(pattern_vis)); plt.ylabel("pattern_vis"); plt.title("{:.1f} % sampled".format(actual_measured_percent))
plt.subplot(323)
plt.set_cmap(plt.gray())
plt.imshow(abs(first)); plt.ylabel('first iter (=NUFFT)')
plt.subplot(325)
plt.set_cmap(plt.gray())
plt.imshow(abs(current_shrink)) ; plt.ylabel('final recon')
plt.subplot(324)
plt.set_cmap(plt.gray())
plt.imshow(np.log(abs(np.fft.fftshift(kspace_full)))); plt.ylabel('kspace_nufft')
plt.subplot(326)
plt.set_cmap(plt.gray())
plt.imshow(np.log(abs(np.fft.fftshift((kspace))))); plt.ylabel('kspace*pattern')
plt.show()



#%% Plot all iter
idx=np.linspace(1,all_iter.shape[2],25)-1       # make 25 example iterations
red_iter=all_iter[:,:,tuple(idx.astype(int))]   # choose them from all iters
Tot=red_iter.shape[2]
Rows = Tot // 5 
if Tot % 5 != 0:
    Rows += 1
Position = range(1,Tot + 1) # Position index

fig = plt.figure()
for k in range(Tot):
  ax = fig.add_subplot(Rows,5,Position[k])
  ax.imshow((abs((red_iter[:,:,k])))); plt.title('iter {}'.format(idx[k].astype(int)))
  print(k)
plt.show()
