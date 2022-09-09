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
from GradOpt_python.new_core.util import plot_kspace_trajectory
from GradOpt_python.new_core import util
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
 
# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
from new_core.sim_data import SimData
sz=[64,64]
# (i) load a phantom object from file
if 1:
    obj_p = SimData.load('../data/phantom2D.mat')
    obj_p = SimData.load('../data/numerical_brain_cropped.mat')
    obj_p.T2dash[:] = 30e-3
    obj_p = obj_p.resize(sz[0],sz[1],1)
      
# obj_p.plot_sim_data()

# lets just use the PD as an MRI image 
space= np.squeeze(util.to_full(obj_p.PD, obj_p.mask))
space = np.fft.ifftshift(space)
kspace_adc = np.fft.ifft2(space)
kspace_adc = np.fft.ifftshift(kspace_adc) # this is a kspace as it woudk come from an ADC, with low frequencies centered.

plt.imshow(np.log(np.abs(kspace_adc)))
          
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

kspace_full = np.fft.ifftshift(kspace_adc)  # high  frequencies centered as FFT needs it

kspace=kspace_full
recon_nufft = (np.fft.fftshift(np.fft.fft2(kspace_full))) # fully sampled recon


#%% S6.3 undersampling and undersampled reconstruction
# space= space/ np.linalg.norm(space[:])   # normalization of the data somethimes helps

# parameters of iterative reconstructio using total variation denoising  
denoising_strength = 10e-6
number_of_iterations = 8000

# parameters of random subsampling pattern
percent = 0.25        # this is the amount of data that is randomly measured
square_size = 16      # size of square in center of k-space 


# generate a random subsampling pattern
np.random.seed(np.random.randint(100))
pattern = np.random.random_sample(kspace.shape)
pattern=pattern<percent  # random data

pattern[sz[0]//2-square_size//2:sz[0]//2+square_size//2,sz[0]//2-square_size//2:sz[0]//2+square_size//2] = 1   # square in center of k-space 
pattern = np.fft.fftshift(pattern) # high  frequencies centered as kspace and as FFT needs it

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
