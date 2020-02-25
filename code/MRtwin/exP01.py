# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:56:54 2020
@author: zaissmz
"""
# this is a comment

#%% this forms a section: run with CTRL+ENTER
# run full file F5
# run single current line F9  (or marked commands)
# restart kernel CTRL + . in console

#%% these are imports: libraries used in the script later.
import os, sys
import numpy as np
import scipy
import scipy.io
from  scipy import ndimage
import torch
from torch import optim
import matplotlib
import matplotlib.pyplot as plt

# to test if all your versions are fine, tested before were
#torch.__version__  : '1.3.0'
#np.__version__ 	 : '1.18.1'
#scipy.__version__: '1.4.1'
#matplotlib.__version__: '3.1.1'
print('torch.__version__ :',torch.__version__, '; tested 1.3.0' )
print('np.__version__ :',np.__version__ , '; tested 1.18.1' )
print('scipy.__version__:',scipy.__version__, '; tested 1.4.1' )
print('matplotlib.__version__ :',matplotlib.__version__, '; tested 3.1.1' )
	

#%% variables (int, float, bool)
A = 4.0
B = 3.14 *1e-2
C=A**2   # power
D=A**0.5 # square root
print(A,B,A+B,A*B,C,D)

do_real_scan = False
print(do_real_scan,not do_real_scan, (1>0),A<1)

#%% variable (string)
experiment_id = 'exA09_GRE_fully_relaxed' # single line string
#multi line string:
excercise = """   
A09.1. plot the k-space as an image
A09.2. excite only certain k space lines with 90 degree, other rf_event to 0
A09.3. try different flip angles.
A09.4. try different phantoms.
A09.5. try to prolong the echo time, what do you observe?
"""
print(experiment_id)

# formatting ( express variables as string)
#https://pyformat.info/
formatted_string="f={:.2}*exp(-{:.2}*t)+{}".format(5.0, 7.1,'constant')
print(formatted_string)

#%% numpy , library for many mathematical function
import numpy as np

C= 1+1j*1
R=np.real(C)
I=np.imag(C)
magn=np.abs(C)
phase=np.angle(C)*180/np.pi
print(C,R,I,magn,phase)

print(C,C.real,C.imag,np.abs(C),np.angle(C)*180/np.pi)

#%% vectors and arrays
sz = np.array([36,36])   # image size
print(sz,sz.shape)

array = np.array([[0.2,0.3],[0.1,2]])  # this could be an image matrix   
print(array,'shape=',array.shape)

array =  np.arange(0,11,1) #  (start,end,step)
print(array,'shape=',array.shape)

array = np.linspace(0,1,10)  # (start, end, number of entries)
print(array,'shape=',array.shape)

array = np.array([np.linspace(0,1,10),np.linspace(0,1,10)])  # this could be an image matrix   
print(array,'shape=',array.shape)

array = np.zeros(5)
print(array,'shape=',array.shape)

array = np.zeros([5,5])
print(array,'shape=',array.shape)


#%% array and vector operations I, simple broadcasting
array = np.zeros([5,5,2])  # this coudl be a complex image
print(array,'shape=',array.shape)
array[0,0,:]=1.0
#array[0,:,0]=2.0  # also try the following lines, line by line
#array[:,4,1]=3.0
#array[:,0,:]=4.0
#array[:,:,:]=5.0
#array[:]=0.123
print(array[:,:,0],'\n\n',array[:,:,1],'shape=',array.shape)

#%% array and vector operations II, advanced broadcasting
array = np.zeros([5,5,2])  # this coudl be a complex image
#array[:,-1,:]=1.0  #last entry
#array[:,-3,:]=1.0  # third last entry
#array[:,1:-2,:]=1.0 # from index one til -2 ( be careful, last value is not counted in)
#array[:,::2,:]=3.0 # all in steps of 2
#array[:,1::2,:]=3.0 # from 1 to end in steps of 2
#array[0,1::2,0]=np.linspace(0.5,0.6,2) # from 1 to end in steps of 2

array = np.ones([5,5,2]); array[:,1:-2,:]*=0  # initialize to ones, then multiply certain entries by 0

print(array[:,:,0],'\n\n',array[:,:,1],'shape=',array.shape)

#%%  ravel or flatten of an array
array = np.ones([5,3]) 
array[:,0]=0
array[:,2]=2
# np.flatten('F')
array.flatten('F')
# this is the same as 
np.transpose(array).flatten()
# or
np.transpose(array).ravel()
# or
np.ravel(array,order='F')

#%% function definition
array = np.ones ([5,5,2]) # lets assume this is a complex image, last dimension is just real and imaginary part
array*=1 # make real and image the same, ths magnitude should be sqrt(2) and angle should be 45°
print(array[:,:,0],'\n\n',array[:,:,1],'shape=',array.shape)

def magimg(x):  # function to get the magnitude image
    return np.sqrt(np.sum(np.abs(x)**2,2))

def phaseimg(x):  # function to get the magnitude image
    return np.angle(1j*x[:,:,1]+x[:,:,0])
#    return  np.arctan2(x[:,:,1],x[:,:,0]) # this is an alternative without using complex numbers

print(magimg(array),'shape=',magimg(array).shape)
print(phaseimg(array)*180/np.pi,'shape=',phaseimg(array).shape)

#%% if, else for
x=5
if x>0:
    print('condition fulfilled x>0')
elif x<-2:
    print('condition fulfilled x<-2')
else:
    print('condition not fulfilled x must be -2<x<0')
        
array = np.zeros([5,5,2])  # this could be a complex image
#alter entries by for loops
for i in range(5):
    for j in range(5):
        array[i,j,0]=i  #last entry
        array[i,j,1]=j  #last entry
print(array[:,:,0],'\n\n',array[:,:,1],'shape=',array.shape)


#%% pyplot 1

array = np.ones([5,3]) 
array[:,0]=0
array[:,2]=2
# np.flatten('F')
array.flatten('F')

# plot signal as a function of total time:
plt.subplot(413); plt.ylabel('signal')
time_axis=np.cumsum(array.flatten('F'))
plt.plot(time_axis,array.flatten('F'),label='real')
ax=plt.gca(); ax.grid()

#%% pyplot 2
img= np.random.rand(5,5,2)    # lets assume this is a complex image, last dimension is just real and imaginary part
img[:,:,1]*=0.1

plt.figure("""mag and phase images""")

plt.subplot(141), plt.title('real(img)')  # make subplots with title
ax=plt.imshow(img[:,:,0], interpolation='none')
fig = plt.gcf(); fig.colorbar(ax)  # add a colorbar

plt.subplot(142), plt.title('imag(img)')  # make subplots with title
ax=plt.imshow(img[:,:,1], interpolation='none')
fig = plt.gcf(); fig.colorbar(ax)  # add a colorbar

plt.subplot(143), plt.title('abs(img)')  # make subplots with title
ax=plt.imshow(magimg(img), interpolation='none')
fig = plt.gcf(); fig.colorbar(ax)  # add a colorbar

plt.subplot(144), plt.title('phase(img)')  # make subplots with title
ax=plt.imshow(phaseimg(img), interpolation='none')
fig = plt.gcf(); fig.colorbar(ax)  # add a colorbar

fig.set_size_inches(10, 2)    # resize the figure propery to match your screen
plt.show()

#%% save and load variables
import os, sys
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt

# using numpy
x = np.arange(10)
try:
    os.mkdir('./out')
except:
    pass
outfile='./out/saved.npy'
np.save(outfile, x)

y=np.load(outfile)
print(y,'shape=',y.shape)



# using scipy and .mat files
phantom = scipy.io.loadmat('../../data/phantom2D.mat')['phantom_2D']
brain_phantom = scipy.io.loadmat('../../data/numerical_brain_cropped.mat')['cropped_brain']

plt.figure("""phantom""")
param=['PD','T1','T2','dB0','rB1']
for i in range(5):
    plt.subplot(151+i), plt.title(param[i])
    ax=plt.imshow(phantom[:,:,i], interpolation='none')
    fig = plt.gcf()
    fig.colorbar(ax) 
fig.set_size_inches(18, 3)
plt.show()
    
    


#%% we also use the torch library which has greate parallel computing abilities
# for torch the device where the calculations are perfomed must be set for each variable
# this is very important if you wnat to run simulation on GPU.
import torch
double_precision = False
use_gpu = 0
gpu_dev = 0 

# device setup
def setdevice(x):
    if double_precision:
        x = x.double()
    else:
        x = x.float()
    if use_gpu:
        x = x.cuda(gpu_dev)    
    return x 

A=torch.zeros(5)
setdevice(A)

#%%  torch and numpy
# many things can be done in torch directly e.g:
A=torch.ones(5)*3
B=np.ones(5)*5
# sometimes things need to be done in numpy or in torch, however these arrays are not compatible, e.g:
A-B

#%% to solve this one of the  two functions are necessary:
# torch to numpy
def tonumpy(x):
    return x.detach().cpu().numpy()
# and the builtin function torch.from_numpy
    
C=tonumpy(A)-B
print(C, type(C),'This is a numpy array')


D=A-torch.from_numpy(B)
print(D, type(D),'This is a torch tensor')

print(tonumpy(D), type(tonumpy(D)),'This is again a numpy array')


#%% exercises:  fix the followig codes ( currently they will throw and error)

#%% exercise
array = np.zeros(5,5)  # aim : generate 5x5 matrix if zeroes
print(array,'shape=',array.shape)

#%% exercise
array = np.linspace(0,5,1) #  aim: generate array [0,1,2,3,4,5]
print(array,'shape=',array.shape)

#%% exercise
array =  np.arange(0,5,5) #  aim: generate array [0,1,2,3,4,5]
print(array,'shape=',array.shape)

#%% exercise
array = np.zeros([5,5,2])  
array[:,5,1]=3.0  # aim: we want the last column of the array all filled with 3.0
print(array,'shape=',array.shape)

#%% exercise
array = np.zeros([5,5,2])
array[0,::2,0]=np.ones(2) # aim: fill every second column with vlue given by linspace

#%% exercise, fix this code
A=np.linspace(0,1,5)
B= torch.arange(0,5,1)

B*A


