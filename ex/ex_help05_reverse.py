# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:01:40 2020

@author: zaissmz
"""
import numpy as np
from matplotlib import pyplot as plt

#%% ############################################################################
# reverse in numpy
AA=np.zeros([24,24])
AA[7,:]=np.linspace(10,24,24);
AA[:,5]=np.linspace(10,24,24);

plt.figure(), plt.imshow(AA); plt.title('original')

BB=AA.copy()
BB=AA[::-1,:] # reverse manipulation 1

plt.figure(), plt.imshow(BB); plt.title('reverse manipulation 1')

CC=AA.copy()
CC[::2,:]=AA[::2,::-1]# reverse manipulation 2

plt.figure(), plt.imshow(CC); plt.title('reverse manipulation 2')

#%% ############################################################################
# reverse in numpy




#%%  shift in numpy

AA[::2,1:]=AA[::2,:-1]
print(AA)


kspace[:,1::2]=torch.flip(kspace_adc[:,1::2],[0])
kspace[1:,1::2]=kspace[:-1,1::2]