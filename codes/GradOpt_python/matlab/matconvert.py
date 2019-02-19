# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:13:04 2019

@author: mzaiss
"""
import numpy
import scipy.io

import matplotlib.pyplot as plt

def randomCrop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = numpy.random.randint(0, img.shape[1] - width)
    y = numpy.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask



mat = scipy.io.loadmat('C2P_FISP_phantom_results.mat')

t1=mat['t1im']
t2=mat['t2im']
pd=numpy.real(mat['m0im'])
phase=numpy.angle(mat['m0im'])

f,(axarr) = plt.subplots(2, 2, sharex='all', sharey='all')

axarr[0, 0].imshow(pd)
axarr[0, 0].set_title('M0')
axarr[0, 1].imshow(phase)
axarr[0, 1].set_title('phase')
axarr[1, 0].imshow(t1)
axarr[1, 0].set_title('T1')
axarr[1, 1].imshow(t2)
axarr[1, 1].set_title('T2')
f.set_size_inches(6, 5)
plt.show()     

f,(axarr) = plt.subplots(10, 10, sharex='all', sharey='all')
for ii in range(10):
    for jj in range(10):
        x,y = randomCrop(t1,t1,16,16)
        axarr[ii,jj].imshow(x)
     

