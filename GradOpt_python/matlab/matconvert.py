# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:13:04 2019

@author: mzaiss
"""
import numpy
import scipy.io

import matplotlib.pyplot as plt

def randomCrop(img1, img2, img3, width, height):
    assert img1.shape[0] >= height
    assert img1.shape[1] >= width
    x = numpy.random.randint(0, img1.shape[1] - width)
    y = numpy.random.randint(0, img1.shape[0] - height)
    img1 = img1[y:y+height, x:x+width]
    img2 = img2[y:y+height, x:x+width]
    img3 = img3[y:y+height, x:x+width]
    return img1, img2, img3



mat = scipy.io.loadmat('C2P_FISP_phantom_results.mat')

t1=mat['t1im']
t2=mat['t2im']
pd=numpy.real(mat['m0im'])
phase=numpy.angle(mat['m0im'])

pd=pd[30:150,15:165]
phase=phase[30:150,15:165]
t1=t1[30:150,15:165]
t2=t2[30:150,15:165]

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


MP = numpy.zeros((120,150,3), dtype=numpy.float32)
MP[:,:,0]=pd
MP[:,:,1]=t1/1000
MP[:,:,2]=t2/1000
numpy.save('MP.npy',MP)

stop()
# %% ###     OPTIMIZATION functions phi and init ######################################################
#############################################################################  
data= numpy.zeros((40**2,16,16,3));
f,(axarr) = plt.subplots(40, 40, sharex='all', sharey='all')
for ii in range(40):
    for jj in range(40):
        pd_c,t1_c,t2_c = randomCrop(pd,t1,t2,16,16)
        data[ii*40+jj,:,:,0]=pd_c
        data[ii*40+jj,:,:,1]=t1_c
        data[ii*40+jj,:,:,2]=t2_c;
        axarr[ii,jj].imshow(pd_c)
     
numpy.save('num_brain_slice16x16.npy',data)
