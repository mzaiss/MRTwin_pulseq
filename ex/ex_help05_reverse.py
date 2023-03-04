# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:01:40 2020

@author: zaissmz
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

# %% ##########################################################################
# reverse in numpy
AA = np.zeros([24, 24])
AA[7, :] = np.linspace(10, 24, 24)
AA[:, 5] = np.linspace(10, 24, 24)

plt.figure()
plt.imshow(AA)
plt.title('original')

BB = AA.copy()
BB = AA[::-1, :]  # reverse manipulation 1, reverse first dimension
# BB = np.flip(AA, 0) # does the same

plt.figure()
plt.imshow(BB)
plt.title('reverse manipulation 1')

CC = AA.copy()
CC[::2, :] = AA[::2, ::-1]  # reverse manipulation 2, only every second line
# CC[::2, :] = np.flip(AA[::2, :], 0)  # does the same

plt.figure()
plt.imshow(CC)
plt.title('reverse manipulation 2')

# %% shift in numpy
AA = np.zeros([24, 24])
AA[7, :] = np.linspace(10, 24, 24)
AA[:, 5] = np.linspace(10, 24, 24)

plt.figure()
plt.subplot(131)
plt.imshow(AA)
plt.title('original')

BB = AA.copy()
BB[:-1, ] = AA[1:, :]  # shift manipulation 1, shift part of array

plt.subplot(132)
plt.imshow(BB)
plt.title('shift manipulation 1')

CC = AA.copy()
CC = np.roll(AA, -1, 0)  # shift manipulation 2, roll the array

plt.subplot(133)
plt.imshow(CC)
plt.title('shift manipulation 2 - np.roll')


# %% ##########################################################################
# reverse in torch
AA = torch.zeros([24, 24])
AA[7, :] = torch.linspace(10, 24, 24)
AA[:, 5] = torch.linspace(10, 24, 24)

plt.figure()
plt.imshow(AA)
plt.title('original')

BB = AA.clone()
BB = torch.flip(AA, [0])  # reverse manipulation 1, reverse first dimension
# BB = AA[::-1, :]  # does not work in torch

plt.figure()
plt.imshow(BB)
plt.title('reverse manipulation 1')

CC = AA.clone()
# reverse manipulation 2, only every second line
CC[::2, :] = torch.flip(AA[::2, :], [0])
# CC[::2, :] = AA[::2, ::-1]  # does not work in torch

plt.figure()
plt.imshow(CC)
plt.title('reverse manipulation 2')

# %% shift in torch
AA = torch.zeros([24, 24])
AA[7, :] = torch.linspace(10, 24, 24)
AA[:, 5] = torch.linspace(10, 24, 24)

plt.figure()
plt.subplot(131)
plt.imshow(AA)
plt.title('original')

BB = AA.clone()
BB[:-1, ] = AA[1:, :]  # shift manipulation 1, shift part of array

plt.subplot(132)
plt.imshow(BB)
plt.title('shift manipulation 1')

CC = AA.clone()
CC = torch.roll(AA, -1, 0)  # shift manipulation 2, roll the array

plt.subplot(133)
plt.imshow(CC)
plt.title('shift manipulation 2 - torch.roll')


# %%
# kspace[:, 1::2] = torch.flip(kspace_adc[:, 1::2], [0])
# kspace[1:, 1::2] = kspace[:-1, 1::2]
