# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:21:25 2020

@author: fmglang
"""

import numpy as np
import torch
import scipy.io as sio
import cv2

def simulate_gauss_sensitivities(NCoils, sz, locations=None, widths=None):
# simulate coil sensitivities as Gaussians of given location and width (in pixel coordinates)
    if locations is None: # distribute locations equally across images (along phase enc direction)
        locations = np.zeros([NCoils,2])
        locations[:,0] = np.linspace(0, sz[0]-1, NCoils)
        locations[:,1] = sz[1] / 2
    if widths is None: # automatically determine Gaussian widths (somehow arbitrary)
        widths = np.zeros([NCoils,2])
        widths[:,1] = sz[1] 
        widths[:,0] = sz[0] / NCoils * 3 # overlap can be adjusted here
        
    B1 = torch.zeros((NCoils,sz[0],sz[1]))
    XX,YY = torch.meshgrid(torch.tensor(range(sz[0]), dtype=torch.float32), torch.tensor(range(sz[1]), dtype=torch.float32))

    for ii in range(NCoils): # not normalized...!
        B1[ii,:,:] = torch.exp(-(((XX-locations[ii][0])/widths[ii][0])**2)-(((YY-locations[ii][1])/widths[ii][1])**2))
    return B1

def simulate_linear_phases(NCoils, sz, k=(-2,2)):
    B1 = torch.zeros((NCoils,sz[0],sz[1]))
    XX,YY = torch.meshgrid(torch.tensor(range(sz[0]), dtype=torch.float32), torch.tensor(range(sz[1]), dtype=torch.float32))
    
    for ii in range(NCoils): # not normalized...!
        B1[ii,:,:] = torch.randint(k[0],k[1],(1,1)) * XX + torch.randint(k[0],k[1],(1,1)) * YY
    return B1

def load_external_coil_sensitivities(path, NCoils, sz):
    loaded = sio.loadmat(path)
    B1minus = loaded['B1minus']
    
    B1minus_rescaled = torch.zeros((NCoils, *sz), dtype=torch.complex64)
    for i in range(NCoils):
        re = cv2.resize(np.real(B1minus[i,:,:]), dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
        im = cv2.resize(np.imag(B1minus[i,:,:]), dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
        
        B1minus_rescaled[i,:,:] = torch.tensor(re) + 1j * torch.tensor(im)
        
    return B1minus_rescaled

def load_external_coil_sensitivities3D(path, NCoils, sz):
    loaded = sio.loadmat(path)
    B1minus = loaded['B1minus']
    
    B1minus_rescaled = torch.zeros((NCoils, *sz), dtype=torch.complex64)
    for i in range(NCoils):
        for j in range(B1minus_rescaled.shape[-1]):
            re = cv2.resize(np.real(B1minus[i,:,:,j]), dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
            im = cv2.resize(np.imag(B1minus[i,:,:,j]), dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
            
            B1minus_rescaled[i,:,:,j] = torch.tensor(re) + 1j * torch.tensor(im)
        
    return B1minus_rescaled

def load_fastMRI_coil_sensitivities(path, NCoils, sz, slc):
    loaded = sio.loadmat(path)
    B1minus = loaded['sens_maps'][:,:,:,slc]
    
    B1minus_rescaled = torch.zeros((NCoils, *sz), dtype=torch.complex64)
    for i in range(NCoils):
        re = cv2.resize(np.real(B1minus[i,:,:]), dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
        im = cv2.resize(np.imag(B1minus[i,:,:]), dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
        
        B1minus_rescaled[i,:,:] = torch.tensor(re) + 1j * torch.tensor(im)
        
    return B1minus_rescaled