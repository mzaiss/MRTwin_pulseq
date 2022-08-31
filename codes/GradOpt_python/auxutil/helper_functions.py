"""
Created on Thu Nov  5 11:34:12 2020

@author: weinmusn
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
import cv2

global double_precision
global use_gpu

def e(gt,x):
    return 100*np.linalg.norm((gt-x).ravel())/np.linalg.norm(gt.ravel())
    
# torch to numpy
def tonumpy(x):
    return x.detach().cpu().numpy()

# get magnitude image
def magimg(x):
    if torch.is_tensor(x):
        return torch.sqrt(torch.sum(torch.abs(x)**2,-1))
    else:
        return np.sqrt(np.sum(np.abs(x)**2,-1))

def magimg_torch(x):
  return torch.sqrt(torch.sum(torch.abs(x)**2,1))

def tomag_torch(x):
    return torch.sqrt(torch.sum(torch.abs(x)**2,-1)+1e-8)

# device setter
def setdevice(x):
    if double_precision:
        x = x.double()
    else:
        x = x.float()
    if use_gpu:
        x = x.cuda(gpu_dev)    
    return x
    
def imshow(x, title=None):
    plt.imshow(x, interpolation='none')
    if title != None:
        plt.title(title)
    plt.ion()
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    plt.show()     

def stop():
    sys.tracebacklimit = 0
    class ExecutionControl(Exception): pass
    raise ExecutionControl('stopped by user')
    sys.tracebacklimit = 1000
    
def signal_normilize(S):
    mean_S = torch.mean(S)
    std_S = torch.std(S)    

# Calculation of chemical shift artifact, fat-water in pixel (B0 in T)
def chemical_shift(event_time_adc,B0=3.0):
    rBW_pixel = 1/np.sum(tonumpy(event_time_adc))
    if B0 == 3:
        freq_diff = 430 # Frequence difference between water and fat in Hz
    elif B0 == 1.5:
        freq_diff = 215
    return (rBW_pixel,rBW_pixel/freq_diff)

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def phaseimg(x):
    return np.angle(1j*x[:,:,1]+x[:,:,0])

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def fftshift_torch(x):
    axes = tuple(range(x.ndim))
    shift = [dim // 2 for dim in x.shape]
    return torch.roll(x,shift,axes)

def get_mask(in_vivo,sz,phantom,threshold=0.2,seg=0,method='step',aver=False,parameter='T1'): # 0 - just mask, 1 - seg in gm, wm, and csf, >2 - equal distributed regions; method - 'step': same step size, method - 'number': same number per step, aver - False: No average calculation
    if seg == 0:
        if in_vivo == True:
            mask = torch.ones(phantom[0,:,:].shape)
            mask[phantom[0,:,:] < threshold] = 0 # The last one is the most relaxed one, scaling factor
        else:
            mask = torch.ones([sz[0],sz[1],1]) # 0 - GM, 1 - WM, 2 - CSF
            mask[phantom[:,:,4] <= 0.61,:,0] = 0 # B1
            mask[phantom[:,:,0] <= threshold,:,0] = 0 # PD
    elif seg == 1:
        mask = torch.ones([sz[0],sz[1],3]) # 0 - GM, 1 - WM, 2 - CSF
        mask[phantom[:,:,4] < 0.61,:,:] = 0 # B1
        mask[phantom[:,:,0] <= threshold,:] = 0 # PD
        if parameter=='T1':
            mask[phantom[:,:,1] > 0.9,:,0] = 0 # T1
            mask[phantom[:,:,1] <= 0.9,:,1] = 0 # T1
            mask[phantom[:,:,1] > 1.7,:,1] = 0 # T1
            mask[phantom[:,:,1] <= 1.7,:,2] = 0 # T1
        elif parameter=='T2':
            mask[phantom[:,:,2] > 0.1,:,0] = 0 # T2
            mask[phantom[:,:,2] <= 0.1,:,1] = 0 # T2
            mask[phantom[:,:,2] > 0.5,:,1] = 0 # T2
            mask[phantom[:,:,2] <= 0.5,:,2] = 0 # T2
    elif seg == 'all':
        mask_num = torch.ones([sz[0],sz[1],1]) 
        mask_num[phantom[:,:,4] <= 0.61,:] = 0 # B1
        mask_num[phantom[:,:,0] <= threshold,:] = 0 # PD
        num = np.count_nonzero(tonumpy(mask_num[:,:,0]))
        mask = torch.zeros([sz[0],sz[1],num]) 
        locx,locy = np.unravel_index(np.argsort(phantom[:,:,1]*tonumpy(mask_num[:,:,0]),axis=None),phantom[:,:,1].shape)
        for jj in np.arange(-1,-num-1,-1):
            mask[locx[jj],locy[jj],-jj-1] = 1 # T1
    elif method == 'step':
        mask = torch.ones([sz[0],sz[1],seg]) 
        mask[phantom[:,:,4] <= 0.61,:,:] = 0 # B1
        mask[phantom[:,:,0] <= threshold,:] = 0 # PD
        T1max = np.max(phantom[:,:,1])
        limit = 0.1
        T1min = np.ma.min(np.ma.MaskedArray(phantom[:,:,1], phantom[:,:,1]<limit))
        T1step = (T1max-T1min)/seg
        for jj in np.arange(0,seg):
            mask[phantom[:,:,1] <= jj*T1step+T1min,:,jj] = 0 # T1
            mask[phantom[:,:,1] > (jj+1)*T1step+T1min,:,jj] = 0 # T1
    elif method == 'number':
        mask_num = torch.ones([sz[0],sz[1],1]) 
        mask_num[phantom[:,:,4] <= 0.61,:] = 0 # B1
        mask_num[phantom[:,:,0] <= threshold,:] = 0 # PD
        num = np.count_nonzero(tonumpy(mask_num[:,:,0]))
        if seg % 2:
            raise ValueError('Choose even number of segements')
        mask = torch.zeros([sz[0],sz[1],seg]) 
        segT1 = int(np.sqrt(seg))
        num_seg = num//segT1
        num_corr = np.mod(num,segT1)
        if num_seg == 0:
            raise ValueError('Number of segments higher than nonzero voxels. Reduce nummber of segments!')
        if parameter =='T1':
            locx,locy = np.unravel_index(np.argsort(-phantom[:,:,1]*tonumpy(mask_num[:,:,0]),axis=None),phantom[:,:,1].shape)
        elif parameter =='T2':
            locx,locy = np.unravel_index(np.argsort(-phantom[:,:,2]*tonumpy(mask_num[:,:,0]),axis=None),phantom[:,:,2].shape)
        corr = 0
        shift_tot = 0
        for jj in np.arange(segT1):
            shift = 0
            if corr >= segT1:
                shift = 1
                corr -= segT1
            corr += num_corr
            if jj == segT1-1 and np.mod(num,segT1) != 0:
                shift += 1
            mask[locx[jj*num_seg+shift_tot:jj*num_seg+num_seg+shift+shift_tot],locy[jj*num_seg+shift_tot:jj*num_seg+num_seg+shift+shift_tot],jj*segT1] = 1
            shift_tot += shift
            
        segB0 = int(np.sqrt(seg))
        if parameter == 'T1':
            num_mask = np.count_nonzero(torch.from_numpy(phantom[:,:,1]).unsqueeze(2)*mask,axis=(0,1))
        elif parameter =='T2': 
            num_mask = np.count_nonzero(torch.from_numpy(phantom[:,:,2]).unsqueeze(2)*mask,axis=(0,1))
        for ii in np.arange(segB0): 
            num_seg = num_mask[ii*segT1]//segB0
            num_corr = np.mod(num_mask[ii*segT1],segB0)
            if num_seg == 0:
                raise ValueError('Number of segments higher than nonzero voxels. Reduce nummber of segments!')
            tmp = mask[:,:,ii*segT1]
            tmp[tmp == 0] = float('NaN')
            locx,locy = np.unravel_index(np.argsort(phantom[:,:,3]*tonumpy(tmp),axis=None),phantom[:,:,3].shape)
            corr = 0
            shift_tot = 0
            mask[:,:,ii*segT1] = 0
            for jj in np.arange(segB0):
                shift = 0
                if corr >= segB0:
                    shift = 1
                    corr -= segB0
                corr += num_corr
                if jj == segB0-1 and np.mod(num_mask[ii*segT1],segB0) != 0:
                    shift += 1
                mask[locx[jj*num_seg+shift_tot:jj*num_seg+num_seg+shift+shift_tot],locy[jj*num_seg+shift_tot:jj*num_seg+num_seg+shift+shift_tot],ii*segT1+jj] = 1
                shift_tot += shift
        if num != np.count_nonzero(mask):
            raise ValueError('Number of segments higher than nonzero voxels. Reduce nummber of segments!')
    if aver:
        PD_average = torch.sum(torch.from_numpy(phantom[:,:,0]).unsqueeze(2)*mask,axis=(0,1))/torch.count_nonzero(torch.from_numpy(phantom[:,:,0]).unsqueeze(2)*mask,axis=(0,1))
        T1_average = torch.sum(torch.from_numpy(phantom[:,:,1]).unsqueeze(2)*mask,axis=(0,1))/torch.count_nonzero(torch.from_numpy(phantom[:,:,1]).unsqueeze(2)*mask,axis=(0,1))
        T2_average = torch.sum(torch.from_numpy(phantom[:,:,2]).unsqueeze(2)*mask,axis=(0,1))/torch.count_nonzero(torch.from_numpy(phantom[:,:,2]).unsqueeze(2)*mask,axis=(0,1))
        B1_average = torch.sum(torch.from_numpy(phantom[:,:,4]).unsqueeze(2)*mask,axis=(0,1))/torch.count_nonzero(torch.from_numpy(phantom[:,:,4]).unsqueeze(2)*mask,axis=(0,1))
        B0_average = torch.sum(torch.from_numpy(phantom[:,:,3]).unsqueeze(2)*mask,axis=(0,1))/torch.count_nonzero(torch.from_numpy(phantom[:,:,3]).unsqueeze(2)*mask,axis=(0,1))
        return {'mask':setdevice(mask),'T1':setdevice(T1_average),'B1':setdevice(B1_average),'B0':setdevice(B0_average),'T2':setdevice(T2_average),'PD':setdevice(PD_average)}
    else:
        return {'mask':setdevice(mask)}