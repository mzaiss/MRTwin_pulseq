"""
Created on Wed Oct 21 11:25:24 2020

@author: weinmusn
"""
import numpy as np
import torch
from scipy.optimize import curve_fit
import auxutil.helper_functions as hf


#%%########## Calculation of S_0 #############################################
##############################################################################  
def signal_S0B1T2 (xdata, S0, T2, rB1):
    (signal, exflip, te) = xdata
    return signal * torch.sin(exflip*rB1) * torch.exp(-te/T2) * S0

def signal_S0B1T2T2dash (xdata, S0, T2, T2dash, rB1):
    (signal, exflip, te) = xdata
    return signal * torch.sin(exflip*rB1) * torch.exp(-te/T2) * torch.exp(-te/T2dash) * S0 

def signal_B1T1_ADC_torch (xdata, T1, rB1):
    (signal, TI, angle, exflip, TR, delay, n, i) = xdata
    return torch.squeeze((((signal * (torch.cos(exflip*rB1) * torch.exp(-TR/T1))**n
                           + (1-(torch.cos(exflip*rB1) * torch.exp(-TR/T1))**n) / (1-torch.cos(exflip*rB1)*torch.exp(-TR/T1))
                           * (1 - torch.exp(-TR/T1)) - 1) * torch.exp(-delay/T1) + 1) * torch.cos(angle*rB1)
                           - 1) * torch.exp(-TI/T1) + 1)

def signal_B1T1_ADC (xdata, T1, rB1):
    (signal, TI, angle, exflip, TR, delay, n, i) = xdata
    return np.squeeze((((signal * (np.cos(exflip*rB1) * np.exp(-TR/T1))**n
                           + (1-(np.cos(exflip*rB1) * np.exp(-TR/T1))**n) / (1-np.cos(exflip*rB1)*np.exp(-TR/T1))
                           * (1 - np.exp(-TR/T1)) - 1) * np.exp(-delay/T1) + 1) * np.cos(angle*rB1)
                           - 1) * np.exp(-TI/T1) + 1)

def signal_ADC (xdata):
    (M_sat, exflip, TR, n, T1, rB1) = xdata
    # if exflip.shape[0] == 1:
    #     factor = torch.exp(-TR/T1)*torch.cos(exflip*rB1)
    #     addition = (1-torch.exp(-TR/T1))
    #     M_ss = torch.ones([1,M_sat.shape[1],n.shape[-1]])
    #     tmp = torch.ones([1,M_sat.shape[1]])
    #     for ii in np.arange(1,n.shape[-1]):
    #         tmp = tmp*factor[:,:,ii]+addition[:,:,ii]
    #         M_ss[:,:,ii] = tmp # TR and exflip rolled 1 position
    #     return M_ss/M_sat
    # else:   
    M_ss = (1-torch.exp(-TR/T1)) / (1-torch.cos(exflip*rB1)*torch.exp(-TR/T1))
    return ((torch.cos(exflip*rB1) * torch.exp(-TR/T1))**n
        + (1-(torch.cos(exflip*rB1) * torch.exp(-TR/T1))**n) * M_ss/M_sat)

def old_signal_B1T1_torch (data, T1, rB1):
    (TI, angle) = data 
    return torch.squeeze(1 + (torch.cos(angle*rB1)- 1)*torch.exp(-TI/T1))

def old_signal_B1T1 (data, T1, rB1):
    (TI, angle) = data 
    return np.squeeze(1 + (np.cos(angle*rB1) - 1)*np.exp(-TI/T1))

def piecewise_signal_B1T1(data, T1, rB1):
    (TI, angle, exflip, TR, delay, n) = data
    if torch.is_tensor(data[0]) == True:
        signal = hf.setdevice(torch.zeros([len(data[0]),*T1.squeeze().shape]))
    else:
        signal = np.zeros(len(data[0]))
    for i in np.arange(len(data[0])):
        if i == 0:
            xdata = [0, TI[i], angle[i], exflip[i], TR[i], delay[i], n[i], i]
            if torch.is_tensor(data[0]) == True:
                signal[i] = old_signal_B1T1_torch(xdata[1:3], T1, rB1)
            else:
                signal[i] = old_signal_B1T1(xdata[1:3], T1, rB1)
        else:
            if torch.is_tensor(data[0]) == True:
                xdata = [signal[i-1].clone(), TI[i], angle[i], exflip[i-1,0], TR[i-1], delay[i-1], n[i-1], i]
                signal[i] = signal_B1T1_ADC_torch(xdata, T1, rB1)
            else:
                xdata = [signal[i-1], TI[i], angle[i], exflip[i-1], TR[i-1,0], delay[i-1], n[i-1], i]
                signal[i] = signal_B1T1_ADC(xdata, T1, rB1)
    if torch.is_tensor(data[0]) == True:
        signal = signal 
        return (signal)
    else:   
        signal = signal
        return (signal)
    
def piecewise_signal_B1T1T2(data, S0, T1, T2, rB1):
    (TI, angle, exflip, TR, delay, n, te) = data
    if torch.is_tensor(data[0]) == True:
        signal = hf.setdevice(torch.zeros([len(data[0]),*T1.squeeze().shape]))
    else:
        signal = np.zeros(len(data[0]))
    for i in np.arange(len(data[0])):
        if i == 0:
            xdata = [0, TI[i], angle[i], exflip[i], TR[i], delay[i], n[i], i]
            if torch.is_tensor(data[0]) == True:
                signal[i] = old_signal_B1T1_torch(xdata[1:3], T1, rB1)
            else:
                signal[i] = old_signal_B1T1(xdata[1:3], T1, rB1)
        else:
            if torch.is_tensor(data[0]) == True:
                xdata = [signal[i-1].clone(), TI[i], angle[i], exflip[i-1], TR[i-1], delay[i-1], n[i-1], i]
                signal[i] = signal_B1T1_ADC_torch(xdata, T1, rB1)
            else:
                xdata = [signal[i-1], TI[i], angle[i], exflip[i-1], TR[i-1], delay[i-1], n[i-1], i]
                signal[i] = signal_B1T1_ADC(xdata, T1, rB1)
    if torch.is_tensor(data[0]) == True:
        signal = signal * torch.sin(exflip*rB1) * torch.exp(-te/T2) * S0
        return (signal)
    else:   
        signal = signal * np.sin(exflip*rB1) * S0
        return (signal)
    
def piecewise_signal_B1T1T2dash(data, S0, T1, rB1, T2, T2dash):
    (TI, angle, exflip, TR, delay, n, te) = data
    if torch.is_tensor(data[0]) == True:
        signal = hf.setdevice(torch.zeros([len(data[0]),*T1.squeeze().shape]))
    else:
        signal = np.zeros(len(data[0]))
    for i in np.arange(len(data[0])):
        if i == 0:
            xdata = [0, TI[i], angle[i], exflip[i], TR[i], delay[i], n[i], i]
            if torch.is_tensor(data[0]) == True:
                signal[i] = old_signal_B1T1_torch(xdata[1:3], T1, rB1)
            else:
                signal[i] = old_signal_B1T1(xdata[1:3], T1, rB1)
        else:
            if torch.is_tensor(data[0]) == True:
                xdata = [signal[i-1].clone(), TI[i], angle[i], exflip[i-1], TR[i-1], delay[i-1], n[i-1], i]
                signal[i] = signal_B1T1_ADC_torch(xdata, T1, rB1)
            else:
                xdata = [signal[i-1], TI[i], angle[i], exflip[i-1], TR[i-1], delay[i-1], n[i-1], i]
                signal[i] = signal_B1T1_ADC(xdata, T1, rB1)
    if torch.is_tensor(data[0]) == True:
        signal = signal * torch.sin(exflip*rB1) * torch.exp(-te/T2dash) * torch.exp(-te/T2) * S0
        return (signal)
    else:   
        signal = signal * np.sin(exflip*rB1) * np.exp(-te/T2dash) * torch.exp(-te/T2) * S0
        return (signal)

def piecewise_signal_B1T1_S0(data, T1, rB1):
    (TI, angle, exflip, TR, delay, n, S0) = data
    if torch.is_tensor(data[0]) == True:
        signal = torch.zeros(len(data[0]))
    else:
        signal = np.zeros(len(data[0]))
    for i in np.arange(len(data[0])):
        if i == 0:
            xdata = [0, TI[i], angle[i],exflip[i], TR[i], delay[i], n[i], i]
            if torch.is_tensor(data[0]) == True:
                signal[i] = old_signal_B1T1_torch(xdata[1:3], T1, rB1)
            else:
                signal[i] = old_signal_B1T1(xdata[1:3], T1, rB1)
        else:
            if torch.is_tensor(data[0]) == True:
                xdata = [signal[i-1].clone(), TI[i], angle[i], exflip[i-1], TR[i-1], delay[i-1], n[i-1], i]
                signal[i] = signal_B1T1_ADC_torch(xdata, T1, rB1)
            else:
                xdata = [signal[i-1], TI[i], angle[i], exflip[i-1], TR[i-1], delay[i-1], n[i-1], i]
                signal[i] = signal_B1T1_ADC(xdata, T1, rB1)
    if torch.is_tensor(data[0]) == True:
        signal =  signal * torch.sin(exflip*rB1) * S0
        return (signal)
    else:   
        signal = signal * np.sin(exflip*rB1) * S0
        return (signal)
    
def piecewise_signal_B1T1T2dash_S0(data, T1, rB1, T2dash):
    (TI, angle, exflip, TR, delay, n, te, S0) = data
    if torch.is_tensor(data[0]) == True:
        signal = torch.zeros(len(data[0]))
    else:
        signal = np.zeros(len(data[0]))
    for i in np.arange(len(data[0])):
        if i == 0:
            xdata = [0, TI[i], angle[i], exflip[i], TR[i], delay[i], n[i], i]
            if torch.is_tensor(data[0]) == True:
                signal[i] = old_signal_B1T1_torch(xdata[1:3], T1, rB1)
            else:
                signal[i] = old_signal_B1T1(xdata[1:3], T1, rB1)
        else:
            
            if torch.is_tensor(data[0]) == True:
                xdata = [signal[i-1].clone(), TI[i], angle[i], exflip[i-1], TR[i-1], delay[i-1], n[i-1], i]
                signal[i] = signal_B1T1_ADC_torch(xdata, T1, rB1)
            else:
                xdata = [signal[i-1], TI[i], angle[i], exflip[i-1], TR[i-1], delay[i-1], n[i-1], i]
                signal[i] = signal_B1T1_ADC(xdata, T1, rB1)
    if torch.is_tensor(data[0]) == True:
        signal = signal * torch.sin(exflip*rB1) * torch.exp(-te/T2dash) * S0
        return (signal)
    else:   
        signal = signal * np.sin(exflip*rB1) * np.exp(-te/T2dash) * S0
        return (signal)

def quantify_B1T1_ADC_S0 (TI, angle, exflip, TR, delay, n, S, tT1, tB1, S0, p0):
    data = (TI, angle, exflip, TR, delay, n, S0)
    popt, pcov = curve_fit(piecewise_signal_B1T1_S0, data, S/S0, p0=p0, maxfev=10000)#, bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,2]))
    return popt, pcov
    
def quantify_B1T1_ADC (TI, angle, exflip, TR, delay, n, S, p0):
    data = (TI, angle, exflip, TR, delay, n)
    popt, pcov = curve_fit(piecewise_signal_B1T1, data, S, p0=p0, maxfev=100000)#, bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,2]))
    return popt, pcov

def quantify_B1T1T2dash_ADC (TI, angle, exflip, TR, delay, n, te, S, p0):
    data = (TI, angle, exflip, TR, delay, n, te)
    popt, pcov = curve_fit(piecewise_signal_B1T1T2dash, data, S, p0=p0, maxfev=100000)#, bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,2]))
    return popt, pcov

def quantify_B1T1T2dash_ADC_S0 (TI, angle, exflip, TR, delay, n, te, S, S0, p0):
    data = (TI, angle, exflip, TR, delay, n, te, S0)
    popt, pcov = curve_fit(piecewise_signal_B1T1T2dash_S0, data, S/S0, p0=p0, maxfev=10000)#, bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,2]))
    return popt, pcov

def signal_B0_numpy (te, rB0, dPhi):
    return np.ravel(rB0*te+dPhi)

def quantify_B0 (te, S, p0):
    popt, pcov = curve_fit(signal_B0_numpy, te, S, p0 = p0, maxfev=1000000)#, bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,2]))
    return popt, pcov