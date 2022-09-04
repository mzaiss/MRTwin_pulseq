# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:38:09 2020

@author: zaissmz
"""
import numpy as np

#%% ############################################################################
# manual permutation
AA=np.array([[11,12,13],[21,22,23]])

permvec=[1,2,0]
inverse_perm = np.arange(len(permvec))[np.argsort(permvec)]

BB=AA[:,permvec]

CC=BB[:,inverse_perm]

#%% ############################################################################
# centric permutation
NRep=5

# centric permutation
permvec= np.zeros((NRep,),dtype=int) 
permvec[0] = 0
for i in range(1,int(NRep/2)+1):
    permvec[i*2-1] = (-i)
    if i < NRep/2:
        permvec[i*2] = i
permvec=permvec+NRep//2
# centric permutation end


AA=np.array([[11,12,13,14,15],[21,22,23,24,25]])
BB=AA[:,permvec]

inverse_perm = np.arange(len(permvec))[np.argsort(permvec)]

CC=BB[:,inverse_perm]

