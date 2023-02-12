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

#%% ############################################################################
# simpler centric permutation
import numpy as np
lst=np.linspace(1,9,10)

permutation = sorted(np.arange(len(lst)), key=lambda x: abs(len(lst)//2 - x))
sorted_lst = lst[permutation]

# Apply the same permutation to another array
other_arr = np.linspace(1,5,10)
sorted_other_arr = other_arr[permutation]

# Invert the permutation to restore the original order
inverted_permutation = np.argsort(permutation)
restored_lst = sorted_lst[inverted_permutation]
restored_other_arr = sorted_other_arr[inverted_permutation]