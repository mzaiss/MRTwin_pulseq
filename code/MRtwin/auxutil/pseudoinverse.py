# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:42:55 2020

@author: fmglang
"""

import numpy as np

def pinv_reg(A, lambd=0.01):
        '''
        regularized pseudo-inverse of matrix A with reg parameter lambd
        (similar to Matlab implementation)
        '''
        
        m,n = A.shape
        if n > m:
            At = A
            A = A.conj().T
            n = m
            finalTranspose = True
        else:
            At = A.conj().T
            finalTranspose = False
        
        AtA = At.dot(A)
        S = np.linalg.eigvals(AtA)
        lambda_sq = lambd**2 * np.abs(np.max(S))
        
        X = np.linalg.solve(AtA + np.eye(n)*lambda_sq, At) # invert does not work, use solve instead
        
        if finalTranspose: X = X.conj().T
        
        return X