# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

def threshold_kde(x, alpha):
    """
    calculate thresholds by kernel density estimation
    
    Gaussian kernel function is used
    
    parameters
    ----------
    x:
        statistics of offline data (n_samples, n_features)
    
    alpha:
        preset the false alarm rate
    
    
    return
    ------
    thresholds
    
    """
    
    kernel = stats.gaussian_kde(x)
    
    step = np.linspace(-100, 100, 10000)
    pdf = kernel(step)
    
    for i in range(len(step)):

        if sum(pdf[0:(i+1)]) / sum(pdf) > alpha:
            
            break
    
    return step[i]

def threshold_T2(x, freedom, alpha):
    """
    calculate thresholds for mahalanobis distance
    
    parameters
    ----------
    x:
        statistics of offline data (n_samples, n_features)
    
    freedom:
        preset freedom for F distribution 
    
    alpha:
        preset the false alarm rate
    
    
    return
    ------
    thresholds
    
    """
    
    n = x.shape[0]
    
    F = stats.f.isf(1-alpha, freedom, n-freedom)
    
    return freedom * (n-1) * (n+1) * F / (n * (n-freedom))
    
def threshold_SPE(x, alpha):
    """
    calculate thresholds for mahalanobis distance
    
    parameters
    ----------
    x:
        statistics of offline data (n_samples, n_features)
    
    alpha:
        preset the false alarm rate
    
    
    return
    ------
    thresholds
    
    """
    
    miu = np.mean(x)
    S = np.var(x)
    
    g = S / (2 * miu)
    h = 2 * miu * miu / S
    
    return g * stats.chi2.isf(1-alpha, h)
     
    
