# -*- coding: utf-8 -*-
import numpy as np

def statistic_mahalanobis_distance(x, X):
    """
    function to calculate mahalanobis distance for a new sample
    
    parameters
    ----------
    x:
        score of a new sample
    
    X:
        scores of offline data
        
    return
    ------
    mahalanobis distance
    
    """
    
    inverse = np.linalg.inv(np.dot(X.T, X) / X.shape[0])
    
    return np.dot(x.reshape(1,-1), np.dot(inverse, x.reshape(-1,1)))

def statistic_euclid_distance(x, x_re):
    """
    function to calculate euclid distance for a new sample
    
    parameters
    ----------
    x:
        a new sample
    x_re:
        reconstructions of a new sample
    
    return
    ------
    euclid distance
    """
    
    residual = x_re - x
    
    return np.dot(residual.reshape(1,-1), residual.reshape(-1,1))[0,0]