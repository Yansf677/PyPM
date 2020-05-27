# -*- coding: utf-8 -*-

def detective_rate(statistic, threshold):
    """
    function to calculate detective rate 
    
    parameters
    ----------
    statistic:
        statistics of testing data
    
    threshold:
        threshold by the offline data
    
    return
    ------
    fault detective rate or false alarm
    
    """
    
    n_sample = statistic.shape[0]
    
    detective_rate = 0
    for i in range(n_sample):
        if statistic[i] > threshold:
            detective_rate += 1/n_sample
    
    return detective_rate

def accuracy():
    
    """
    f1_score...
    """
        
    