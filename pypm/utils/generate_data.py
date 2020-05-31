# -*- coding: utf-8 -*-

import numpy as np

def generate_oneclass_data(n_sample, magnitude):
    
    transform = np.array([[0.5768, 0.3982, 0.8291, 0.0, 0.0], 
                          [0.3766, 0.3566, 0.4009, 0.3578, 0.0],
                          [0.0, 0.0, 0.2435, 1.7678, 1.3936],
                          [0.0, 0.0, 0.0, 0.8519, 0.8045]])
    
    s = np.random.uniform(-1, 1, (n_sample, 4))
    e = np.random.normal(0, 0.01, (n_sample, 5))
    train_x = np.dot(s, transform) + e
    
    faulty_s = s[:, 0] + magnitude * np.ones((n_sample, 1))
    faulty_e = np.random.normal(0, 0.01, (n_sample, 5))
    test_x = np.dot(faulty_s, transform) + faulty_e
    
    return train_x, test_x
    