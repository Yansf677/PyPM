# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from pypm.models.autoencoder import Autoencoder as AE

from pypm.utils.threshold import threshold_kde
from pypm.utils.threshold import threshold_F
from pypm.utils.threshold import threshold_chi2

from pypm.utils.statistic import statistic_mahalanobis_distance as T2
from pypm.utils.statistic import statistic_euclid_distance as SPE

class DetectionbyAutoencoder:
    
    """
    Fault detection by Autoencoder 
    
    Parameters
    ----------
    x (n_samples, n_features) - The training input samples
    hidden_dims (List) - The structure of autoencoder
    
    Attributes
    ----------
    AE (network) - The model of autoencoder
    features (_, _) - Features calculted by AE
    reconstructions (_, _) - Reconstructions calculated by AE
    sigma (float) - Covariance of features
    T2_offline (float) - Statistics T2 of the offline data
    SPE_offline (float) - Statistics SPE of the offline data
    T2_th (float) - Threshold of T2
    SPE_th (float) - Threshold of SPE
    
    Example
    -------
    >>> from sklearn import preprocessing
    >>> from sklearn.datasets import load_wine
    >>> from pypm.detection.detection_by_autoencoder import DetectionbyAutoencoder
    >>> # Load data
    >>> data = load_wine().data
    array([[1.423e+01, 1.710e+00, 2.430e+00, ..., 1.040e+00, 3.920e+00 ...
    >>> StandardScaler = preprocessing.StandardScaler().fit(data)
    >>> train_data = StandardScaler.transform(data)
    array([[ 1.51861254, -0.5622498 ,  0.23205254, ...,  0.36217728 ...
    >>> # Build an autoencoder
    >>> AE = DetectionbyAutoencoder(train_data, [10, 6, 10])
    >>> AE.offline_modeling()
    >>> # offline statistics
    >>> AE.get_offline_statistics()
    >>> AE.cal_threshold(0.99)
    >>> # Monitoring result
    >>> AE.monitor_multi_sample(data)
    
    """
    def __init__(self, x, hidden_dims):
        
        self.x = x
        self.hidden_dims = np.array(hidden_dims)
        self.feature_dim = self.hidden_dims[self.hidden_dims.shape[0]//2]
        self.AE = AE(x, hidden_dims)
        
    def offline_modeling(self,encode_activation='sigmoid', decode_activation='sigmoid', epochs=1000, batch_size=100):
        
        """ 
        Function to train AE
        
        Parameters
        ----------
        encode_activation (str, default='sigmoid') - The activation in the encoding function
        decode_activation (str, default='sigmoid') - The activation in the decoding function
        epochs (int, default=1000) - The number of iterations
        batch_size (int, default=100) - The number of samples in a batch
        
        """
        
        self.AE.construct_model(encode_activation, decode_activation)
        self.AE.train_model(epochs, batch_size)
        self.features = self.AE.get_features(self.x)
        self.reconstructions = self.AE.get_reconstructions(self.x)
        
    def get_offline_statistics(self):
        
        """ 
        Function to calculate offline statistics
        
        """
        
        self.sigma = np.linalg.inv(np.dot(self.features.T, self.features) / (self.x.shape[0]-1))
        self.T2_offline = np.zeros(self.x.shape[0])
        self.SPE_offline = np.zeros(self.x.shape[0])
        
        for i in range(self.x.shape[0]):
            
            #self.T2_offline[i] = np.dot(np.dot(self.x[i].reshape(1,-1), self.sigma), self.x[i].reshape(-1,1))
            #self.SPE_offline[i] = np.dot(self.x[i].reshape(1,-1), self.x[i].reshape(-1,1))
            
            self.T2_offline[i] = T2(self.features[i], self.sigma)
            self.SPE_offline[i] = SPE(self.x[i], self.reconstructions[i])
            
    def cal_threshold(self, alpha, use_kde=True):
        
        """ 
        Function to calculate thresholds
        
        Parameters
        ----------
        alpha (float) - The significant level
        use_kde (bool, default=True) - The way to estimate the thresholds of statistics
        
        """
        
        # calculate thresholds using kernel density estimation
        if use_kde == True:
            print("Estimate thresholds of T2")
            self.T2_th = threshold_kde(self.T2_offline, alpha)
            print("Estimate thresholds of SPE")
            self.SPE_th = threshold_kde(self.SPE_offline, alpha)
            
        else:
            self.T2_th = threshold_F(self.T2_offline, self.feature_dim, alpha)
            self.SPE_th = threshold_chi2(self.SPE_offline, alpha)
        
    def monitor_a_sample(self, xnew, print_info=True):
        
        """ 
        Function to monitor a single sample
        
        Parameters
        ----------
        xnew (1, n_features) - The test sample
        print_info (bool, default=True) - Choose to print the relevant information
        
        """
        
        features_new = self.AE.get_features(xnew.reshape(1,-1))
        reconstructions = self.AE.get_reconstructions(xnew.reshape(1,-1))
        T2_new = T2(features_new, self.sigma)
        SPE_new = SPE(xnew, reconstructions)
        
        if T2_new > self.T2_th or SPE_new > self.SPE_th:
            if print_info == True:
                print("THis is a faulty sample")
        else:
            if print_info == True:
                print("This is a fault-free sample")
        
    def monitor_multi_sample(self, Xnew, print_info=True):
        
        """ 
        Function to monitor multiple samples
        
        Parameters
        ----------
        xnew (n_samples, n_features) - The test samples
        print_info (bool, default=True) - Choose to print the relevant information
        
        """
        
        Features_new = self.AE.get_features(Xnew)
        Reconstructions = self.AE.get_reconstructions(Xnew)
        
        n_sample = Xnew.shape[0]
        T2_new = np.zeros(n_sample)
        SPE_new = np.zeros(n_sample)
        detective_rate = 0
        for i in range(n_sample):
            T2_new[i] = T2(Features_new[i], self.sigma)
            SPE_new[i] = SPE(Xnew[i], Reconstructions[i])
            if T2_new[i] > self.T2_th or SPE_new[i] > self.SPE_th:
                if print_info == True:
                    print("The {}th sample is a faulty sample".format(i+1))
                detective_rate += 1/n_sample
            else:
                if print_info == True:
                    print("The {}th sample is a faulty sample".format(i+1))
        
        plt.figure()
        plt.subplot(211); plt.plot(T2_new);  plt.hlines(self.T2_th, 0, n_sample, colors='r', linestyles='dashed');  plt.legend(['statistics', 'control limit']); plt.title('Detection by Autoencoder'); plt.ylabel('T2');
        plt.subplot(212); plt.plot(SPE_new); plt.hlines(self.SPE_th, 0, n_sample, colors='r', linestyles='dashed'); plt.legend(['statistics', 'control limit']); plt.title('Detection by Autoencoder'); plt.ylabel('SPE');
        plt.show()
        
        print('The detective rate is {}'.format(detective_rate))
