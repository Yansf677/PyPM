# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from pypm.models.fc_autoencoder import FcAutoencoder as AE

from pypm.utils.threshold import threshold_kde
from pypm.utils.threshold import threshold_F
from pypm.utils.threshold import threshold_chi2

from pypm.utils.statistic import statistic_mahalanobis_distance as T2
from pypm.utils.statistic import statistic_euclid_distance as SPE

class DetectionbyAutoencoder:
    
    def __init__(self, x, hidden_dims):
        
        self.x = x
        self.hidden_dims = np.array(hidden_dims)
        self.feature_dim = self.hidden_dims[self.hidden_dims.shape[0]//2]
        self.AE = AE(x, hidden_dims)
        
    def offline_modeling(self,encode_activation='sigmoid', decode_activation='sigmoid', epochs=1000, batch_size=100):
        
        self.AE.construct_model(encode_activation, decode_activation)
        self.AE.train_model(epochs, batch_size)
        self.features = self.AE.get_features(self.x)
        self.reconstructions = self.AE.get_reconstructions(self.x)
        
    def get_offline_statistics(self):
        
        self.sigma = np.linalg.inv(np.dot(self.features.T, self.features) / (self.x.shape[0]-1))
        self.T2_offline = np.zeros(self.x.shape[0])
        self.SPE_offline = np.zeros(self.x.shape[0])
        
        for i in range(self.x.shape[0]):
            
            #self.T2_offline[i] = np.dot(np.dot(self.x[i].reshape(1,-1), self.sigma), self.x[i].reshape(-1,1))
            #self.SPE_offline[i] = np.dot(self.x[i].reshape(1,-1), self.x[i].reshape(-1,1))
            
            self.T2_offline[i] = T2(self.features[i], self.sigma)
            self.SPE_offline[i] = SPE(self.x[i], self.reconstructions[i])
            
    def cal_threshold(self, alpha, use_kde=True):
        
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
