import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

from pypm.models.FC_Autoencoder import FC_Autoencoder as AE

from pypm.utils.threshold import threshold_kde
from pypm.utils.threshold import threshold_F
from pypm.utils.threshold import threshold_chi2

from pypm.utils.statistic import statistic_mahalanobis_distance as T2
from pypm.utils.statistic import statistic_euclid_distance as SPE

class Detection_by_Autoencoder:
    
    def __init__(self, x, hidden_dims):
        
        self.x = x
        
        self.hidden_dims = np.array(hidden_dims)
        
        self.AE = AE(x, hidden_dims)
        
    def offline_modeling(self):
        
        self.AE.construct_model()
        
        self.AE.train_model()
        
        self.features = self.AE.get_features(self.x)
        
        self.reconstructions = self.AE.get_reconstructions(self.x)
        
    def get_offline_statistics(self):
        
        self.sigma = np.linalg.inv(np.dot(self.scores.T, self.scores) / (self.x.sahpe[0]-1))
        self.T2_offline = np.zeros(self.x.sahpe[0])
        self.SPE_offline = np.zeros(self.x.sahpe[0])
        
        for i in range(self.x.shape[0]):
            
            #self.T2_offline[i] = np.dot(np.dot(self.x[i].reshape(1,-1), self.sigma), self.x[i].reshape(-1,1))
            #self.SPE_offline[i] = np.dot(self.x[i].reshape(1,-1), self.x[i].reshape(-1,1))
            
            self.T2_offline[i] = T2(self.x[i], self.sigma)
            self.SPE_offline[i] = SPE(self.x[i], self.reconstructions[i])
            
    def cal_threshold(self, use_kde=True):
        
        # calculate thresholds using kernel density estimation
        if use_kde == True:
            self.T2_th = threshold_kde(self.T2_offline)
            self.SPE_th = threshold_kde(self.SPE_offline)
            
        else:
            self.T2_th = threshold_F(self.T2_offline)
            self.SPE_th = threshold_chi2(self.SPE_offline)
        
        
    def monitor_a_sample(self, xnew):
        
        T2_new = np.dot(np.dot(xnew.reshape(1,-1), self.sigma), xnew.reshape(-1,1))
        SPE_new = np.dot(xnew.reshape(1,-1), xnew.reshape(-1,1))
        
        if T2_new > self.T2_th or SPE_new > self.SPE_th:
            print("THis is a faulty sample")
        else:
            print("This is a fault-free sample")
        
    def monitor_multi_sample(self, Xnew):
        
        n_sample = Xnew.shape[0]
        T2_new = np.zeros(n_sample)
        SPE_new = np.zeros(n_sample)
        detective_rate = 0
        for i in range(n_sample):
            T2_new[i] = T2(Xnew[i], self.sigma)
            SPE_new[i] = SPE(Xnew[i], self.AE.get_reconstructions(Xnew[i]))
            if T2_new[i] > self.T2_th or SPE_new[i] > self.SPE_th:
                print("The {}th sample is a faulty sample".format(i))
                detective_rate += 1/n_sample
            else:
                print("The {}th sample is a faulty sample".format(i))
        
        plt.figure()
        plt.subplot(211); plt.plot(T2_new);  plt.hlines(self.T2_th, 0, n_sample, colors='r', linestyles='dashed');  plt.legend(['statistics', 'control limit']); plt.title('Detection by Autoencoder'); plt.ylabel('T2');
        plt.subplot(212); plt.plot(SPE_new); plt.hlines(self.SPE_th, 0, n_sample, colors='r', linestyles='dashed'); plt.legend(['statistics', 'control limit']); plt.title('Detection by Autoencoder'); plt.ylabel('SPE');
        plt.show()
        
        print('The detective rate is {}'.format(detective_rate))

if __name__ == '__main__':
    
    data = pd.read_csv(os.path.dirname(os.getcwd()) + r'\\datasets\\Tennessee.csv')
    StandardScaler = preprocessing.StandardScaler().fit(np.array(data))
    train_data = StandardScaler.transform(np.array(data))
    
    AE1 = AE(train_data, [33, 21, 33])
    AE1.construct_model()
    AE1.train_model()
    features = AE1.get_features(train_data)
