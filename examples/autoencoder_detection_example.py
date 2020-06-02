# -*- coding: utf-8 -*-

from sklearn.datasets import load_wine
from sklearn import preprocessing

from pypm.detection.detection_by_autoencoder import DetectionbyAutoencoder as D_AE

if __name__ == '__main__':
    
    x1 = load_wine().data[0:59, :]
    x2 = load_wine().data[59:130, :]
    x3 = load_wine().data[130:, :]
    
    StandardScaler = preprocessing.StandardScaler().fit(x1)
    x1 = StandardScaler.transform(x1)
    x2 = StandardScaler.transform(x2)
    x3 = StandardScaler.transform(x3)
    
    D_AE = D_AE(x1, [10, 8, 10])
    D_AE.offline_modeling()
    D_AE.get_offline_statistics()
    D_AE.cal_threshold(0.99, use_kde=True)
    D_AE.monitor_multi_sample(x2, print_info=False)
    D_AE.monitor_multi_sample(x3, print_info=False)
