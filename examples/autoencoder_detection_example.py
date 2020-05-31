# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing

from pypm.detection.detection_by_autoencoder import DetectionbyAutoencoder as DE

if __name__ == '__main__':
    
    data = pd.read_csv(os.getcwd() + r'\\pypm\\datasets\\Tennessee.csv')
    data1 = pd.read_csv(os.getcwd() + r'\\pypm\\datasets\\d01_te.csv')
    StandardScaler = preprocessing.StandardScaler().fit(np.array(data))
    train_data = StandardScaler.transform(np.array(data))
    test_data = StandardScaler.transform(np.array(data1))
    
    DE = DE(train_data, [40,33,40])
    DE.offline_modeling()
    DE.get_offline_statistics()
    DE.cal_threshold(0.99, use_kde=True)
    DE.monitor_multi_sample(test_data, print_info=False)
    
