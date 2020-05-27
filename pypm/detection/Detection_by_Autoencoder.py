import os

import numpy as np
import pandas as pd
from sklearn import preprocessing

from models.FC_Autoencoder import FC_Autoencoder as AE

data = pd.read_csv(os.path.dirname(os.getcwd()) + r'\\datasets\\Tennessee.csv')
StandardScaler = preprocessing.StandardScaler().fit(np.array(data))
train_data = StandardScaler.transform(np.array(data))

AE1 = AE(train_data, [33, 21, 33])
AE1.construct_model()

