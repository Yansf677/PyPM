# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_wine

from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping

class FcAutoencoder:
    
    def __init__(self, x, hidden_dims):
        
        self.x = x
        
        self.hidden_dims = np.array(hidden_dims)
        
    def construct_model(self, encode_activation='sigmoid', decode_activation='sigmoid', use_linear=True):
        
        input_layer = Input(shape=(self.x.shape[1], ))
        
        # Single hidden layer
        if self.hidden_dims.shape[0] == 1:
            
            latent_layer = Dense(self.hidden_dims[0], activation = encode_activation)(input_layer)
            
            if use_linear == True:
                output_layer = Dense(self.x.shape[1], activation = 'linear')(latent_layer)
            else:
                output_layer = Dense(self.x.shape[1], activation = decode_activation)(latent_layer)
            
        # deep structure
        else:
            
            encode_layer = Dense(self.hidden_dims[0], activation = encode_activation)(input_layer)
            for i in range(self.hidden_dims.shape[0]//2 - 1):
                encode_layer = Dense(self.hidden_dims[i + 1], activation = encode_activation)(encode_layer)
            
            latent_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2], activation = encode_activation)(encode_layer)
            
            decode_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 1], activation = decode_activation)(latent_layer)
            for i in range(self.hidden_dims.shape[0]//2 - 1):
                decode_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 2 + i], activation = decode_activation)(decode_layer)
            
            if use_linear == True:
                output_layer = Dense(self.x.shape[1], activation = 'linear')(decode_layer)
            else:
                output_layer = Dense(self.x.shape[1], activation = decode_activation)(decode_layer)
            
        self.FcAutoencoder = Model(input=input_layer, output=output_layer)
        self.FcEncoder = Model(input=input_layer, output=latent_layer)
        
    def train_model(self, epochs=1000, batch_size=100, optimizer='Adam', loss='mean_squared_error', use_Earlystopping=True):
        
        self.FcAutoencoder.compile(optimizer=optimizer, loss=loss)
        
        if use_Earlystopping == True:
            self.history = self.FcAutoencoder.fit(self.x, self.x, epochs = epochs, batch_size = batch_size, shuffle = True, 
                                    validation_split = 0.10, callbacks = [EarlyStopping(monitor='val_loss', patience = 30)])
        else:
            self.history = self.FcAutoencoder.fit(self.x, self.x, epochs = epochs, batch_size = batch_size, shuffle = True)
        
    def get_features(self, x_test):
        
        return self.FcEncoder.predict(x_test)
        
    def get_reconstructions(self, x_test):
        
        return self.FcAutoencoder.predict(x_test)
        
    def save_model(self, FcAutoencoder_name=None, FcEncoder_name=None):
        
        if FcAutoencoder_name != None:
            self.FcAutoencoder.save(FcAutoencoder_name + '.h5')
        else:
            print("FcAutoencoder is not saved !")
        if FcEncoder_name != None:
            self.FcEncoder.save(FcEncoder_name + '.h5')
        else:
            print("FcEncoder is not saved !")
        
    def load_model(self, FcAutoencoder_name=None, FcEncoder_name=None):
        
        if FcAutoencoder_name != None:
            self.FcAutoencoder = load_model(FcAutoencoder_name + '.h5')
        else:
            print("FcAutoencoder is not load !")
        if FcEncoder_name != None:
            self.FcEncoder = load_model(FcEncoder_name + '.h5')
        else:
            print("FcEncoder is not load !")
        
if __name__ == '__main__':
    
    # load data and preprocess
    data = load_wine().data
    StandardScaler = preprocessing.StandardScaler().fit(data)
    train_data = StandardScaler.transform(data)
    
    # Build an autoencoder
    Autoencoder = FcAutoencoder(train_data, [10, 6, 10])
    Autoencoder.construct_model()
    
    # Train model
    Autoencoder.train_model()
    
    # Save model
    Autoencoder.save_model('Autoencoder', 'Encoder')
    
    # Get features & reconstructions
    Features = Autoencoder.get_features(train_data)
    Reconstructions = Autoencoder.get_reconstructions(train_data)
    
