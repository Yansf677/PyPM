# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_wine

from keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping


class FcLabelAutoencoder:
    
    def __init__(self, x, labels, hidden_dims, use_onehot=False):
        
        self.x = x
        if use_onehot == True:
            self.labels = to_categorical(labels)
        else:    
            if labels.ndim == 1:
                self.labels = labels.reshape(-1,1)
            else:
                self.labels = labels
        self.x_and_label = np.hstack((self.x, self.labels))
        self.hidden_dims = np.array(hidden_dims)
        
    def construct_model(self, encode_activation='sigmoid', decode_activation='sigmoid', use_linear=True):
        
        input_layer_label = Input(shape=(self.x.shape[1] + self.labels.shape[1], ))
        input_layer = Input(shape=(self.x.shape[1], ))
        
        # AE
        if self.hidden_dims.shape[0] == 1:
            
            latent_layer_label = Dense(self.hidden_dims[0], activation = encode_activation)(input_layer_label)
            latent_layer = Dense(self.hidden_dims[0], activation = encode_activation)(input_layer)
            input_layer_latent = Input(shape=(self.hidden_dims[0], ))
            
            if use_linear == True:
                output_layer_label = Dense(self.x.shape[1], activation = 'linear')(latent_layer)
                output_layer = Dense(self.x.shape[1], activation = 'linear')(input_layer_latent)
            else:
                output_layer_label = Dense(self.x.shape[1] + self.labels.shape[1], activation = decode_activation)(latent_layer)
                output_layer = Dense(self.x.shape[1], activation = decode_activation)(input_layer_latent)
            
        # DAE
        else:
            
            encode_layer_label = Dense(self.hidden_dims[0], activation = encode_activation)(input_layer_label)
            encode_layer = Dense(self.hidden_dims[0], activation = encode_activation)(input_layer)
            for i in range(self.hidden_dims.shape[0]//2 - 1):
                encode_layer_label = Dense(self.hidden_dims[i + 1], activation = encode_activation)(encode_layer_label)
                encode_layer = Dense(self.hidden_dims[i + 1], activation = encode_activation)(encode_layer)
            
            latent_layer_label = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2], activation = encode_activation)(encode_layer_label)
            latent_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2], activation = encode_activation)(encode_layer)
            input_layer_latent = Input(shape=(self.hidden_dims[self.hidden_dims.shape[0]//2], ))
            
            decode_layer_label = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 1], activation = decode_activation)(latent_layer_label)
            decode_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 1], activation = decode_activation)(input_layer_latent)
            for i in range(self.hidden_dims.shape[0]//2 - 1):
                decode_layer_label = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 2 + i], activation = decode_activation)(decode_layer_label)
                decode_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 2 + i], activation = decode_activation)(decode_layer)
            
            if use_linear == True:
                output_layer_label = Dense(self.x.shape[1] + self.labels.shape[1], activation = 'linear')(decode_layer_label)
                output_layer = Dense(self.x.shape[1], activation = 'linear')(decode_layer)
            else:
                output_layer_label = Dense(self.x.shape[1] + self.labels.shape[1], activation = decode_activation)(decode_layer_label)
                output_layer = Dense(self.x.shape[1], activation = decode_activation)(decode_layer)
            
        self.FcLabelAutoencoder = Model(input=input_layer_label, output=output_layer_label)
        self.FcLabelEncoder = Model(input=input_layer_label, output=latent_layer_label)
        self.FcEncoder = Model(input=input_layer, output=latent_layer)
        self.FcDecoder = Model(input=input_layer_latent, output=output_layer)
        
    def train_model(self, epochs=1000, batch_size=100, optimizer='Adam', loss='mean_squared_error', use_Earlystopping=True):
        # Train label autoencoder
        self.FcLabelAutoencoder.compile(optimizer=optimizer, loss=loss)
        if use_Earlystopping == True:
            self.history_label = self.FcLabelAutoencoder.fit(self.x_and_label, self.x_and_label, epochs=epochs, batch_size=batch_size, shuffle=True, 
                                    validation_split=0.10, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        else:
            self.history_label = self.FcLabelAutoencoder.fit(self.x_and_label, self.x_and_label, epochs = epochs, batch_size = batch_size, shuffle = True)
        
        # Learn label features
        self.FcEncoder.compile(optimizer=optimizer, loss=loss)
        if use_Earlystopping == True:
            self.history_encoder = self.FcEncoder.fit(self.x, self.FcLabelEncoder.predict(self.x_and_label), epochs=epochs, batch_size=batch_size, shuffle=True,
                                validation_split=0.10, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        else:
            self.history_encoder = self.FcEncoder.fit(self.x, self.FcLabelEncoder.predict(self.x_and_label), epochs=epochs, batch_size=batch_size, shuffle=True)
        
        # reconstruct features
        self.FcDecoder.compile(optimizer=optimizer, loss=loss)
        if use_Earlystopping == True:
            self.history_decoder = self.FcDecoder.fit(self.FcEncoder.predict(self.x), self.x, epochs=epochs, batch_size=batch_size, shuffle=True,
                                validation_split=0.10, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        else:
            self.history_decoder = self.FcDecoder.fit(self.FcEncoder.predict(self.x), self.x, epochs=epochs, batch_size=batch_size, shuffle=True)
        
    def get_features(self, x_test):
        
        return self.FcEncoder.predict(x_test)
        
    def get_reconstructions(self, x_test):
        
        return self.FcDecoder.predict(self.FcEncoder.predict(x_test))
        
    def save_model(self, FcEncoder_name=None, FcDecoder_name=None):
        
        if FcEncoder_name != None:
            self.FcEncoder.save(FcEncoder_name + '.h5')
        else:
            print("FcEncoder is not saved !")
        if FcDecoder_name != None:
            self.FcDecoder.save(FcDecoder_name + '.h5')
        else:
            print("FcDecoder is not saved !")
        
    def load_model(self, FcEncoder_name=None, FcDecoder_name=None):
        
        if FcEncoder_name != None:
            self.FcEncoder = load_model(FcEncoder_name + '.h5')
        else:
            print("FcEncoder is not load !")
        if FcDecoder_name != None:
            self.FcDecoder = load_model(FcDecoder_name + '.h5')
        else:
            print("FcDecoder is not load !")
        
if __name__ == '__main__':
    
    # load data and preprocess
    data = load_wine().data
    labels = load_wine().target
    
    StandardScaler = preprocessing.StandardScaler().fit(data)
    train_data = StandardScaler.transform(data)
    
    # Build an labelautoencoder
    LabelAutoencoder = FcLabelAutoencoder(train_data, labels, [10, 8, 10], use_onehot=True)
    LabelAutoencoder.construct_model()
    
    # Train model
    LabelAutoencoder.train_model()
    
    # Save model
    LabelAutoencoder.save_model('LabelAutoencoder', 'LabelEncoder')
    
    # Get features & reconstructions
    Features = LabelAutoencoder.get_features(train_data)
    Reconstructions = LabelAutoencoder.get_reconstructions(train_data)
