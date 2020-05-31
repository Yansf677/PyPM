import os

import numpy as np
import pandas as pd
from sklearn import preprocessing

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
            self.labels = labels
        self.x_and_label = np.hstack((x, labels))
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
            
        self.FC_autoencoder_label = Model(input=input_layer_label, output=output_layer_label)
        self.FC_encoder_label = Model(input=input_layer_label, output=latent_layer_label)
        self.FC_encoder = Model(input=input_layer, output=latent_layer)
        self.FC_decoder = Model(input=input_layer_latent, output=output_layer)
        
    def train_model(self, epochs=1000, batch_size=100, optimizer='Adam', loss='mean_squared_error', use_Earlystopping=True):
        # Train label autoencoder
        self.FC_autoencoder_label.compile(optimizer=optimizer, loss=loss)
        if use_Earlystopping == True:
            self.history_label = self.FC_autoencoder_label.fit(self.x_and_label, self.x_and_label, epochs=epochs, batch_size=batch_size, shuffle=True, 
                                    validation_split=0.10, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        else:
            self.history_label = self.FC_autoencoder_label.fit(self.x_and_label, self.x_and_label, epochs = epochs, batch_size = batch_size, shuffle = True)
        
        # Learn label features
        self.FC_encoder.compile(optimizer=optimizer, loss=loss)
        if use_Earlystopping == True:
            self.history_encoder = self.FC_encoder.fit(self.x, self.FC_encoder_label.predict(self.x_and_label), epochs=epochs, batch_size=batch_size, shuffle=True,
                                validation_split=0.10, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        else:
            self.history_encoder = self.FC_encoder.fit(self.x, self.FC_encoder_label.predict(self.x_and_label), epochs=epochs, batch_size=batch_size, shuffle=True)
        
        # reconstruct features
        self.FC_decoder.compile(optimizer=optimizer, loss=loss)
        if use_Earlystopping == True:
            self.history_decoder = self.FC_decoder.fit(self.FC_encoder.predict(self.x), self.x, epochs=epochs, batch_size=batch_size, shuffle=True,
                                validation_split=0.10, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        else:
            self.history_decoder = self.FC_decoder.fit(self.FC_encoder.predict(self.x), self.x, epochs=epochs, batch_size=batch_size, shuffle=True)
        
    def get_features(self, x_test):
        
        return self.FC_encoder.predict(x_test)
        
    def get_reconstructions(self, x_test):
        
        return self.FC_decoder.predict(self.FC_encoder.predict(x_test))
        
    def save_model(self, FC_encoder_name=None, FC_decoder_name=None):
        
        if FC_encoder_name != None:
            self.FC_encoder.save(FC_encoder_name + '.h5')
        if FC_decoder_name != None:
            self.FC_decoder.save(FC_decoder_name + '.h5')
        
    def load_model(self, FC_encoder_path=None, FC_decoder_path=None):
        
        if FC_encoder_path != None:
            self.FC_encoder = load_model(FC_encoder_path + '.h5')
        if FC_decoder_path != None:
            self.FC_decoder = load_model(FC_decoder_path + '.h5')
        
if __name__ == '__main__':
    
    # load data and preprocess
    data1 = pd.read_csv(os.path.dirname(os.getcwd()) + r'\\datasets\\d01_te.csv')
    StandardScaler1 = preprocessing.StandardScaler().fit(np.array(data1))
    train_data1 = StandardScaler1.transform(np.array(data1))
    
    data2 = pd.read_csv(os.path.dirname(os.getcwd()) + r'\\datasets\\d02_te.csv')
    StandardScaler2 = preprocessing.StandardScaler().fit(np.array(data2))
    train_data2 = StandardScaler2.transform(np.array(data2))
    
    train_data = np.vstack((train_data1, train_data2))
    labels = np.hstack((np.zeros(959), np.ones(959))).reshape(-1, 1)
    
    # Build an labelautoencoder
    LabelAutoencoder = FcLabelAutoencoder(train_data, labels, [33,21,10,21,33], use_onehot=False)
    LabelAutoencoder.construct_model(encode_activation='sigmoid', decode_activation='relu')
    
    # Train model
    LabelAutoencoder.train_model(epochs=200, batch_size=100)
    
    # Save model
    LabelAutoencoder.save_model('LabelAutoencoder', 'LabelEncoder')
    
    # Get features & reconstructions
    Features = LabelAutoencoder.get_features(train_data)
    Reconstructions = LabelAutoencoder.get_reconstructions(train_data)
