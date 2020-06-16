# -*- coding: utf-8 -*-

import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping

class Autoencoder:
    
    """
    Autoencoder (AE)
    
    Parameters
    ----------
    x (n_samples, n_features) - The training input samples
    hidden_dims (List) - The structure of autoencoder
    
    Attributes
    ----------
    Autoencoder (network) - The model of autoencoder
    Encoder (network) - The encoder part 
    
    Example
    -------
    >>> from sklearn import preprocessing
    >>> from sklearn.datasets import load_wine
    >>> from pypm.models.autoencoder import Autoencoder
    >>> # Load data
    >>> data = load_wine().data
    array([[1.423e+01, 1.710e+00, 2.430e+00, ..., 1.040e+00, 3.920e+00 ...
    >>> StandardScaler = preprocessing.StandardScaler().fit(data)
    >>> train_data = StandardScaler.transform(data)
    array([[ 1.51861254, -0.5622498 ,  0.23205254, ...,  0.36217728 ...
    >>> # Build an autoencoder
    >>> Autoencoder = Autoencoder(train_data, [10, 6, 10])
    >>> Autoencoder.construct_model()
    >>> # Train model
    >>> Autoencoder.train_model()
    >>> # Save model
    >>> Autoencoder.save_model('Autoencoder', 'Encoder')
    >>> # Get features & reconstructions
    >>> Features = Autoencoder.get_features(train_data)
    array([[0.7609496 , 0.37115023, 0.70390266, 0.37966228, 0.60897684 ...
    >>> Reconstructions = Autoencoder.get_reconstructions(train_data)
    array([[-0.05012968,  0.35567132, -0.4547131 , ...,  0.11404108 ...
    
    """
    
    def __init__(self, x, hidden_dims):
        
        self.x = x
        self.hidden_dims = np.array(hidden_dims)
        
    def construct_model(self, encode_activation='sigmoid', decode_activation='sigmoid', use_linear=True):
        
        """ 
        Function to initialize a autoencoder
        
        Parameters
        ----------
        encode_activation (str, default='sigmoid') - The activation in the encoding function
        decode_activation (str, default='sigmoid') - The activation in the decoding function
        use_linear (bool, default=True) - Whether use the linear transform in the output layer
        
        """
        
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
            
        self.Autoencoder = Model(input=input_layer, output=output_layer)
        self.Encoder = Model(input=input_layer, output=latent_layer)
        
    def train_model(self, epochs=1000, batch_size=100, optimizer='Adam', loss='mean_squared_error', use_Earlystopping=True):
        
        """ 
        Function to train the autoencoder
    
        Parameters
        ----------
        epochs (int, default=1000) - The number of iterations
        batch_size (int, default=100) - The number of samples in a batch
        optimizer (str, default='Adam') - The type of optimization when training
        loss (str, default='mean_squared_error') - The objective used when training
        use_Earlystopping (bool, default=True) - Whether use the early stopping when training
        
        """
        
        self.Autoencoder.compile(optimizer=optimizer, loss=loss)
        
        if use_Earlystopping == True:
            self.history = self.Autoencoder.fit(self.x, self.x, epochs = epochs, batch_size = batch_size, shuffle = True, 
                                    validation_split = 0.10, callbacks = [EarlyStopping(monitor='val_loss', patience = 30)])
        else:
            self.history = self.Autoencoder.fit(self.x, self.x, epochs = epochs, batch_size = batch_size, shuffle = True)
        
    def get_features(self, x_test):
        
        """ 
        Function to calculate features
        
        Parameters
        ----------
        x_test (_, n_features) - Test samples
        
        Return
        ------
        featurs (_, _)
        
        """
        
        return self.Encoder.predict(x_test)
        
    def get_reconstructions(self, x_test):
        
        """ 
        Function to calculate reconstructions
        
        Parameters
        ----------
        x_test (_, n_features) - Test samples
        
        Return
        ------
        reconstruction (_, _)
        
        """
        
        return self.Autoencoder.predict(x_test)
        
    def save_model(self, Autoencoder_name=None, Encoder_name=None):
        
        """ 
        Function to save the trained model
        
        Parameters
        ----------
        Autoencoder_name (str, default=None) - Name of autoencoder
        Encoder_name (str, default=None) - Name of encoder
        
        """
        
        if Autoencoder_name != None:
            self.Autoencoder.save(Autoencoder_name + '.h5')
        else:
            print("Autoencoder is not saved !")
        if Encoder_name != None:
            self.Encoder.save(Encoder_name + '.h5')
        else:
            print("Encoder is not saved !")
        
    def load_model(self, Autoencoder_name=None, Encoder_name=None):
        
        """ 
        Function to load the trained model
        
        Parameters
        ----------
        Autoencoder_name (str, default=None) - Name of autoencoder
        Encoder_name (str, default=None) - Name of encoder
        
        """
        
        if Autoencoder_name != None:
            self.Autoencoder = load_model(Autoencoder_name + '.h5')
        else:
            print("Autoencoder is not load !")
        if Encoder_name != None:
            self.Encoder = load_model(Encoder_name + '.h5')
        else:
            print("Encoder is not load !")
        