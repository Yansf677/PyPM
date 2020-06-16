# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping

class SparseAutoencoder:
    
    """
    Sparse autoencoder (SAE)
    
    Parameters
    ----------
    x (n_samples, n_features) - The training input samples
    hidden_dims (List) - The structure of autoencoder
    
    Attributes
    ---------
    SparseAutoencoder (network) - The model of sparse autoencoder
    SparseEncoder (network) - The encoder part 
    
    Example
    -------
    >>> from sklearn import preprocessing
    >>> from sklearn.datasets import load_wine
    >>> from pypm.models.sparse_autoencoder import SparseAutoencoder
    >>> # Load data
    >>> data = load_wine().data
    array([[1.423e+01, 1.710e+00, 2.430e+00, ..., 1.040e+00, 3.920e+00 ...
    >>> StandardScaler = preprocessing.StandardScaler().fit(data)
    >>> train_data = StandardScaler.transform(data)
    array([[ 1.51861254, -0.5622498 ,  0.23205254, ...,  0.36217728 ...
    >>> # Build a SparseAutoencoder
    >>> SparseAutoencoder = SparseAutoencoder(train_data, [20, 10, 20])
    >>> SparseAutoencoder.construct_model()
    >>> # Train model
    >>> SparseAutoencoder.train_model()
    >>> # Save model
    >>> SparseAutoencoder.save_model('SparseAutoencoder', 'SparseEncoder') 
    >>> # Get features & reconstructions
    >>> Features = SparseAutoencoder.get_features(train_data)
    array([[0.26172   , 0.44321376, 0.318091  , ..., 0.13301125, 0.31837162 ...
    >>> Reconstructions = SparseAutoencoder.get_reconstructions(train_data)
    array([[ 0.00241652, -0.12601265, -0.04094902, ...,  0.1492601 ...
    
    """
    
    def __init__(self, x, hidden_dims):
        
        self.x = x
        self.hidden_dims = np.array(hidden_dims)
        
    def construct_model(self, p=0.01, beta=1, encode_activation='sigmoid', decode_activation='sigmoid', use_linear=True):
        
        """ 
        Function to initialize a sparse autoencoder
        
        Parameters
        ----------
        encode_activation (str, default='sigmoid') - The activation in the encoding function
        decode_activation (str, default='sigmoid') - The activation in the decoding function
        use_linear (bool, default=True) - Whether use the linear transform in the output layer
        
        """
        
        def sparse_constraint(activ_matrix):

            p_hat = K.mean(activ_matrix) # average over the batch samples
            #KLD = p*(K.log(p)-K.log(p_hat)) + (1-p)*(K.log(1-p)-K.log(1-p_hat))
            KLD = p*(K.log(p/p_hat)) + (1-p)*(K.log(1-p/1-p_hat))
            
            return -beta * K.sum(KLD) # sum over the layer units
    
        input_layer = Input(shape=(self.x.shape[1], ))
        
        # AE
        if self.hidden_dims.shape[0] == 1:
            
            latent_layer = Dense(self.hidden_dims[0], activation = encode_activation, activity_regularizer=sparse_constraint)(input_layer)
            
            if use_linear == True:
                output_layer = Dense(self.x.shape[1], activation = 'linear')(latent_layer)
            else:
                output_layer = Dense(self.x.shape[1], activation = decode_activation)(latent_layer)
            
        # DAE
        else:
            
            encode_layer = Dense(self.hidden_dims[0], activation = encode_activation, activity_regularizer=sparse_constraint)(input_layer)
            for i in range(self.hidden_dims.shape[0]//2 - 1):
                encode_layer = Dense(self.hidden_dims[i + 1], activation = encode_activation, activity_regularizer=sparse_constraint)(encode_layer)
            
            latent_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2], activation = encode_activation, activity_regularizer=sparse_constraint)(encode_layer)
            
            decode_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 1], activation = decode_activation, activity_regularizer=sparse_constraint)(latent_layer)
            for i in range(self.hidden_dims.shape[0]//2 - 1):
                decode_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 2 + i], activation = decode_activation, activity_regularizer=sparse_constraint)(decode_layer)
            
            if use_linear == True:
                output_layer = Dense(self.x.shape[1], activation = 'linear')(decode_layer)
            else:
                output_layer = Dense(self.x.shape[1], activation = decode_activation)(decode_layer)
           
        self.SparseAutoencoder = Model(input=input_layer, output=output_layer)
        self.SparseEncoder = Model(input=input_layer, output=latent_layer)
        
    def train_model(self, epochs=1000, batch_size=100, optimizer='Adam', loss='mean_squared_error', use_Earlystopping=True):
        
        """ 
        Function to train the sparse autoencoder
        
        Parameters
        ----------
        epochs (int, default=1000) - The number of iterations
        batch_size (int, default=100) - The number of samples in a batch
        optimizer (str, default='Adam') - The type of optimization when training
        loss (str, default='mean_squared_error') - The objective used when training
        use_Earlystopping (bool, default=True) - Whether use the early stopping when training
        
        """
        
        self.SparseAutoencoder.compile(optimizer=optimizer, loss=loss)
        
        if use_Earlystopping == True:
            self.history = self.SparseAutoencoder.fit(self.x, self.x, epochs = epochs, batch_size = batch_size, shuffle = True, 
                                    validation_split = 0.10, callbacks = [EarlyStopping(monitor='val_loss', patience = 10)])
        else:
            self.history = self.SparseAutoencoder.fit(self.x, self.x, epochs = epochs, batch_size = batch_size, shuffle = True)
        
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
        
        return self.SparseEncoder.predict(x_test)
        
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
        
        return self.SparseAutoencoder.predict(x_test)
        
    def save_model(self, SparseAutoencoder_name=None, SparseEncoder_name=None):
        
        """ 
        Function to save the trained model
        
        Parameters
        ----------
        Autoencoder_name (str, default=None) - Name of autoencoder
        Encoder_name (str, default=None) - Name of encoder
        
        """
        
        if SparseAutoencoder_name != None:
            self.SparseAutoencoder.save(SparseAutoencoder_name + '.h5')
        else:
            print("SparseAutoencoder is not saved !")
        if SparseEncoder_name != None:
            self.SparseEncoder.save(SparseEncoder_name + '.h5')
        else:
            print("SparseEncoder is not saved !")
        
    def load_model(self, SparseAutoencoder_name=None, SparseEncoder_name=None):
        
        """ 
        Function to load the trained model
        
        Parameters
        ----------
        Autoencoder_name (str, default=None) - Name of autoencoder
        Encoder_name (str, default=None) - Name of encoder
        
        """
        
        if SparseAutoencoder_name != None:
            self.SparseAutoencoder = load_model(SparseAutoencoder_name + '.h5')
        else:
            print("SparseAutoencoder is not load !")
        if SparseEncoder_name != None:
            self.SparseEncoder = load_model(SparseEncoder_name + '.h5')
        else:
            print("SparseEncoder is not load !")
    