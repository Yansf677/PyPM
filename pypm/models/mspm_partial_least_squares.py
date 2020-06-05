# -*- coding: utf-8 -*-

from sklearn.cross_decomposition import PLSRegression 
from sklearn import preprocessing 

class MspmPartialLeastSquares:
    """
    This module is to construct a partial_least_squares (PLS) model for feature analysis.
    
    Parameters
    ----------
    
    x (n_samples, n_features) – The training input samples
    y (n_samples, n_targets) – The training target samples
    n_components – The number of feature scores
    preprocess (default = True) - the preprocessing of the data

    Attributes
    ----------
    pls - model of PLS
    
    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> from pypm.models.mspm_partial_least_squares import MspmPartialLeastSquares
    >>> data = load_iris()
    >>> x = data.data
    array([[5.1, 3.5, 1.4, 0.2]...
    >>> y = data.target
    array([0, 0, 0, 0, 0, 0, 0...
    >>> PLS_model = MspmPartialLeastSquares(x, y, 3)
    >>> PLS_model.construct_pls_model()
    >>> Features = PLS_model.extract_pls_feature(x)
    array([[-2.26393268e+00,  1.74075256e-01,  3.62141834e-01]...
    >>> Prediction = PLS_model.pls_predict(x)
    array([[-8.05094197e-02]...
    
    """
    def __init__(self, x, y, n_components, preprocess = True):
        
        self.x = x
        self.y = y
        self.preprocess = preprocess
        self.n_components = n_components
        
        if self.preprocess:
            self.Xscaler  = preprocessing.StandardScaler().fit(self.x)
            self.x = self.Xscaler.transform(self.x)
        
    def construct_pls_model(self):
        """
        Function to construct a pls model.
        
        """
        self.pls = PLSRegression(self.n_components)
        self.pls.fit(self.x, self.y)
        
    def extract_pls_feature(self, x_test):
        """
        Function to extract the PCA feature of given data using the trained-well PCA model.
        
        Parameters
        ----------
        x_test (_, n_features) - The testing samples
        
        """
        if self.preprocess:
            x_test = self.Xscaler.transform(x_test)
        return self.pls.transform(x_test)
    
    def pls_predict(self, x_test):
        
        if self.preprocess:
            x_test = self.Xscaler.transform(x_test)
        return self. pls.predict(x_test)
