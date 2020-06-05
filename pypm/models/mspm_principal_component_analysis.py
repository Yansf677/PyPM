# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA 
from sklearn import preprocessing

class MspmPrincipalComponentAnalysis:
    """
    This module is to construct a principal component analysis (PCA) model for feature analysis.
    
    Parameters
    ----------
    
    x (n_samples, n_features) – The training input samples
    n_components – The number of principal components
    preprocess (default = True) - the preprocessing of the data

    Attributes
    ----------
    pca - model of PCA
    
    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> from pypm.models.mspm_principal_component_analysis import MspmPrincipalComponentAnalysis
    >>> data = load_iris()
    >>> x = data.data
    array([[5.1, 3.5, 1.4, 0.2]...
    >>> y = data.target
    array([0, 0, 0, 0, 0, 0, 0...
    >>> PCA_model = MspmPrincipalComponentAnalysis(x, 2)
    >>> PCA_model.construct_pca_model()
    >>> Features = PCA_model.extact_pca_feature(x)
    array([[-2.68412563,  0.31939725]...
    
    """
    def __init__(self, x, n_components, preprocess = True):
        
        self.x = x
        self.preprocess = preprocess
        self.n_components = n_components
        
        if self.preprocess:
            self.Xscaler  = preprocessing.StandardScaler().fit(self.x)
            self.x = self.Xscaler.transform(self.x)
        
    def construct_pca_model(self):
        """
        Function to construct a pca model.
        
        """
        self.pca = PCA(self.n_components)
        self.pca.fit(self.x)
        
    def extact_pca_feature(self, x_test):
        """
        Function to extract the PCA feature of given data using the trained-well PCA model.
        
        Parameters
        ----------
        x_test (_, n_features) - The testing samples
        
        """
        if self.preprocess:
            x_test = self.Xscaler.transform(x_test)
        return self.pca.transform(x_test)
