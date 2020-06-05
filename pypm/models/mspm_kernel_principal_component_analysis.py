# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA  
from sklearn import preprocessing
from sklearn import kernel_approximation as ks 

class MspmKernelPrincipalComponentAnalysis:
    """
    This module is to construct a kernel principal component analysis (KPCA) model for feature analysis.
    
    Parameters
    ----------
    
    x (n_samples, n_features) – The training input samples
    n_components – The number of principal components
    nkernel_components (int) – Dimensions of training data after kernel mapping
    kernel (optional, default=’rbf’) – the function of kernel mapping – select the type (‘linear’ | ‘poly’ | ‘rbf’ | ‘sigmoid’ | ‘cosine’) of kernel mapping
    preprocess (default = True) - the preprocessing of the data
    gamma, coef0, degree (float, default: gamma =None, coef0 = 1,degree = 3) – The parameters of kernel mapping

    Attributes
    ----------
    kpca - model of KPCA
    
    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> from pypm.models.mspm_kernel_principal_component_analysis import MspmKernelPrincipalComponentAnalysis
    >>> data = load_iris()
    >>> x = data.data
    array([[5.1, 3.5, 1.4, 0.2]...
    >>> y = data.target
    array([0, 0, 0, 0, 0, 0, 0...
    >>> KPCA_model = MspmKernelPrincipalComponentAnalysis(x, n_components = 3, nkernel_components = 10, kernel = 'rbf', preprocess = True)
    >>> xKernel = KPCA_model.convert_to_kernel(x)
    array([[-1.00741186, -1.03778048,	-3.2924193e-01, ...,	2.064873775...
    >>> KPCA_model.construct_kpca_model()
    >>> Features = KPCA_model.extact_kpca_feature(x)
    array([[ 0.379209, -0.107399, 2.20307]...
    """
    def __init__ (self, x, n_components, nkernel_components = 100, kernel = 'rbf', preprocess = True, gamma =None, coef0 = 1,degree = 3):
        
        self.x = x 
        self.kernel = kernel
        self.n_components = n_components
        self.preprocess = preprocess
        self.gamma = gamma
        self.coef0 = coef0
        self.degree  = degree
        self.nkernel_components = nkernel_components
        
        self.kX = ks.Nystroem(kernel=self.kernel, gamma = self.gamma, coef0 = self.coef0, degree = self.degree, n_components = self.nkernel_components)
        self.Xkernel = self.kX.fit_transform(x)
        if self.preprocess:
            self.Xscaler  = preprocessing.StandardScaler().fit(self.Xkernel)
            self.Xkernel = self.Xscaler.transform(self.Xkernel)
        
    def convert_to_kernel(self,x_test):
        
        """
        Function to kernel map the data
        
        Parameters
        ----------
        x_test (_, n_features) - The testing samples
        
        """
        xkernel = self.kX.fit_transform(x_test)
        if self.preprocess:
            xkernel = self.Xscaler.transform(xkernel)

        return xkernel
     
    def construct_kpca_model(self):
        """
        Function to construct a KPCA model.
        
        """
        self.kpca = PCA(self.n_components)
        self.kpca.fit(self.Xkernel)
        
    def extract_kpca_feature(self, x_test):
        """
        Function to extract the KPCA feature of given data using the trained-well KPCA model.
        
        Parameters
        ----------
        x_test (_, n_features) - The testing samples
        
        """
        xkernel = self.convert_to_kernel(x_test)
        return self.kpca.transform(xkernel)

