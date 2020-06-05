# -*- coding: utf-8 -*-

from sklearn.cross_decomposition import PLSRegression
from sklearn import kernel_approximation as ks
from sklearn import preprocessing

class MspmKernelPartialLeastSquares(PLSRegression):
    """
    This module is to construct a kernel_partial_least_squares (KPLS) model for feature analysis.
    
    Parameters
    ----------
    
    x (n_samples, n_features) – The training input samples
    y (n_samples, n_targets) – The training target samples
    max_iter (int, default=500) – the maximum number of iterations of the NIPALS inner loop (used only if algorithm=”nipals”)
    n_components (int) – The number of feature scores
    nkernel_components (int) – Dimensions of training data after kernel mapping
    scale (bool, default=Ture) – whether to scale the data
    tol (non-negative real) – Tolerance used in the iterative algorithm default 1e-06
    kernel (optional, default=’rbf) – the function of kernel mapping – select the type (‘linear’ | ‘poly’ | ‘rbf’ | ‘sigmoid’ | ‘cosine’) of kernel mapping
    preprocess (default = True) - the preprocessing of the data
    gamma, coef0, degree (float, default: gamma =None, coef0 = 1,degree = 3) – The parameters of kernel mapping

    Attributes
    ----------
    pls - model of PLS
    
    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> from pypm.models.mspm_kernel_partial_least_squares import MspmKernelPartialLeastSquares
    >>> data = load_iris()
    >>> x = data.data
    array([[5.1, 3.5, 1.4, 0.2]...
    >>> y = data.target
    array([0, 0, 0, 0, 0, 0, 0...
    >>> KPLS_model = MspmKernelPartialLeastSquares(x, y, n_components = 3, nkernel_components = 10, kernel = 'linear', preprocess = True)
    >>> xKernel = KPLS_model.convert_to_kernel(x)
    array([[ 1.34028706,  1.18280966, -1.21577202, ..., -1.27535676...
    >>> KPLS_model.constrcut_kpls_model()
    >>> Features = KPLS_model.extract_kpls_feature(x)
    array([[-3.97355862e+00,  2.56776542e-01,  4.96648268e-02]...
    >>> Prediction = KPLS_model.kpls_predict(x)
    array([[-0.08336091]...
    
    """
    
    def __init__(self, x, y, copy=True, max_iter=500, n_components=1, nkernel_components = 100, scale=True, tol=1e-06, kernel = 'linear',preprocess = True, gamma =None, coef0 = 1,degree = 3):
        super(MspmKernelPartialLeastSquares, self).__init__(copy=copy, max_iter=max_iter, n_components=n_components, scale=scale, tol=tol)
        
        self.x = x
        self.y = y
        self.kernel = kernel
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
            
        
    def convert_to_kernel(self, x_test):
        """
        Function to kernel map the data
        
        Parameters
        ----------
        x_test (n_samples, n_features) - The testing samples
        
        """
        Xkernel = self.kX.transform(x_test)
        if self.preprocess:
            xkernel = self.Xscaler.transform(Xkernel)

        return xkernel
    
    def constrcut_kpls_model(self):
        """
        Function to construct KPLS model
        
        """
        
        self.kpls = PLSRegression(self.n_components)
        self.kpls.fit(self.Xkernel, self.y)
        
    def extract_kpls_feature(self, x_test):
        """
        Function to extract the KPLS feature of given data using the trained-well KPLS model.
        
        Parameters
        ----------
        x_test (_, n_features) - The testing samples
        
        """
        xkernel = self.convert_to_kernel(x_test)
        #return self.kpls.fit_transform(x, y)[0]
        return self.kpls.transform(xkernel)
    
    def kpls_predict(self, x_test):
        """
        Function to extract the KPLS feature of given data using the trained-well KPLS model.
        
        Parameters
        ----------
        x_test (_, n_features) - The testing samples
        
        """
        xkernel = self.convert_to_kernel(x_test)
        return self.kpls.predict(xkernel)