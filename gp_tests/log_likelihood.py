# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cholesky, cho_solve


def log_marginal_likelihood(X, y, kernel, kernel_params):
    """
    This function takes in four arguments:
    
    X: a 2D numpy array of shape (n,d) where n is the number of data points 
        and d is the number of features or dimensions.
    y: a 1D numpy array of shape (n,) containing the target variable values.
    kernel: the kernel function that you want to use. In this example I assumed 
        that you are passing a function which takes in X,X' and the kernel parameters 
        and returns the covariance matrix
    kernel_params: a dictionary containing the parameters of the kernel function.
    """
    K = kernel(X, **kernel_params)
    try:
        L = cholesky(K, lower=True)
    except:
        nugget = 0.01
        K = nugget*np.identity(X.size) + K
        L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), y)
    lml = -0.5*y.T@alpha - np.sum(np.log(np.diag(L))) -0.5*X.shape[0]*np.log(2*np.pi)
    return lml
