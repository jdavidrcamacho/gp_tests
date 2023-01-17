# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cholesky, cho_solve

def exponential_kernel(t, l=1.0):
    """
    t: this parameter is a 1D numpy array of shape (N,), where N is the number 
        of data points. It is used to create a grid of points using broadcasting.
    l (lambda): is the length scale parameter. 
        It determines how far apart two points need to be in input space for 
        their similarity to drop significantly. A smaller value of l will result 
        in a higher variance, meaning that the model will be more sensitive to 
        small changes in the input. A larger value of l will result in a lower 
        variance, meaning that the model will be less sensitive to small changes 
        in the input.
    """
    x = t[:, None]
    y = t[None, :]
    res = np.exp(-np.sum((x-y)**2, axis=-1)/(2*l**2))
    return res


def periodic_kernel(t, p=1.0):
    """
    t: this parameter is a 1D numpy array of shape (N,), where N is the number 
        of data points. It is used to create a grid of points using broadcasting.
    p (period): is the period of the kernel. 
        It determines how often the function to be modeled repeats. The smaller 
        the value of p the shorter the period and more frequently the function 
        oscillates.
    """
    x = t[:, None]
    y = t[None, :]
    res = np.exp(-2*np.sin(np.pi*np.abs(x-y)/p)**2)
    return res


def rational_quadratic_kernel(t, alpha=1.0, l=1.0):
    """
    t: this parameter is a 1D numpy array of shape (N,), where N is the number 
        of data points. It is used to create a grid of points using broadcasting.
    alpha: this parameter controls the scale of the kernel. A higher value of alpha 
        results in a kernel with a smaller scale, meaning that the function will 
        be less sensitive to small changes in the input. A lower value of alpha 
        will result in a kernel with a larger scale, meaning that the function 
        will be more sensitive to small changes in the input. The default value is 1.0
    l: this parameter is known as the length scale parameter. 
        It determines how far apart two points need to be in input space for 
        their similarity to drop significantly. A smaller value of l will result 
        in a higher variance, meaning that the model will be more sensitive to 
        small changes in the input. A larger value of l will result in a lower 
        variance, meaning that the model will be less sensitive to small changes 
        in the input. The default value is 1.0
    """
    x = t[:, None]
    y = t[None, :]
    res = (1 + (np.sum((x-y)**2, axis=-1)/(2*alpha*l**2)))**(-alpha)
    return res


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
