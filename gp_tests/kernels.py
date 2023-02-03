# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import cdist

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


def matern_32_kernel(t, l=1.0):
    """
    x and y are the two input vectors for which the kernel value is to be computed.
    l (lambda) is the length scale parameter. It determines how far apart two points need to be in input space for 
        their similarity to drop significantly.
    """
    x = t[:, None]
    y = t[None, :]
    r = np.sqrt(3*np.sum((x-y)**2))/l
    res = (1+r)*np.exp(-r)
    return res


def matern_52_kernel(t, l=1.0):
    """
    x and y are the two input vectors for which the kernel value is to be computed.
    l (lambda) is the length scale parameter. It determines how far apart two points need to be in input space for 
        their similarity to drop significantly.
    """
    x = t[:, None]
    y = t[None, :]
    r = np.sqrt(5*np.sum((x-y)**2))/l
    res = (1+r+r**2/3)*np.exp(-r)
    return res


def exponential_kernel(t, l=1.0):
    """
    x and y are the two input vectors for which the kernel value is to be computed.
    l (lambda) is the length scale parameter. 
        It determines how far apart two points need to be in input space for 
        their similarity to drop significantly. A smaller value of l will result 
        in a higher variance, meaning that the model will be more sensitive to 
        small changes in the input. A larger value of l will result in a lower 
        variance, meaning that the model will be less sensitive to small changes 
        in the input.
    """
    x = t[:, None]
    y = t[None, :]
    res = np.exp(-np.sqrt(np.sum((x-y)**2))/(2*l))
    return res


def spectral_mixture_kernel(t, num_components=1, amplitude_scale=1.0, frequency_scale=1.0, length_scale=1.0):
    """
    t is the input array for which the kernel value is to be computed.
    num_components is the number of complex sinusoids used in the kernel.
    amplitude_scale, frequency_scale, and length_scale are the kernel parameters.
    """
    x, y = np.meshgrid(t, t)
    K = np.zeros((len(t), len(t)))
    for i in range(num_components):
        amplitude = amplitude_scale * np.random.rand()
        frequency = frequency_scale * np.random.rand()
        length = length_scale * np.random.rand()
        K += amplitude * np.exp(-2*np.sin(np.pi*np.sqrt(np.sum((x-y)**2, axis=-1))*frequency)**2/(length**2))
    return K


def polynomial_kernel(t, d=2, c=1):
    """
    t is the input vector for which the kernel value is to be computed.
    d is the degree of the polynomial.
    c is a constant.
    """
    x, y = np.meshgrid(t, t)
    return (np.dot(x, y) + c) ** d


def sigmoid_kernel(t, a=1, b=0):
    """
    t is the input vector for which the kernel value is to be computed.
    a is a constant.
    b is a constant.
    """
    x, y = np.meshgrid(t, t)
    return np.tanh(a * np.dot(x, y) + b)


def laplacian_kernel(t, l=1):
    """
    t is the input vector for which the kernel value is to be computed.
    l is the length scale parameter.
    """
    x, y = np.meshgrid(t, t)
    return np.exp(-np.linalg.norm(x-y, ord=2) / l)


def cauchy_kernel(t, l=1):
    """
    t is the input vector for which the kernel value is to be computed.
    l is the length scale parameter.
    """
    x, y = np.meshgrid(t, t)
    return 1 / (1 + np.linalg.norm(x-y, ord=2) ** 2 / l ** 2)


def anova_kernel(t):
    """
    t is the input vector for which the kernel value is to be computed.
    """
    x, y = np.meshgrid(t, t)
    return np.sum((x-y)**2)


def linear_kernel(t):
    """
    t is the input vector for which the kernel value is to be computed.
    """
    x, y = np.meshgrid(t, t)
    return np.dot(x, y)


def exp_sine_squared_kernel(t, l=1, P=1):
    """
    t is the input vector for which the kernel value is to be computed.
    l is the length scale parameter.
    P is the period of the signal.
    """
    x, y = np.meshgrid(t, t)
    return np.exp(-2*np.sin(np.pi*np.abs(x-y)/P)**2/l**2)


def matern_kernel(t, l=1, nu=3/2):
    """
    t is the input vector for which the kernel value is to be computed.
    l is the length scale parameter.
    nu is the smoothness parameter.
    """
    x, y = np.meshgrid(t, t)
    dist = np.linalg.norm(x-y, ord=2)
    if nu == 1/2:
        return np.exp(-dist/l)
    elif nu == 3/2:
        return (1 + np.sqrt(3)*dist/l) * np.exp(-np.sqrt(3)*dist/l)
    elif nu == 5/2:
        return (1 + np.sqrt(5)*dist/l + 5*dist**2/3/l**2) * np.exp(-np.sqrt(5)*dist/l)


def neural_network_kernel(t, weights, activations):
    """
    t is the input vector for which the kernel value is to be computed.
    weights is a list of the weights of the neural network.
    activations is a list of the activation functions of the neural network.
        Common activation functions used in neural networks include the sigmoid 
        function,  the rectified linear unit (ReLU), and the hyperbolic tangent 
        (tanh) function
    """
    x, y = np.meshgrid(t, t)
    hidden_units = len(weights)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    for i in range(hidden_units):
        x = activations[i](np.dot(x, weights[i]))
        y = activations[i](np.dot(y, weights[i]))
    return np.dot(x, y.T)


def piecewise_kernel(t, C=1.0, k1=None, k1_params={}):
    """
    t: array-like of shape (n,d) the input data
    C: scalar the kernel parameter
    k1: function the stationary kernel function 
    k1_params: dictionary the kernel parameters 
    """
    if k1 is None:
        k1 = lambda x, y: np.exp(-np.sum((x-y)**2, axis=-1))
    k1_value = k1(t, t, **k1_params)
    k2_value = C*np.sum(np.eye(t.shape[0]), axis=-1)
    return k1_value + k2_value


def piecewise_kernel(t, stationary_kernel, C=1.0, **kwargs):
    """
    t: array-like of shape (n,d) the input data
    stationary_kernel: callable, the stationary kernel function
    C: scalar, the kernel parameter
    """
    k1 = stationary_kernel(t, **kwargs)
    d = t.shape[1]
    k2 = C * (cdist(t, t) == 0).sum(axis=1)
    return k1 + k2


def paciorek_kernel(t, stationary_kernel, p=1.0, **kwargs):
    """
    t: array-like of shape (n,d) the input data
    stationary_kernel: callable, the stationary kernel function
    p: scalar, the non-stationary kernel parameter
    """
    k1 = stationary_kernel(t, **kwargs)
    dist = cdist(t, t)
    k2 = (1 + dist**2)**(-p)
    return k1 * k2
