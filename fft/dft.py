import numpy as np
import time


def NDFT(x):
    """
    Naive DFT
    """

    start = time.perf_counter()

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = np.dot(e, x)

    end = time.perf_counter()
    
    return X, end - start

def RFFT(x):
    """
    Fast Fourier Transform, recursive
    """
    N = len(x)
    
    if N == 1:
        return x
    else:
        n = np.arange(N)
        X_even = RFFT(x[::2])
        X_odd = RFFT(x[1::2])
        factor = np.exp(-2j*np.pi*n/ N)
        
        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X
    
def DFT(signal, optimization_level=0):
    """
    Discrete Fourier Transform
    """
    start = time.perf_counter()
    frequency = np.zeros(len(signal))

    if optimization_level == 0:
        frequency = NDFT(signal)

    elif optimization_level == 1:
        frequency = RFFT(signal)

    end = time.perf_counter()
    return frequency, end - start

