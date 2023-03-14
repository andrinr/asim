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
    
def DFFT(x):
    N = len(x)
    
    memory = np.zeros(2, N)
    memory[0] = x

    clock = 0
    size = 2
    while size < N:
        read = memory[clock % 2]
        write = memory[(clock + 1) % 2]
        n_segments = int(N/size)

        for offset in range(0, size, 2):
            n = np.arange(segment_size)

            factor = np.exp(-2j*np.pi*n/ segment_size)
            even = read[offset::size]
            odd = read[offset+1::size]

            write[offset]
            
            X = np.concatenate(\
                [even+factor[:int(N/2)]*odd,
                even+factor[int(N/2):]*odd])
            return X
        
        size *= 2
        clock += 1

    
    
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

    elif optimization_level == 2:
        frequency = DFFT(signal)

    end = time.perf_counter()
    return frequency, end - start

