import numpy as np
from scipy.ndimage import convolve
from grid import build_matrix_from_kernel

def test_matrix_from_kernel():
    laplace_2d = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    dim = 20

    A = build_matrix_from_kernel(laplace_2d, dim)

    print(A)

    x = np.random.rand(dim, dim)

    assert np.allclose(convolve(x, laplace_2d), A @ x)

