import numpy as np

def build_matrix_from_kernel(kernel : np.ndarray, dim : int) -> np.ndarray:
    A = np.zeros((dim, dim))

    row = np.zeros(dim)
    row[0] = kernel[0, 1]
    row[1] = kernel[0, 2]
    row[dim - 2] = kernel[0, 0]

    for i in range(dim):
        A[i] = np.roll(row, i)

    return A
