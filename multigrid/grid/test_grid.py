import numpy as np
from grid import Multigrid

def test_interpolation():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    restrict_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
    project_kernel = restrict_kernel / 4

    grid = Multigrid(
        stencil=stencil, 
        b=x, 
        restrict_kernel=restrict_kernel, 
        project_kernel=project_kernel, 
        schema="rrrppp")
    
    x_prime = grid.interpolate(x)
    x_prime = np.round(x_prime, 2)
    print(x)
    print(x_prime)

    assert False

