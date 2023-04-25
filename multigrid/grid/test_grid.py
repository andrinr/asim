import numpy as np
from grid import Multigrid

def gen_grid():
    b = np.random.rand(128, 128)
    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    restrict_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
    project_kernel = restrict_kernel / 4

    grid = Multigrid(
        stencil=stencil, 
        b=b, 
        restrict_kernel=restrict_kernel, 
        prolong_kernel=project_kernel, 
        schema="rrpp")
    
    return grid

def test_schema():
    grid = gen_grid()

    parsed_schema = [
        'restrict',
        'restrict',
        'smooth',
        'prolongate',
        'prolongate',
        'apply'
    ]

    assert (grid.schema == parsed_schema)
