import numpy as np
from grid import Multigrid

def gen_grid():
    b = np.random.rand(32, 32)
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

def test_convergence():
    grid = gen_grid()

    x = grid.solve()
    error = grid.error(x)

    x = grid.solve(x)
    error2 = grid.error(x)

    print(error, error2)
    assert (error2 < error)
    pass