import numpy as np
from grid import Multigrid
import solver 

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
    
    return grid, stencil, b

def test_schema():
    grid, _, _ = gen_grid()

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
    grid, stencil, b = gen_grid()

    x_prime = solver.StencilJacobi(
        stencil=stencil,
        b=b,
        max_iterations=200
    ).solve()

    print(grid.error(x_prime))

    x = grid.solve()
    error = grid.error(x)

    x = grid.solve(x)
    error2 = grid.error(x)
    
    assert (error2 < error)

    for i in range(200):
        x = grid.solve(x)
        error = grid.error(x)

    print(error)

    assert (error < 0.01)
