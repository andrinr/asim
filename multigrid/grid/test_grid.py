import numpy as np
from grid import Multigrid

def gen_grid():
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

def test_schema():
    grid = gen_grid()

    grid.parse_schema(schema="rrpp")

    parsed_schema = [
        'restrict', 
        'restrict', 
        'smooth', 
        'project', 
        'project', 
        'apply'
    ]

    assert grid.schema == parsed_schema
