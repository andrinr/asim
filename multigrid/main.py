import numpy as np
from grid import sample_galaxy, Grid
import matplotlib.pyplot as plt

galaxy = sample_galaxy(rate=200)

stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
restriction_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
prolongation_kernel = restriction_kernel / 4

multigrid = Grid(
    stencil=stencil,
    b=galaxy,
    restriction_kernel=restriction_kernel,
    prolongation_kernel=prolongation_kernel,
    schema="rrrppp",
)

# Ax = b
# set x = b
# b - Ax = defect
# A * correction = defect

multigrid.solve()

plt.imshow(multigrid)
plt.show()