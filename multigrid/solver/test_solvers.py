import numpy as np
import solver
from scipy.ndimage import convolve

def gen_Ab_random():
    
    n = 200
    m = 100
    # Create PD symmetric matrix
    A = np.random.rand(n, n)
    A = A + A.T * 0.5
    A = A + np.eye(n) * n
    x = np.random.rand(n, m)

    b = A @ x

    return A, b, x

def test_jacobi_random():

    A, b, x = gen_Ab_random()

    jacobi = solver.Jacobi(A, b, max_iterations=100)
    x_approx = jacobi.solve()

    assert np.allclose(x_approx, x)

def test_jacobi_stencil():

    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    b = np.random.rand(10, 10)

    jacobi = solver.StencilJacobi(stencil, b, max_iterations=500)
    x_approx = jacobi.solve()

    b_approx = convolve(x_approx, stencil, mode="constant")
    assert np.allclose(b_approx, b)
    
def test_gauss_seidel_random():

    A, b, x = gen_Ab_random()

    gauss_seidel = solver.Seidel(A, b, max_iterations=1000)
    x_approx = gauss_seidel.solve()

    assert np.allclose(x_approx, x)