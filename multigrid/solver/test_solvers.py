import numpy as np
import solver

def gen_Ab_random():
    
    n = 100
    # Create PD symmetric matrix
    A = np.random.rand(n, n)
    A = A + A.T * 0.5
    A = A + np.eye(n) * n
    x = np.random.rand(n)

    b = A @ x

    return A, b

def test_jacobi_random():

    A, b = gen_Ab_random()

    jacobi = solver.Jacobi(A, b, max_iterations=100)
    x = jacobi.solve()

    x_numpy = np.linalg.solve(A, b)

    assert np.allclose(x, x_numpy)
    
def test_gauss_seidel_random():

    A, b = gen_Ab_random()

    gauss_seidel = solver.Seidel(A, b, max_iterations=100)
    x = gauss_seidel.solve()

    x_numpy = np.linalg.solve(A, b)

    assert np.allclose(x, x_numpy)
