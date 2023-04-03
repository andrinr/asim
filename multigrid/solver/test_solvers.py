import numpy as np
import solver

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
    
def test_gauss_seidel_random():

    A, b, x = gen_Ab_random()

    gauss_seidel = solver.Seidel(A, b, max_iterations=100)
    x_approx = gauss_seidel.solve()

    assert np.allclose(x_approx, x)

# def test_sor_random():
    
#     A, b = gen_Ab_random()

#     sor = solver.SOR(A, b, max_iterations=1000)
#     x = sor.solve()

#     x_numpy = np.linalg.solve(A, b)

#     assert np.allclose(x, x_numpy)