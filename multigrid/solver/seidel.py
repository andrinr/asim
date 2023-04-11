import numpy as np
import solver
from scipy.linalg import solve_triangular

class Seidel(solver.Solver):
        
    def __init__(self, A : np.ndarray, b : np.ndarray, max_iterations : int = 256):
        super().__init__(A, b, max_iterations)
        
    def solve(self):
        L = np.tril(self.A, 0)
        L_inv = solve_triangular(L, np.identity(L.shape[0]), lower=True)
        U = np.triu(self.A, 1)
        x = np.zeros(self.b.shape)

        T = -L_inv @ U
        C = L_inv @ self.b

        for i in range(self.max_iterations):
            x = T @ x + C
            if self.check_convergence(x):
                break

        return x