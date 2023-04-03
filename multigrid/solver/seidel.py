import numpy as np
import solver

class Seidel(solver.Solver):
        
    def __init__(self, A : np.ndarray, b : np.ndarray, max_iterations : int = 16):
        super().__init__(A, b, max_iterations)
        
    def solve(self):
        L = np.tril(self.A, 0)
        L_inv = np.linalg.inv(L)
        U = np.triu(self.A, 1)
        x = np.zeros(self.b.shape)

        T = -L_inv @ U
        C = L_inv @ self.b

        for i in range(self.max_iterations):
            x = T @ x + C

        return x