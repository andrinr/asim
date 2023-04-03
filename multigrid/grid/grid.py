import numpy as np
import grid
import solver

class Multigrid:

    def __init__(self, A : np.ndarray, b : np.ndarray, max_iterations : int = 16):
        self.A = A
        self.A_inv = np.linalg.inv(A)
        self.b = b
        self.max_iterations = max_iterations

    def solve(self):
        x = self.A_inv @ self.b
        residual = self.A @ x - self.b
        for i in range(self.max_iterations):
            correction = solver.Seidel(self.A, residual).solve()
            x = x + correction

        return x

