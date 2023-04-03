import numpy as np
import solver

class Jacobi(solver.Solver):
        
    def __init__(self, A : np.ndarray, b : np.ndarray, max_iterations : int = 256):
        super().__init__(A, b, max_iterations)
        
    def solve(self):
        D = np.diag(self.A)
        D_inv = np.linalg.inv(np.diagflat(D))
        LU = self.A - np.diagflat(D)

        x = np.zeros(self.b.shape)

        for i in range(self.max_iterations):
            x = D_inv @ (self.b - LU @ x)
            if self.check_convergence(x):
                break

        return x