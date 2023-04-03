import numpy as np
import solver

class SOR(solver.Solver):
        
    def __init__(self, A : np.ndarray, b : np.ndarray, max_iterations : int = 256):
        super().__init__(A, b, max_iterations)
        
    def solve(self, relaxation_factor : float = 1.5):
        L = np.tril(self.A, -1)
        U = np.triu(self.A, 1)
        D = np.diag(np.diag(self.A))

        x = np.zeros(self.b.shape)

        A = np.linalg.inv(D- relaxation_factor * L)
        B = relaxation_factor * U + (1 - relaxation_factor) * D
        C = relaxation_factor * A @ self.b
       
        for i in range(self.max_iterations):
            x = A @ B @ x + C
            if self.check_convergence(x):
                break

        return x