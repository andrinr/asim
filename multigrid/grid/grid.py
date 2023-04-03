import numpy as np
import grid
import solver

class Multigrid:

    def __init__(self, 
            A : np.ndarray, 
            b : np.ndarray, 
            R : np.ndarray,
            P : np.ndarray,
            schema : str = "dduu",
            pre_smoothing_steps : int = 16):
        self.A = A
        self.A_inv = np.linalg.inv(A)
        self.b = b
        self.pre_smoothing_steps = pre_smoothing_steps
        self.R = R
        self.P = P

    def solve(self):
        # initial guess
        x = self.A_inv @ self.b

        # pre smoothing
        self.relax(x, self.pre_smoothing_steps)

        # coarse grid correction
        defect = self.defect(x)
        coarse_defect = self.restrict(defect)
        

        return x
    
    def relax(self, x : np.ndarray, steps : int) -> np.ndarray:
        # iterative improvement
        for i in range(steps):
            correction = solver.Seidel(self.A, self.defect(x)).solve()
            x = x + correction
    
    def defect(self, x : np.ndarray) -> np.ndarray:
        return self.A @ x - self.b
    
    def restrict(self, x : np.ndarray) -> np.ndarray:
        pass 

    def prolongate(self, x : np.ndarray) -> np.ndarray:
        pass

    def check_convergence(self, x : np.ndarray) -> bool:
        r = self.A @ x - self.b
        return np.linalg.norm(r) < self.epsilon
    
    def parse_schema(self, schema : str):
        self.schema = [char for char in schema]
        assert all([char in ["d", "u"] for char in self.schema])
        assert schema.count("d") == schema.count("u")
        assert self.schema[0] == "d"
        assert self.schema[-1] == "u"
        assert len(self.schema) % 2 == 1
