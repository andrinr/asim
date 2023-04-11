import numpy as np
import solver
from scipy.ndimage import convolve

class Multigrid:

    def __init__(self, 
            stencil : np.ndarray, 
            b : np.ndarray, 
            recurse_kernel : np.ndarray,
            project_kernel : np.ndarray,
            schema : str = "rrpp",
            pre_smoothing_steps : int = 16):
        self.L = stencil
        self.b = b
        if self.b.ndim != 1:
            # row major flattening
            self.b = self.b.flatten()

        self.pre_smoothing_steps = pre_smoothing_steps
        self.recurse_kernel = recurse_kernel
        self.project_kernel = project_kernel
        self.parse_schema(schema=schema)

    def solve(self):
        # We want to find x such that Ax = b
        # We can rewrite this as x = A_inv * b but this is not efficient
        # Instead we can use multigrid to solve this problem

        # initial guess
        x = self.b

        # pre smoothing
        x = self.relax(x, self.pre_smoothing_steps)

        # coarse grid correction
        for map in self.schema:
            if map == "r":
                x = self.restrict(x)
            elif map == "p":
                x = self.prolongate(x)
            else:
                raise Exception("Invalid map")
        
        return x
    
    def relax(self, x : np.ndarray, steps : int) -> np.ndarray:
        # iterative improvement
        for i in range(steps):
            correction = solver.Seidel(A, -self.defect(x)).solve()
            x = x + correction

        return x
    
    def defect(self, A : np.ndarray, b : np.ndarray, x : np.ndarray) -> np.ndarray:
        return A @ x - b
    
    def restrict(self, x : np.ndarray) -> np.ndarray:
        pass 

    def prolongate(self, x : np.ndarray) -> np.ndarray:
        pass
    
    def parse_schema(self, schema : str):
        self.schema = [char for char in schema]
        assert all([char in ["r", "p"] for char in self.schema])
        assert schema.count("r") == schema.count("p")
        assert self.schema[0] == "r"
        assert self.schema[-1] == "p"
