import numpy as np
import solver
from scipy.ndimage import convolve
from scipy.interpolate import RegularGridInterpolator
from grid import grid_points

class Multigrid:

    def __init__(self, 
            stencil : np.ndarray, 
            phi : np.ndarray, 
            restrict_kernel : np.ndarray,
            project_kernel : np.ndarray,
            schema : str = "rrpp",
            pre_smoothing_steps : int = 16):
        self.stencil = stencil
        self.phi = phi
        self.pre_smoothing_steps = pre_smoothing_steps
        self.restrict_kernel = restrict_kernel
        self.project_kernel = project_kernel
        self.max_depth = self.parse_schema(schema=schema)

    def solve(self):
        # We want to find x such that Ax = b
        # We can rewrite this as x = A_inv * b but this is not efficient
        # Instead we can use multigrid to solve this problem

        # initial guess
        x = self.phi

        # pre smoothing
        x = self.relax(x, self.pre_smoothing_steps)

        d = self.defect(self.stencil, self.phi, x)

        # coarse grid correction
        for map in self.schema:
            if map == "r":
                d = self.restrict(d)
                eps = None
                self.relax
            elif map == "p":
                x = self.interpolate(x)
        
        return x
    
    def recurse(self, x : np.ndarray, depth : int) -> np.ndarray:
        if depth == self.max_depth:
            return self.relax(x, self.pre_smoothing_steps)
        else:
            x = self.relax(x, self.pre_smoothing_steps)
            d = self.defect(self.stencil, self.phi, x)
            d = self.restrict(d)
            x = self.recurse(x, depth + 1)
            x = self.interpolate(x)
            x = self.relax(x, self.pre_smoothing_steps)
            return x
    
    def relax(self, x : np.ndarray, steps : int) -> np.ndarray:
        correction = solver.StencilJacobi(
            self.stencil, -self.defect(x), max_iterations=steps).solve()
        x = x + correction

        return x
    
    def defect(self, A : np.ndarray, b : np.ndarray, x : np.ndarray) -> np.ndarray:
        return A @ x - b
    
    def restrict(self, x : np.ndarray) -> np.ndarray:
        convolve(x, self.restrict_kernel, mode="constant")[::2, ::2]


    def interpolate(self, x : np.ndarray) -> np.ndarray:
        x_res = np.zeros((2 * x.shape[0], 2 * x.shape[1]))
        x_sparse = np.zeros((2 * x.shape[0], 2 * x.shape[1]))

        x_sparse[::2, ::2] = x
        x_res = convolve(x_sparse, self.project_kernel, mode="constant")

        x_res[::2, ::2] = x * self.project_kernel[1, 1]
        
        return x_res
    
    def parse_schema(self, schema : str) -> int:
        self.schema = [char for char in schema]
        assert all([char in ["r", "p"] for char in self.schema])
        assert schema.count("r") == schema.count("p")
        assert self.schema[0] == "r"
        assert self.schema[-1] == "p"
        return len(self.schema) // 2