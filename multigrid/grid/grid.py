import numpy as np
import solver
from scipy.ndimage import convolve
from scipy.interpolate import RegularGridInterpolator
from grid import grid_points

class Multigrid:

    def __init__(self, 
            stencil : np.ndarray, 
            b : np.ndarray, 
            restrict_kernel : np.ndarray,
            project_kernel : np.ndarray,
            schema : str = "rrpp",
            smoothing_steps : int = 16):
        self.stencil = stencil
        self.b = b
        self.smoothing_steps = smoothing_steps
        self.restrict_kernel = restrict_kernel
        self.project_kernel = project_kernel
        self.max_depth = self.parse_schema(schema=schema)

        self.defects = []
        self.bs = []
        self.xs = []

        n = b.shape[0]
        m = b.shape[1]

        self.defects.append(np.zeros((n,m)))
        self.bs.append(self.b)
        self.xs.append(self.b)
        
        for depth in range(self.max_depth-1):
            assert n % 2 == 0 and m % 2 == 0

            n = n / 2
            m = m / 2

            self.defects.append(np.zeros((n,m)))
            self.xs.append(np.zeros((n,m)))
            self.bs.append(self.restrict(self.bs[depth]))

    def solve(self):
        # We want to find x such that Ax = b
        # We can rewrite this as x = A_inv * b but this is not efficient
        # Instead we can use multigrid to solve this problem

        # initial guess
        x = self.b

        # Compute & Store defect
        self.defects[0] = self.get_defect()

        # coarse grid correction
        depth = 0
        for map in self.schema:
            if map == "r":
                # pre smoothing
                self.defects[depth] = self.get_defect(self.b[depth], self.xs[depth])
                correction = self.get_correction(x, self.defects[depth], self.smoothing_steps)
                self.xs[depth] += correction

                # Compute defect and restrict for next lower level
                self.defects[depth + 1] = self.restrict(self.get_defect[depth])

                # new depth
                depth += 1

            elif map == "p":
                depth -= 1
                self.defects[depth] = self.interpolate(self.get_defect[depth+1])

                # post smoothing
                self.defects[depth] = self.get_defect(self.b[depth], self.xs[depth])
                correction = self.get_correction(x, self.defects[depth], self.smoothing_steps)
                self.xs[depth] += correction


        return x
    
    def get_correction(self, defect : np.ndarray, steps : int) -> np.ndarray:
        correction = solver.StencilJacobi(
            self.stencil, defect, max_iterations=steps).solve()

        return correction
    
    def get_defect(self, b : np.ndarray, x : np.ndarray) -> np.ndarray:
        return b - convolve(x, self.stencil, mode='constant')
    
    def restrict(self, x : np.ndarray) -> np.ndarray:
        convolve(x, self.restrict_kernel, mode="constant")[::2, ::2]

    def interpolate(self, x : np.ndarray) -> np.ndarray:
        x_res = np.zeros((2 * x.shape[0], 2 * x.shape[1]))

        x_res[::2, ::2] = x
        x_res = convolve(x_res, self.project_kernel.T, mode="constant")

        return x_res
    
    def parse_schema(self, schema : str) -> int:
        self.schema = [char for char in schema]
        assert all([char in ["r", "p"] for char in self.schema])
        assert schema.count("r") == schema.count("p")
        assert self.schema[0] == "r"
        assert self.schema[-1] == "p"
        return len(self.schema) // 2