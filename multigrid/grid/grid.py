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
            prolong_kernel : np.ndarray,
            schema : str = "rrpp",
            correction_steps : int = 16,
            exact_steps : int = 128):
        self.stencil = stencil
        self.b = b
        self.smoothing_steps = correction_steps
        self.correction_steps = exact_steps
        self.restrict_kernel = restrict_kernel
        self.prolong_kernel = prolong_kernel
        self.max_depth = self.parse_schema(schema=schema)

        self.defects = []
        self.bs = []
        self.xs = []
        self.corrections = []

        n = b.shape[0]
        m = b.shape[1]

        self.defects.append(np.zeros((n,m)))
        self.bs.append(self.b)
        self.xs.append(self.b)
        self.corrections.append(np.zeros((n,m)))
        
        for depth in range(self.max_depth):
            assert n % 2 == 0 and m % 2 == 0

            n = int(n / 2)
            m = int(m / 2)

            self.defects.append(np.zeros((n,m)))
            self.xs.append(np.zeros((n,m)))
            self.bs.append(self.restrict(self.bs[depth]))
            self.corrections.append(np.zeros((n,m)))

    def solve(self, x : np.ndarray = None):
        # We want to find x such that Ax = b
        # We can rewrite this as x = A_inv * b but computing A_inv is very expensive

        # initial (bad) guess
        if x is None:
            x = self.b
        depth = 0

        print("Solving...")
        print("Error: ", self.error(x))
        # pre smoothing
        self.defects[depth] =\
            self.get_defect(self.bs[depth], x)
        
        self.corrections[depth] = solver.StencilJacobi(
            stencil=self.stencil,
            b=-self.defects[depth],
            max_iterations=self.smoothing_steps
        ).solve()
        print("defect: ", -self.defects[depth])
        print("x: ", x)
        print("b: ", self.bs[depth])
        print("corrections: ", self.corrections[depth])
        x += self.corrections[depth]

        print("Error: ", self.error(x))

        # fine grid defect
        self.defects[depth] =\
            self.get_defect(self.bs[depth], self.xs[depth])
        
        # coarse grid correction
        for map in self.schema:
            if map == "restrict":
                depth += 1

                # Restrict defect and correction
                self.defects[depth] =\
                    self.restrict(self.defects[depth-1])
                self.corrections[depth] =\
                    self.restrict(self.corrections[depth-1])
                
                # Improve correction
                self.corrections[depth] = solver.StencilJacobi(
                    stencil=self.stencil,
                    b=self.defects[depth],
                    max_iterations=self.smoothing_steps
                ).solve(self.corrections[depth])

            elif map == "prolongate":
                depth -= 1
                # Prolong defect and correction
                self.defects[depth] =\
                    self.restrict(self.defects[depth+1])
                self.corrections[depth] =\
                    self.restrict(self.corrections[depth+1])

                # Improve correction
                self.corrections[depth] = solver.StencilJacobi(
                    stencil=self.stencil,
                    b=self.defects[depth],
                    max_iterations=self.smoothing_steps
                ).solve(self.corrections[depth])

            elif map == "smooth":
                # Improve correction
                self.corrections[depth] = solver.StencilJacobi(
                    stencil=self.stencil,
                    b=self.defects[depth],
                    max_iterations=self.smoothing_steps
                ).solve(self.corrections[depth])
                
            elif map == "apply":
                pass
            
            else: 
                raise ValueError("Unknown map: {}".format(map))


        # post smoothing
        self.defects[depth] =\
            self.get_defect(self.bs[depth], x)
        
        self.corrections[depth] = solver.StencilJacobi(
            stencil=self.stencil,
            b=self.defects[depth],
            max_iterations=self.smoothing_steps
        ).solve()
        x += self.corrections[depth]

        return x
    
    def get_defect(self, b : np.ndarray, x : np.ndarray) -> np.ndarray:
        return convolve(x, self.stencil, mode='constant') - b
    
    def restrict(self, x : np.ndarray) -> np.ndarray:
        return convolve(x, self.restrict_kernel, mode="constant")[::2, ::2]

    def prolongate(self, x : np.ndarray) -> np.ndarray:
        x_res = np.zeros((2 * x.shape[0], 2 * x.shape[1]))

        x_res[::2, ::2] = x
        x_res = convolve(x_res, self.prolong_kernel.T, mode="constant")

        return x_res
    
    def error(self, x : np.ndarray) -> float:
        return np.linalg.norm(self.get_defect(self.b, x), ord=2)
    
    def parse_schema(self, schema : str) -> int:
        schema_list = [char for char in schema]
        assert all([char in ["r", "p"] for char in schema_list])
        assert schema.count("r") == schema.count("p")
        assert schema_list[0] == "r"
        assert schema_list[-1] == "p"

        print(schema_list)

        def map_func(x):
            if x == "r":
                return "restrict"
            elif x == "p":
                return "prolongate"
            else:
                raise ValueError("Invalid schema")
            
        schema_list = list(map(map_func, schema_list))

        new_schema_list = []
        depth = 0
        max_depth = 0
        for i in range(len(schema_list) - 1):
            new_schema_list.append(schema_list[i])
            if schema_list[i] == "restrict":
                depth += 1
            else:
                depth -= 1

            max_depth = max(max_depth, depth)
            # detect "rp" which is a valley where 
            if schema_list[i] == "restrict" and schema_list[i+1] == "prolongate":
                new_schema_list.append("smooth")

            # detect "pr" which is a peak
            if schema_list[i] == "prolongate" and schema_list[i+1] == "restrict":
                new_schema_list.append("apply")

        new_schema_list.append(schema_list[-1])
        new_schema_list.append("apply")
        self.schema = new_schema_list
        return max_depth