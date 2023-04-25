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
            approximation_steps : int = 16,
            exact_steps : int = 128):
        self.stencil = stencil
        self.b = b
        self.approx_steps = approximation_steps
        self.exact_steps = exact_steps
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

    def solve(self):
        # We want to find x such that Ax = b
        # We can rewrite this as x = A_inv * b but computing A_inv is very expensive

        # initial guess
        x = self.b

        # Compute & Store defect
        self.defects[0] = self.get_defect()

        # coarse grid correction
        depth = 0
        for map in self.schema:
            if map == "restrict":
                # pre approximate smoothing

                # QUESTION: How to compute defect? Do we always use the same x or do we update the x 
                # and then prolong and correct it?
                self.defects[depth] =\
                    self.get_defect(self.bs[depth], self.xs[depth])
                self.corrections[depth] =\
                    self.get_correction(self.defects[depth], self.approx_steps)
                self.xs[depth] += self.corrections[depth]

                # Recompute defect and restrict for next lower level
                self.defects[depth] = self.get_defect(self.bs[depth], self.xs[depth])
                self.defects[depth + 1] = self.restrict(self.defects[depth])

                # new depth
                depth += 1

            elif map == "prolongate":
                if depth == self.max_depth - 1:
                    # exact smoothing using precomputed defect
                    self.xs[depth] =\
                        self.smooth(self.defects[depth], self.bs[depth], self.exact_steps)

                depth -= 1
                self.defects[depth] =\
                    self.prolongate(self.bs[depth], self.get_defect[depth+1])
                
                self.xs[depth] += self.defects[depth]

                # post approximate smoothing
                self.smooth(depth, self.approx_steps)

            elif map == "apply":
                x = self.xs[0]
            
            elif map == "smooth":
                self.smooth(depth, self.approx_steps)

            else: 
                raise ValueError("Unknown map: {}".format(map))

        return x
            

        return self.xs[0]
    
    def get_correction(self, defect : np.ndarray, steps : int) -> np.ndarray:
        correction = solver.StencilJacobi(
            self.stencil, defect, max_iterations=steps).solve()

        return correction
    
    def get_defect(self, b : np.ndarray, x : np.ndarray) -> np.ndarray:
        return b - convolve(x, self.stencil, mode='constant')
    
    def smooth(self, defect : np.ndarray, x : np.ndarray, steps):
        correction = self.get_correction(defect, steps)
        x += correction
        return x
    
    def restrict(self, x : np.ndarray) -> np.ndarray:
        return convolve(x, self.restrict_kernel, mode="constant")[::2, ::2]

    def prolongate(self, x : np.ndarray) -> np.ndarray:
        x_res = np.zeros((2 * x.shape[0], 2 * x.shape[1]))

        x_res[::2, ::2] = x
        x_res = convolve(x_res, self.prolong_kernel.T, mode="constant")

        return x_res
    
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

        print(new_schema_list)

        self.schema = new_schema_list

        print(max_depth)

        return max_depth