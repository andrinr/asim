import numpy as np
from abc import ABC, abstractmethod
from scipy.ndimage import convolve

class Solver(ABC):
    
    def __init__(self, 
        A : np.ndarray, 
        b : np.ndarray, 
        max_iterations : int = 16,
        epsilon : float = 1e-10):
        self.A = A
        self.b = b
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == b.shape[0]

        self.max_iterations = max_iterations
        assert self.max_iterations > 0

        self.epsilon = epsilon
        assert self.epsilon > 0
       
    @abstractmethod
    def solve(self) -> np.ndarray:
        pass

    def check_convergence(self, x : np.ndarray) -> bool:
        r = self.A @ x - self.b
        return np.linalg.norm(r) < self.epsilon
    
class StencilSolver(ABC):
    
    def __init__(self,
        stencil : np.ndarray,
        b : np.ndarray,
        max_iterations : int = 16,
        epsilon : float = 1e-10):

        self.stencil = stencil
        self.b = b

        self.max_iterations = max_iterations
        assert self.max_iterations > 0

        self.epsilon = epsilon
        assert self.epsilon > 0
       
    @abstractmethod
    def solve(self) -> np.ndarray:
        pass

    def check_convergence(self, x : np.ndarray) -> bool:
        r = convolve(x, self.stencil, mode="constant") - self.b
        return np.linalg.norm(r) < self.epsilon