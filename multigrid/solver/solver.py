import numpy as np
from abc import ABC, abstractmethod

class Solver(ABC):
    
    def __init__(self, A : np.ndarray, b : np.ndarray, max_iterations : int = 16):
        self.A = A
        self.b = b

        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == b.shape[0]

        self.max_iterations = max_iterations
       
    @abstractmethod
    def solve(self) -> np.ndarray:
        pass