import torch
from abc import ABC, abstractmethod
import Trace as tr

class Object(ABC):
    def __init__(self, position : list):
        self.position = torch.tensor(position, dtype=torch.float32)

    @abstractmethod
    def intersect(self, rays : tr.Rays, horizon : float = 2^20, tolerance : float = 1e-5):
        pass