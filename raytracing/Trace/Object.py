import torch
from abc import ABC

class Object(ABC):
    def __init__(self, position : list):
        self.position = torch.tensor(position, dtype=torch.float32)
