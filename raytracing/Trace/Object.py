import torch
from abc import ABC, abstractmethod

class Object(ABC):
    def __init__(self, position : list):
        self.position = torch.tensor(position, dtype=torch.float32)