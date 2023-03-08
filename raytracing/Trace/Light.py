import Trace as tr
import torch

class Light(tr.Object):
    def __init__(
            self,
            position : list,
            color : list,
            intensity : float
        ):
        super().__init__(position)
        self.color = torch.tensor(color, dtype=torch.float32)
        self.intensity = intensity