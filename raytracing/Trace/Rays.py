import torch
from torch.nn.functional import normalize

class Rays:
    def __init__(
            self, 
            origin : torch.TensorType, 
            direction : torch.TensorType,
            n : int, 
            m : int):
        self.n = n
        self.m = m
        self.origin = origin
        self.direction = direction

    def create(origin, resolution : tuple, distance = 1):
        n = resolution[0]
        m = resolution[1]
        frustum = torch.ones((resolution[0], resolution[1], 3), dtype=torch.float32)
        frustum[..., 0] = torch.linspace(-1, 1, resolution[0]).view(-1, 1)
        frustum[..., 1] = torch.linspace(-1, 1, resolution[1]).view(1, -1)
        frustum[..., 2] = distance
        # rays. xzy pos and xyz dir
        ray_origin = origin.view(1, 1, 3).expand(resolution[0], resolution[1], 3)
        ray_direction = frustum - ray_origin

        ray_origin = ray_origin.view(n*m, 3)
        ray_direction = normalize(ray_direction.view(n*m, 3))

        return Rays(ray_origin, ray_direction, n, m)

