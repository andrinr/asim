import torch
import numpy as np
import Trace as tr

class Sphere(tr.Object):
    def __init__(
            self,
            position : list,
            radius : float
        ):
        super().__init__(position)
        self.radius = radius

    def intersect(self, rays : tr.Rays, horizon : float = 2^20, tolerance : float = 1e-5):
        a = torch.sum(rays.direction ** 2, dim=1)
        ray_to_sphere = torch.subtract(rays.origin, self.position)
        b = 2 * torch.sum(ray_to_sphere * rays.direction, dim=1)
        c = torch.sum(ray_to_sphere ** 2, dim=1) - self.radius ** 2
        disc = b ** 2 - 4 * a * c
        mask = disc >= 0
        
        if torch.sum(mask) == 0:
            return torch.full((rays.n * rays.m, 1), horizon + 1), torch.zeros((rays.n * rays.m, 3))
        
        mask = mask.unsqueeze(1)
        
        q = -(b + torch.sign(b) * torch.sqrt(disc)) / 2
        t_0 = torch.div(q, a).unsqueeze(1)
        t_0[(t_0 < 0) |  ~mask] = horizon + 1
        t_1 = torch.div(c, q).unsqueeze(1)
        t_1[(t_1 < 0) | ~mask] = horizon + 1
        t_0 = torch.min(t_0, t_1)

        normals = (rays.origin + t_0 * rays.direction - self.position) / self.radius

        return t_0, normals