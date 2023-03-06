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

    def to(self, device):
        self.origin = self.origin.to(device)
        self.direction = self.direction.to(device)

        return self
    

class Light:
    def __init__(self, pos : list, color : list = None):
        self.pos = torch.tensor(pos, dtype=torch.float32)
        self.color = torch.tensor(color, dtype=torch.float32)

    def to(self, device):
        self.pos = self.pos.to(device)
        self.color = self.color.to(device)
        return self

class Sphere:
    def __init__(self, 
        pos : list, 
        radius : list,
        color : list,
        specular : float = 0.0,
        diffuse : float = 0.0,
        ambient : float = 0.0,
        shininess : float = 0.0,
        refractive_index : float = 1.0,
        transparency : float = 1.0,
        reflection : float = 0.0):
        self.pos = torch.tensor(pos, dtype=torch.float32)
        self.radius = torch.tensor(radius, dtype=torch.float32)
        self.color = torch.tensor(color, dtype=torch.float32)
        self.specular = specular
        self.diffuse = diffuse
        self.ambient = ambient
        self.shininess = shininess
        self.refractive_index = refractive_index
        self.transparency = transparency
        self.reflection = reflection

    def intersect(self, rays : Rays):
        a = torch.sum(rays.direction ** 2, dim=1)
        ray_to_sphere = torch.subtract(rays.origin, sphere.pos)
        b = 2 * torch.sum(ray_to_sphere * rays.direction, dim=1)
        c = torch.sum(ray_to_sphere ** 2, dim=1) - sphere.radius ** 2
        disc = b ** 2 - 4 * a * c
        mask = disc >= 0
        mask = mask.unsqueeze(1)
        
        q = -(b + torch.sign(b) * torch.sqrt(disc)) / 2
        t_0 = torch.div(q, a).unsqueeze(1)
        t_0[(t_0 < 0) | (torch.abs(t_0) < self.tolerance) | ~mask] = self.maxDistance + 1
        t_1 = torch.div(c, q).unsqueeze(1)
        t_1[(t_1 < 0) | (torch.abs(t_1) < self.tolerance) | ~mask] = self.maxDistance + 1
        t_0 = torch.min(t_0, t_1)

        return t_0, mask

    def to(self, device):
        self.pos = self.pos.to(device)
        self.radius = self.radius.to(device)
        self.color = self.color.to(device)
        return self