import torch

class Rays:
    def __init__(
            self, 
            origin : torch.TensorType, 
            direction : torch.TensorType, 
            refraction_index : torch.TensorType,
            n : int, 
            m : int):
        self.n = n
        self.m = m
        self.origin = origin
        self.direction = direction
        self.refraction_index = refraction_index

    def create(origin, resolution : tuple, distance = 1, refraction_index = 1.0):
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
        ray_direction = ray_direction.view(n*m, 3)
        ray_direction = torch.div(ray_direction, torch.norm(ray_direction, dim=1, keepdim=True))

        refraction_index = torch.ones((n*m, 1), dtype=torch.float32) * refraction_index
        return Rays(ray_origin, ray_direction, refraction_index, n, m)

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

    def to(self, device):
        self.pos = self.pos.to(device)
        self.radius = self.radius.to(device)
        self.color = self.color.to(device)
        return self