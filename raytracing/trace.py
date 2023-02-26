import torch
import typing

class Rays:
    def __init__(self, origin : torch.TensorType, direction : torch.TensorType):
        self.origin = origin
        self.direction = direction
        self.color = torch.ones_like(self.direction)

    def create(origin, resolution : tuple, distance = 1):
        frustum = torch.ones((resolution[0], resolution[1], 3), dtype=torch.float32)
        frustum[..., 0] = torch.linspace(-1, 1, resolution[0]).view(-1, 1)
        frustum[..., 1] = torch.linspace(-1, 1, resolution[1]).view(1, -1)
        frustum[..., 2] = distance
        # rays. xzy pos and xyz dir
        ray_origin = origin.view(1, 1, 3).expand(resolution[0], resolution[1], 3)
        ray_direction = frustum - ray_origin
        ray_direction = torch.div(ray_direction, torch.norm(ray_direction, dim=2, keepdim=True))

        return Rays(ray_origin, ray_direction)

    def to(self, device):
        self.origin = self.origin.to(device)
        self.direction = self.direction.to(device)
        return self

class Light:
    def __init__(self, pos : list):
        self.pos = torch.tensor(pos, dtype=torch.float32)

    def to(self, device):
        self.pos = self.pos.to(device)
        return self

class Sphere:
    def __init__(self, 
        pos : list, 
        radius : list,
        color : list):
        self.pos = torch.tensor(pos, dtype=torch.float32)
        self.radius = torch.tensor(radius, dtype=torch.float32)
        self.color = torch.tensor(color, dtype=torch.float32)

    def to(self, device):
        self.pos = self.pos.to(device)
        self.radius = self.radius.to(device)
        self.color = self.color.to(device)
        return self

class Tracer:
    def __init__(
            self, 
            spheres : list[Sphere], 
            light : Light, 
            device : str,
            tolerance : float = 0.01,
            maxDistance : float = 1000) -> None:

        self.device = device
        self.spheres = spheres
        for sphere in spheres:
            sphere = sphere.to(device)

        self.light = light.to(device)
        self.tolerance = tolerance
        self.maxDistance = maxDistance

    def __call__(self, rays : Rays):
        """
        GPU ray tracer
        """
        rays = rays.to(self.device)
        t = torch.full(
            [rays.direction.shape[0], 
            rays.direction.shape[1], 1],
            self.maxDistance).to(self.device)
        
        for sphere in self.spheres:
            a = torch.sum(rays.direction ** 2, dim=2)
            ray_to_sphere = torch.subtract(rays.origin, sphere.pos)
            b = 2 * torch.sum(ray_to_sphere * rays.direction, dim=2)
            c = torch.sum(ray_to_sphere ** 2, dim=2) - sphere.radius ** 2
            disc = b ** 2 - 4 * a * c
            mask = disc >= 0
            mask = mask.unsqueeze(2) 
            
            q = -(b + torch.sign(b) * torch.sqrt(disc)) / 2
            t_0 = torch.div(q, a).unsqueeze(2)
            t_0[(t_0 < 0) | (torch.abs(t_0) < self.tolerance) | ~mask] = self.maxDistance
            t_1 = torch.div(c, q).unsqueeze(2)
            t_1[(t_1 < 0) | (torch.abs(t_1) < self.tolerance) | ~mask] = self.maxDistance

            t = torch.min(t, torch.min(t_0, t_1))

        update = t < self.maxDistance
        update = update.squeeze()

        new_origin = torch.mul(t, rays.direction) + rays.origin
        
        new_direction = self.light.pos - new_origin
        new_direction = torch.div(new_direction, torch.norm(new_direction, dim=2, keepdim=True))

        new_origin[~update] = rays.origin[~update]
        new_direction[~update] = rays.direction[~update]

        new_rays = Rays(new_origin, new_direction)
        return new_rays, update
