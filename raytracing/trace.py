import torch

class Rays:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def create(origin, resolution, distance = 1):
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

class Light:
    def __init__(self, pos):
        self.pos = pos

    def to(self, device):
        self.pos = self.pos.to(device)

class Sphere:
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

    def to(self, device):
        self.pos = self.pos.to(device)
        self.radius = self.radius.to(device)

    def intersect(self, rays : Rays, light : Light):
        a = torch.sum(rays.direction ** 2, dim=2)
        ray_to_sphere = torch.subtract(rays.origin, self.pos)
        b = 2 * torch.sum(ray_to_sphere * rays.direction, dim=2)
        c = torch.sum(ray_to_sphere ** 2, dim=2) - self.radius ** 2
        disc = b ** 2 - 4 * a * c
        mask = disc >= 0
        mask = mask.unsqueeze(2)
        
        q = - (b + torch.sign(b) * torch.sqrt(disc)) / 2
        t0 = torch.div(q, a).unsqueeze(2)
        t1 = torch.div(c, q).unsqueeze(2)
        t = torch.min(t0, t1)

        mask = mask & (torch.abs(t) > 0.001)
        intersection_point = torch.mul(t, rays.direction) + rays.origin

        #normals = intersection_point - sphere_pos
        #normals = torch.div(normals, torch.norm(normals, dim=2, keepdim=True))

        ray_direction_shadow = light.pos - intersection_point
        ray_direction_shadow = torch.div(ray_direction_shadow, torch.norm(ray_direction_shadow, dim=2, keepdim=True))

        return intersection_point, ray_direction_shadow, mask


