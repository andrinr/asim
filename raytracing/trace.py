import torch

class Rays:
    def __init__(
            self, 
            origin : torch.TensorType, 
            direction : torch.TensorType, 
            n : int, 
            m : int,
            color : torch.TensorType = None):
        self.n = n
        self.m = m
        self.origin = origin
        self.direction = direction
        self.color = color

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
        ray_direction = ray_direction.view(n*m, 3)
        ray_direction = torch.div(ray_direction, torch.norm(ray_direction, dim=1, keepdim=True))
        ray_color = torch.ones_like(ray_direction)
        return Rays(ray_origin, ray_direction, n, m, ray_color)

    def to(self, device):
        self.origin = self.origin.to(device)
        self.direction = self.direction.to(device)
        self.color = self.color.to(device)
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
        shininess : float = 0.0):
        self.pos = torch.tensor(pos, dtype=torch.float32)
        self.radius = torch.tensor(radius, dtype=torch.float32)
        self.color = torch.tensor(color, dtype=torch.float32)
        self.specular = specular
        self.diffuse = diffuse
        self.ambient = ambient
        self.shininess = shininess

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
            ambient : list,
            device : str,
            tolerance : float = 0.01,
            maxDistance : float = 1000) -> None:

        self.device = device
        self.spheres = spheres
        for sphere in spheres:
            sphere = sphere.to(device)

        self.light = light.to(device)
        self.ambient = torch.tensor(ambient, dtype=torch.float32).to(device)
        self.tolerance = tolerance
        self.maxDistance = maxDistance

    def __call__(self, rays : Rays):
        """
        GPU ray tracer
        """
        rays = rays.to(self.device)
        nm = rays.direction.shape[0]

        t = torch.full((nm, 1), self.maxDistance).to(self.device)
    
        t_ind = torch.full((nm, 1), -1).to(self.device)
        
        n_spheres = len(self.spheres)
        ambient_koef = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
        diffuse_koef = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
        specular_koef = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
        shininess_koef = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
        colors = torch.zeros((3, n_spheres), dtype=torch.float32).to(self.device)
        positions = torch.zeros((3, n_spheres), dtype=torch.float32).to(self.device)

        for index, sphere in enumerate(self.spheres):
            a = torch.sum(rays.direction ** 2, dim=1)
            ray_to_sphere = torch.subtract(rays.origin, sphere.pos)
            b = 2 * torch.sum(ray_to_sphere * rays.direction, dim=1)
            c = torch.sum(ray_to_sphere ** 2, dim=1) - sphere.radius ** 2
            disc = b ** 2 - 4 * a * c
            mask = disc >= 0
            mask = mask.unsqueeze(1)
            
            q = -(b + torch.sign(b) * torch.sqrt(disc)) / 2
            t_0 = torch.div(q, a).unsqueeze(1)
            t_0[(t_0 < 0) | (torch.abs(t_0) < self.tolerance) | ~mask] = self.maxDistance
            t_1 = torch.div(c, q).unsqueeze(1)
            t_1[(t_1 < 0) | (torch.abs(t_1) < self.tolerance) | ~mask] = self.maxDistance
            t_0 = torch.min(t_0, t_1)
            t = torch.min(t, t_0)
            t_ind[t_0 == t] = index

            # store params in datastructure
            ambient_koef[index] = sphere.ambient
            diffuse_koef[index] = sphere.diffuse
            specular_koef[index] = sphere.specular
            shininess_koef[index] = sphere.shininess
            colors[:,index] = sphere.color
            positions[:,index] = sphere.pos

        # compute one hot encodings of sphere index which was hit by each ray
        t_ind = t_ind.squeeze(1)
        one_hot_indices = torch.nn.functional.one_hot(t_ind, num_classes=n_spheres)
        one_hot_indices = one_hot_indices.float()
        one_hot_indices = one_hot_indices.to(self.device)
        one_hot_indices = one_hot_indices.unsqueeze(-1)
        
        #base_color
        colors = colors.view(1, n_spheres, 3)
        colors = colors.expand(nm, -1, -1)
        base_color = torch.matmul(colors, one_hot_indices)
        base_color = base_color.squeeze()
        
        positions = positions.view(1, n_spheres, 3)
        positions = positions.expand(nm, -1, -1)
        sphere_pos = torch.matmul(positions, one_hot_indices)
        sphere_pos = sphere_pos.squeeze()
        
        one_hot_indices = one_hot_indices.squeeze()
        ambient_koef = torch.sum(torch.mul(one_hot_indices, ambient_koef), dim=1, keepdim=True)
        specular_koef = torch.sum(torch.mul(one_hot_indices, specular_koef), dim=1, keepdim=True)
        diffuse_koef = torch.sum(torch.mul(one_hot_indices, diffuse_koef), dim=1, keepdim=True)
        shininess_koef = torch.sum(torch.mul(one_hot_indices, shininess_koef), dim=1, keepdim=True)

        print(ambient_koef.shape)

        new_origin = torch.mul(t, rays.direction) + rays.origin
        normal = torch.sub(new_origin, sphere_pos)
        normal = torch.div(normal, torch.norm(normal, dim=1, keepdim=True))
        light = self.light.pos - new_origin
        light = torch.div(light, torch.norm(light, dim=1, keepdim=True))
        halfway = torch.add(light, -rays.direction)
        halfway = torch.div(halfway, torch.norm(halfway, dim=1, keepdim=True))
        angle_light_normal = torch.sum(normal * light, dim=1, keepdim=True)
        torch.clamp(angle_light_normal, min=0, out=angle_light_normal)

        halfway_angle = torch.sum(normal * halfway, dim=1, keepdim=True)
        torch.clamp(halfway_angle, min=0, out=halfway_angle)
        # ambient contribution
        color = self.ambient * ambient_koef
        # diffuse contribution
        color += diffuse_koef * base_color * angle_light_normal
        # specular contribution
        color += specular_koef * self.light.color * halfway_angle ** shininess_koef


        update = t < self.maxDistance
        update = update.squeeze()
        color[~update] = torch.tensor([0, 0, 0], dtype=torch.float32).to(self.device)

        new_direction = self.light.pos - new_origin
        new_direction = torch.div(new_direction, torch.norm(new_direction, dim=1, keepdim=True))

        new_origin[~update] = rays.origin[~update]
        new_direction[~update] = rays.direction[~update]

        new_rays = Rays(new_origin, new_direction, rays.n, rays.m, color)
        return new_rays, update
