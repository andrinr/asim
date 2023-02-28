import torch
from torch.nn.functional import normalize
from primitives import Rays, Light, Sphere

class Tracer:
    def __init__(
            self, 
            spheres : list[Sphere], 
            light : Light, 
            ambient : list,
            device : str,
            max_recursion_depth : int = 4,
            tolerance : float = 0.01,
            max_distance : float = 1000) -> None:

        self.device = device
        self.max_recursion_depth = max_recursion_depth
        self.spheres = spheres
        for sphere in spheres:
            sphere = sphere.to(device)

        self.light = light.to(device)
        self.ambient = torch.tensor(ambient, dtype=torch.float32).to(device)
        self.tolerance = tolerance
        self.maxDistance = max_distance

    def __call__(self, rays : Rays, recursion_depth : int = 0, shadow : bool = False):
        """
        GPU ray tracer
        """
        rays = rays.to(self.device)
        nm = rays.direction.shape[0]

        t = torch.full((nm, 1), self.maxDistance).to(self.device)
    
        t_ind = torch.full((nm, 1), 0).to(self.device)
        
        n_spheres = len(self.spheres)
        ambient_koef = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
        diffuse_koef = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
        specular_koef = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
        shininess_koef = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
        refraction_index = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
        transparency_koef = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
        reflection_koef = torch.zeros(n_spheres, dtype=torch.float32).to(self.device)
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
            refraction_index[index] = sphere.refractive_index
            colors[:,index] = sphere.color
            positions[:,index] = sphere.pos
            transparency_koef[index] = sphere.transparency
            reflection_koef[index] = sphere.reflection

        update = t < self.maxDistance
        update = update.squeeze()

        if shadow:
            return update

        # compute one hot encodings of sphere index which was hit by each ray
        t_ind = t_ind.squeeze(1)
        one_hot_indices = torch.nn.functional.one_hot(t_ind, num_classes=n_spheres)
        one_hot_indices = one_hot_indices.float()
        one_hot_indices = one_hot_indices.to(self.device)
        one_hot_indices = one_hot_indices.unsqueeze(-1)
        
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
        refraction_index = torch.sum(torch.mul(one_hot_indices, refraction_index), dim=1, keepdim=True)
        transparency_koef = torch.sum(torch.mul(one_hot_indices, transparency_koef), dim=1, keepdim=True)
        reflection_koef = torch.sum(torch.mul(one_hot_indices, reflection_koef), dim=1, keepdim=True)

        new_origin = torch.mul(t, rays.direction) + rays.origin
        normal = normalize(torch.sub(new_origin, sphere_pos))
        light = normalize(self.light.pos - new_origin)
        halfway = normalize(torch.add(light, -rays.direction))
     
        angle_light_normal = torch.sum(normal * light, dim=1, keepdim=True)
        torch.clamp(angle_light_normal, min=0, out=angle_light_normal)

        halfway_angle = torch.sum(normal * halfway, dim=1, keepdim=True)
        torch.clamp(halfway_angle, min=0, out=halfway_angle)

        shadow_ray_direction = normalize(self.light.pos - new_origin)
        shadow_rays = Rays(new_origin, shadow_ray_direction, None, rays.n, rays.m)
        shadow = self(shadow_rays, 0, True)
        shadow = ~shadow.unsqueeze(-1)

        # Blinn Phong: ambient, diffuse, specular
        color = self.ambient * ambient_koef
        color += diffuse_koef * base_color * angle_light_normal * shadow
        color += specular_koef * self.light.color * halfway_angle ** shininess_koef * shadow

        if recursion_depth < self.max_recursion_depth:

            ext_refl_direction = normalize(rays.direction - 2 * torch.sum(rays.direction * normal, dim=1, keepdim=True) * normal)
            # same refraction index as the ray does not enter the sphere
            ext_refl_rays = Rays(new_origin, ext_refl_direction, rays.refraction_index ,rays.n, rays.m)

            # for the refraction formula:
            # https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
            eta =  refraction_index / rays.refraction_index
            dot_prod = torch.sum(normal * rays.direction, dim=1, keepdim=True)
            k = 1 - eta ** 2 * (1 - dot_prod ** 2)
            k[k < 0] = 0
            int_refl_direction = normalize(eta * rays.direction - (eta * dot_prod + torch.sqrt(k)) * normal)
            int_refl_rays = Rays(new_origin, int_refl_direction, refraction_index, rays.n, rays.m)
            
            reflection = self(ext_refl_rays, recursion_depth + 1, False)
            refraction = self(int_refl_rays, recursion_depth + 1, False)

            # Fresnel
            # https://en.wikipedia.org/wiki/Schlick%27s_approximation
            r0 = (rays.refraction_index - refraction_index) / (rays.refraction_index + refraction_index)
            r0 = r0 ** 2
            fresnel = r0 + (1 - r0) * (1 - torch.abs(dot_prod)) ** 5

            color += fresnel * reflection * reflection_koef
            color += (1 - fresnel) * refraction * transparency_koef

        color[~update] = torch.tensor([0, 0, 0], dtype=torch.float32).to(self.device)

        return color
