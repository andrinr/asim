import torch
from torch.nn.functional import normalize
import Trace as tr
from tqdm import tqdm

class Tracer:
    def __init__(
            self, 
            meshes : list[tr.Mesh], 
            light : tr.Light, 
            ambient : list,
            horizon : float = 1000.0,
            max_recursion_depth : int = 4,
            air_refraction_index : float = 1.0) -> None:
        self.max_recursion_depth = max_recursion_depth
        self.meshes = meshes
        self.light = light
        self.ambient = torch.tensor(ambient, dtype=torch.float32)
        self.air_refraction_index = air_refraction_index
        self.horizon = horizon

    def __call__(self, rays_grid : tr.RaysGrid, recursion_depth : int = 0, shadow : bool = False):
        local_n = rays_grid.rays[0].n
        local_m = rays_grid.rays[0].m
        color = torch.zeros((local_n * rays_grid.n, local_m * rays_grid.m, 3), dtype=torch.float32)
        for k in tqdm(range(rays_grid.n * rays_grid.m)):
            i = k // rays_grid.m
            j = k % rays_grid.m

            local_color = self.trace(rays_grid.rays[k], recursion_depth, shadow)
            color[i * local_n:(i + 1) * local_n, j * local_m:(j + 1) * local_m] = local_color.view(local_n, local_m, 3)
        
        return color

    def trace(self, rays : tr.Rays, recursion_depth : int = 0, shadow : bool = False):
        """
        GPU ray tracer
        """
        nm = rays.direction.shape[0]

        t = torch.full((nm, 1), float('inf'))
    
        n_meshes = len(self.meshes)
        material_id = torch.full((nm, 1), 0)
        normals = torch.zeros((nm, 3), dtype=torch.float32)
        
        ambient_koef = torch.zeros(n_meshes, dtype=torch.float32)
        diffuse_koef = torch.zeros(n_meshes, dtype=torch.float32) 
        specular_koef = torch.zeros(n_meshes, dtype=torch.float32) 
        shininess_koef = torch.zeros(n_meshes, dtype=torch.float32) 
        refraction_koef = torch.ones(n_meshes, dtype=torch.float32) * self.air_refraction_index
        transparency_koef = torch.zeros(n_meshes, dtype=torch.float32) 
        reflection_koef = torch.zeros(n_meshes, dtype=torch.float32) 
        colors = torch.zeros((3, n_meshes), dtype=torch.float32) 
   
        for index, mesh in enumerate(self.meshes):
            t_0, normals_ = mesh.object.intersect(rays, self.horizon)
            t = torch.min(t, t_0)
            material_id[t_0 == t] = index
            t = t.squeeze()
            t_0 = t_0.squeeze()

            normals[t_0 == t, :] = normals_[t_0 == t, :]

            t = t.unsqueeze(1)
            t_0 = t_0.unsqueeze(1)

            # store material koeficients
            ambient_koef[index] = mesh.material.ambient
            diffuse_koef[index] = mesh.material.diffuse
            specular_koef[index] = mesh.material.specular
            shininess_koef[index] = mesh.material.shininess
            refraction_koef[index] = mesh.material.refractive_index
            colors[:,index] = mesh.material.color
            transparency_koef[index] = mesh.material.transparency
            reflection_koef[index] = mesh.material.reflection

        update = t < self.horizon
        update = update.squeeze()

        if shadow:
            return update
        
        if torch.sum(update) == 0:
            return torch.zeros((nm, 3), dtype=torch.float32)

        # compute one hot encodings of sphere index which was hit by each ray
        material_id = material_id.squeeze(1)
        material_id = torch.nn.functional.one_hot(material_id, num_classes=n_meshes)
        material_id = material_id.float()
        material_id = material_id 

        ambient_koef = torch.sum(torch.mul(material_id, ambient_koef), dim=1, keepdim=True)
        specular_koef = torch.sum(torch.mul(material_id, specular_koef), dim=1, keepdim=True)
        diffuse_koef = torch.sum(torch.mul(material_id, diffuse_koef), dim=1, keepdim=True)
        shininess_koef = torch.sum(torch.mul(material_id, shininess_koef), dim=1, keepdim=True)
        reflection_koef = torch.sum(torch.mul(material_id, reflection_koef), dim=1, keepdim=True)
        transparency_koef = torch.sum(torch.mul(material_id, transparency_koef), dim=1, keepdim=True)
        refraction_koef = torch.sum(torch.mul(material_id, refraction_koef), dim=1, keepdim=True)

        material_id = material_id.unsqueeze(-1)
        
        colors = colors.view(1, 3, n_meshes)
        colors = colors.expand(nm, -1, -1)
        base_color = torch.matmul(colors, material_id)
        base_color = base_color.squeeze()
    
        new_origin = torch.mul(t, rays.direction) + rays.origin
        light = normalize(self.light.position - new_origin)
        halfway = normalize(torch.add(light, -rays.direction))
     
        angle_light_normal = torch.sum(normals * light, dim=1, keepdim=True)
        torch.clamp(angle_light_normal, min=0, out=angle_light_normal)

        halfway_angle = torch.sum(normals * halfway, dim=1, keepdim=True)
        torch.clamp(halfway_angle, min=0, out=halfway_angle)

        shadow_ray_direction = normalize(self.light.position - new_origin)
        shadow_rays = tr.Rays(new_origin + 0.0001 * normals, shadow_ray_direction, rays.n, rays.m)
        shadow = self.trace(shadow_rays, 0, True)
        shadow = shadow.unsqueeze(-1)
        shadow = ~shadow

        # Blinn Phong: ambient, diffuse, specular
        color = (1-transparency_koef) * self.ambient * ambient_koef
        color += (1-transparency_koef) * diffuse_koef * base_color * angle_light_normal * shadow
        color += (1-transparency_koef) * specular_koef * self.light.color * halfway_angle ** shininess_koef * shadow

        if recursion_depth + 1 < self.max_recursion_depth:
            
            if torch.sum(reflection_koef) > 0:
                dot_prod = torch.sum(normals * rays.direction, dim=1, keepdim=True)

                ext_refl_direction = normalize(rays.direction - 2 * dot_prod * normals)
                # same refraction index as the ray does not enter the sphere
                ext_refl_rays = tr.Rays(new_origin + 0.001 * normals, ext_refl_direction, rays.n, rays.m)
                ext_reflection = self.trace(ext_refl_rays, recursion_depth + 1, False)
                
                color = torch.max(ext_reflection * reflection_koef, color)

            if torch.sum(transparency_koef) > 0:
                dot_prod = torch.sum(normals * rays.direction, dim=1, keepdim=True)

                entering = torch.sum(normals * rays.direction, dim=1, keepdim=True) < 0
                # for the refraction formula:
                # https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
                n1 = torch.where(entering, self.air_refraction_index, refraction_koef)
                n2 = torch.where(entering, refraction_koef, self.air_refraction_index)
                n = n1 / n2;
                k = 1. - n ** 2 * (1 - dot_prod ** 2)
                
                int_refl_direction = torch.where(
                    k < 0, 
                    torch.zeros(nm, 3), 
                    normalize(n * rays.direction + (n * dot_prod - torch.sqrt(k)) * normals)
                )

                # fix origin offset
                int_refl_rays = tr.Rays(new_origin, int_refl_direction, rays.n, rays.m)
                
                int_reflection = self.trace(int_refl_rays, recursion_depth + 1, False)
                color = torch.max(color, int_reflection * transparency_koef)

            # Fresnel
            # https://en.wikipedia.org/wiki/Schlick%27s_approximation
            # r0 = (n1 - n2) / (n1 + n1)
            # r0 = r0 ** 2 
            # fresnel = r0 + (1 - r0) * (1 - torch.abs(dot_prod)) ** 5
            
            #color = torch.max(ext_reflection * reflection_koef, color)
            #color = torch.max(int_reflection * transparency_koef, color)

            #color += fresnel * (ext_reflection * reflection_koef)
            #color += (1-fresnel) * int_reflection * transparency_koef

            #color = sphere_pos

            #color = torch.max(fresnel * ext_reflection * reflection_koef, color)
            #color = torch.max((1-fresnel) * int_reflection * transparency_koef, color)

        #color = t_0 * torch.tensor([1.0, 1.0, 1.0]) * 0.1
        #color = normals * 0.5 + 0.5
        color[~update] = torch.tensor([0, 0, 0], dtype=torch.float32) 

        return color
