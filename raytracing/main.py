import torch

frustum_resolution = (256, 256)

# raytracer
eye_pos = torch.tensor([0, 0, 0], dtype=torch.float32)
frustum = torch.zeros((frustum_resolution[0], frustum_resolution[1], 3), dtype=torch.float32)
frustum[..., 0] = torch.linspace(-1, 1, frustum_resolution[0]).view(-1, 1)
frustum[..., 1] = torch.linspace(-1, 1, frustum_resolution[1]).view(1, -1)
frustum[..., 2] = 1

# rays. xzy pos and xyz dir
rays = torch.zeros((frustum_resolution[0], frustum_resolution[1], 6), dtype=torch.float32)
rays[..., :3] = eye_pos.view(1, 1, 3)

# sphere xyz pos and radius
sphere = torch.tensor([0, 0, 4, 1], dtype=torch.float32)

# light xyz pos and rgb color
light = torch.tensor([0, 0, 3, 1, 1, 1], dtype=torch.float32)

# check if ray intersects with object
def ray_sphere_intersect(rays, sphere):
    n_spheres = sphere.shape[0]
    for i in range(n_spheres):
        sphere_pos = sphere[i, :3]
        sphere_radius = sphere[i, 3]
        
        
