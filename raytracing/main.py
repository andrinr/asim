import torch
import matplotlib.pyplot as plt

frustum_resolution = (1000, 1000)

# raytracer
eye_pos = torch.tensor([0, 0, 0], dtype=torch.float32)
frustum = torch.zeros((frustum_resolution[0], frustum_resolution[1], 3), dtype=torch.float32)
frustum[..., 0] = torch.linspace(-1, 1, frustum_resolution[0]).view(-1, 1)
frustum[..., 1] = torch.linspace(-1, 1, frustum_resolution[1]).view(1, -1)
frustum[..., 2] = 1

# rays. xzy pos and xyz dir
ray_origin = eye_pos.view(1, 1, 3).expand(frustum_resolution[0], frustum_resolution[1], 3)
ray_direction = frustum - ray_origin

# sphere xyz pos and radius
sphere_pos = torch.tensor([0, 0, 3], dtype=torch.float32)
sphere_radius = 1

# light xyz pos and rgb color
light_pos = torch.tensor([0, 0, 5], dtype=torch.float32)
light_color = torch.tensor([1, 1, 1], dtype=torch.float32)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

ray_origin = ray_origin.to(device)
ray_direction = ray_direction.to(device)
sphere_pos = sphere_pos.to(device)
light_pos = light_pos.to(device)

# check if ray intersects with object
def ray_sphere_intersect(ray_origin, ray_direction, sphere_pos, sphere_radius):
    a = torch.sum(ray_direction ** 2, dim=2)
    ray_to_sphere = torch.subtract(ray_origin, sphere_pos)
    b = 2 * torch.sum(ray_to_sphere * ray_direction, dim=2)
    c = torch.sum(ray_to_sphere ** 2, dim=2) - sphere_radius ** 2
    alpha = b ** 2 - 4 * a * c >= 0
    alpha = alpha.unsqueeze(2)
    
    q = - b + torch.sign(b) * torch.sqrt(b ** 2 * 4 * a * c) / 2

    t0 = torch.div(q, a).unsqueeze(2)
    t1 = torch.div(c, q).unsqueeze(2)

    p0 = torch.mul(ray_direction, t0) + ray_origin
    p1 = ray_direction * t1 + ray_origin
    return alpha, p0, p1

#hit_mask, p0, p1 = ray_sphere_intersect(ray_origin, ray_direction, sphere_pos, sphere_radius).numpy()
alpha, p0, p1 = ray_sphere_intersect(ray_origin, ray_direction, sphere_pos, sphere_radius)

normals = p0 - sphere_pos
normals = torch.div(normals, torch.norm(normals, dim=2, keepdim=True))

light_dir = light_pos - p0
light_dir = torch.div(light_dir, torch.norm(light_dir, dim=2, keepdim=True))

diffuse = (normals+0.5)*alpha
diffuse = p1

image = diffuse.cpu().numpy().astype(float)
print(image)
print(image.shape)
print(image.dtype)
plt.imshow(image)
plt.show()
