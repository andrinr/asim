import torch
import matplotlib.pyplot as plt

frustum_resolution = (256, 256)

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

# check if ray intersects with object
def ray_sphere_intersect(ray_origin, ray_direction, sphere_pos, sphere_radius):
    a = torch.sum(ray_direction ** 2, dim=2)
    ray_to_sphere = torch.subtract(ray_origin, sphere_pos)
    b = 2 * torch.sum(ray_to_sphere * ray_direction, dim=2)
    c = torch.sum(ray_to_sphere ** 2, dim=2) - sphere_radius ** 2
    hit = b ** 2 - 4 * a * c >= 0
    q = - b + torch.sign(b) * torch.sqrt(b ** 2 * 4 * a * c) / 2

    t0 = q / a
    t1 = c / q

    return hit, ray_direction * t0 + ray_origin, ray_direction * t1 + ray_origin

hit_mask, p0, p1 = ray_sphere_intersect(ray_origin, ray_direction, sphere_pos, sphere_radius).numpy()
image = hit_mask.astype(float)
print(image)
print(image.shape)
print(image.dtype)
plt.imshow(image)
plt.show()
