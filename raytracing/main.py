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
rays = torch.zeros((frustum_resolution[0], frustum_resolution[1], 6), dtype=torch.float32)
rays[..., :3] = eye_pos.view(1, 1, 3)
rays[..., 3:] = frustum - rays[..., :3]
print(rays)

# sphere xyz pos and radius
sphere = torch.tensor([0, 0, 4, 1], dtype=torch.float32)

# light xyz pos and rgb color
light = torch.tensor([0, 0, 3, 1, 1, 1], dtype=torch.float32)

# check if ray intersects with object
def ray_sphere_intersect(rays, sphere):
    a = torch.tensordot(rays[...,3:], rays[...,3:], dims=([2], [2]))
    print("a", a)
    b = 2 * torch.tensordot(rays[..., :3], rays[..., :3] - sphere[:3], dims=([2], [2]))
    print("b", b)
    c = torch.dot(sphere[:3], sphere[:3]) +\
        torch.tensordot(rays[..., :3], rays[..., :3], dims=([2], [2])) -\
        2 * torch.tensordot(sphere[:3], rays[..., :3]) -\
        sphere[3] ** 2
    print("c", c)

    discriminant = b ** 2 - 4 * a * c

    print(discriminant)

    return discriminant >= 0

image = ray_sphere_intersect(rays, sphere).numpy()
image = image.astype(float)
print(image)
print(image.shape)
print(image.dtype)
plt.imshow(image)
plt.show()
