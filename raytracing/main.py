import torch
import matplotlib.pyplot as plt

frustum_resolution = (2000, 2000)

# raytracer
eye_pos = torch.tensor([0, 0, 0], dtype=torch.float32)
frustum = torch.zeros((frustum_resolution[0], frustum_resolution[1], 3), dtype=torch.float32)
frustum[..., 0] = torch.linspace(-1, 1, frustum_resolution[0]).view(-1, 1)
frustum[..., 1] = torch.linspace(-1, 1, frustum_resolution[1]).view(1, -1)
frustum[..., 2] = 1

# rays. xzy pos and xyz dir
ray_origin = eye_pos.view(1, 1, 3).expand(frustum_resolution[0], frustum_resolution[1], 3)
ray_direction = frustum - ray_origin

# light xyz pos and rgb color
light_pos = torch.tensor([2, 1, 0], dtype=torch.float32)
light_color = torch.tensor([1, 1, 1], dtype=torch.float32)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

spheres = []

for i in range(3):
    sphere_pos = torch.rand(3, dtype=torch.float32)
    sphere_pos[2] += 3
    sphere_radius = torch.rand(1, dtype=torch.float32)
    sphere_pos = sphere_pos.to(device)
    sphere_radius = sphere_radius.to(device)
    spheres.append((sphere_pos, sphere_radius))

ray_origin = ray_origin.to(device)
ray_direction = ray_direction.to(device)
light_pos = light_pos.to(device)
light_color = light_color.to(device)

# check if ray intersects with object
def render_sphere(ray_origin, ray_direction, sphere_pos, sphere_radius):
    a = torch.sum(ray_direction ** 2, dim=2)
    ray_to_sphere = torch.subtract(ray_origin, sphere_pos)
    b = 2 * torch.sum(ray_to_sphere * ray_direction, dim=2)
    c = torch.sum(ray_to_sphere ** 2, dim=2) - sphere_radius ** 2
    disc = b ** 2 - 4 * a * c
    alpha = disc >= 0
    alpha = alpha.unsqueeze(2)
    
    q = - (b + torch.sign(b) * torch.sqrt(disc)) / 2

    t0 = torch.div(q, a).unsqueeze(2)
    t1 = torch.div(c, q).unsqueeze(2)
    t = torch.min(t0, t1)

    p = torch.mul(t, ray_direction) + ray_origin

    normals = p - sphere_pos
    normals = torch.div(normals, torch.norm(normals, dim=2, keepdim=True))

    light_dir = light_pos - p
    light_dir = torch.div(light_dir, torch.norm(light_dir, dim=2, keepdim=True))

    dot = torch.sum(normals * light_dir, dim=2, keepdim=True)
    bw = (dot > 0) * alpha

    return bw, alpha

image = torch.zeros((frustum_resolution[0], frustum_resolution[1], 3), dtype=torch.float32)
image = image.to(device)
for sphere_pos, sphere_radius in spheres:
    print(sphere_pos)
    bw, alpha = render_sphere(ray_origin, ray_direction, sphere_pos, sphere_radius)
    image = image + bw

image = image.detach().cpu().numpy()

plt.imshow(image)
plt.show()
