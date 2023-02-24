import torch
import matplotlib.pyplot as plt

from trace import Rays, Light, Sphere

frustum_resolution = (7000, 7000)

rays = Rays.create(torch.tensor([0, 0, 0], dtype=torch.float32), frustum_resolution)

light = Light(torch.tensor([2, 1, 0], dtype=torch.float32))
sphere = Sphere(torch.tensor([0, 0, 3], dtype=torch.float32), torch.tensor([1], dtype=torch.float32))
sphee2 = Sphere(torch.tensor([0, 1, 3], dtype=torch.float32), torch.tensor([0.5], dtype=torch.float32))

# gpu acceleration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rays.to(device)
light.to(device)
sphere.to(device)

intersection_point, ray_direction_shadow, mask_object = sphere.intersect(rays, light)

# free memory of rays
del rays
rays = Rays(intersection_point, ray_direction_shadow)
_, _, mask_light = sphere.intersect(rays, light)

image = mask_light * mask_object
image = image.cpu().numpy()

plt.imshow(image)
plt.show()