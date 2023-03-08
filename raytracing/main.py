import torch
import matplotlib.pyplot as plt
import Trace as tr
from scene1 import create_scene_1

frustum_resolution = (1024, 1024)

meshes, light = create_scene_1()

# rays = tr.Rays.create(
#     torch.tensor([0, 0, 0], dtype=torch.float32), 
#     (-1, -1),
#     (1, 1),
#     (1024, 1024), 1.0)

rays_grid = tr.RaysGrid.create(
    origin = torch.tensor([0, 0, 0], dtype=torch.float32),
    bottom_left=(-1, -1),
    top_right=(1, 1),
    grid_size=(8, 8),
    resolution=(128, 128),
    distance=1.0)

tracer = tr.Tracer(meshes=meshes, light=light, ambient=[1.0, 1.0, 1.0], max_recursion_depth=3)
image = tracer(rays_grid)

image = image.view(frustum_resolution[0], frustum_resolution[1], -1)
image = image.cpu().numpy()

plt.imshow(image)
plt.show()