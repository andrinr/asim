import torch
import matplotlib.pyplot as plt
import Trace as tr
from scene1 import create_scene_1
from scene2 import create_scene_2

# meshes, light = create_scene_1()
meshes, light = create_scene_2()

rays_grid = tr.RaysGrid.create(
    origin = torch.tensor([0, 0, 0], dtype=torch.float32),
    bottom_left=(-1, -1),
    top_right=(1, 1),
    grid_size=(4, 4),
    resolution=(64, 64),
    distance=1.0)

tracer = tr.Tracer(meshes=meshes, light=light, ambient=[1.0, 1.0, 1.0], max_recursion_depth=4)
image = tracer(rays_grid)
image = image.cpu().numpy()

plt.imshow(image)
plt.show()