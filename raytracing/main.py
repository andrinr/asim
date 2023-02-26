import torch
import matplotlib.pyplot as plt
from trace import Rays, Light, Sphere, Tracer

frustum_resolution = (3000, 3000)

rays = Rays.create(torch.tensor([0, 0, 0], dtype=torch.float32), frustum_resolution)
light = Light([2, 1, 0])

spheres = []
spheres.append(Sphere([0, 0, 3], [1], [1.0, 0.0, 0.0]))
spheres.append(Sphere([0, 0.6, 2], [0.2], [0.0, 0.0, 1.0]))
spheres.append(Sphere([0.3, -0.4, 2], [0.5], [0.0, 0.0, 1.0]))

# gpu acceleration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
torch.cuda.empty_cache()

tracer = Tracer(spheres, light, device)

secondary_rays, object_hit = tracer(rays)
tertiary_rays, light_blocked = tracer(secondary_rays)

image = object_hit * ~light_blocked
image = image.cpu().numpy()

plt.imshow(image)
plt.show()