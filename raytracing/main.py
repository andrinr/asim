import torch
import matplotlib.pyplot as plt
from trace import Rays, Light, Sphere, Tracer

frustum_resolution = (1000, 1000)

rays = Rays.create(torch.tensor([0, 0, 0], dtype=torch.float32), frustum_resolution)
light = Light([2, 1, 0])

spheres = []
spheres.append(Sphere([0, 0, 3], [1], [1.0, 0.0, 0.0]))
spheres.append(Sphere([0, 1, 2], [0.2], [0.0, 0.0, 1.0]))

# gpu acceleration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tracer = Tracer(spheres, light, device)

hits_object = tracer(rays)
light_blocked = tracer(rays)

print(hits_object.shape)

image = hits_object & ~light_blocked
image = image.cpu().numpy()

plt.imshow(image)
plt.show()