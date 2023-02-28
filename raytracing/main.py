import torch
import numpy as np
import matplotlib.pyplot as plt
from trace import Rays, Light, Sphere, Tracer

frustum_resolution = (500, 500)

rays = Rays.create(torch.tensor([0, 0, 0], dtype=torch.float32), frustum_resolution)
light = Light(
    pos=[3, 1, -2], 
    color=[1, 1, 1])

spheres = []
spheres.append(Sphere(
    pos=[0, 0, 3], 
    radius=[1], 
    color=[1.0, 0.0, 0.0], 
    specular=1.0, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=16.0))

spheres.append(Sphere(
    pos=[0, 0.6, 2], 
    radius=[0.2], 
    color=[0.0, 0.0, 1.0], 
    specular=1.0, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=15.0))

spheres.append(Sphere(
    pos=[0.3, -0.4, 2], 
    radius=[0.5], 
    color=[0.0, 1.0, 1.0], 
    specular=1.0, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=15.0))

# gpu acceleration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
torch.cuda.empty_cache()

tracer = Tracer(spheres, light, ambient=[1.0, 1.0, 0], device=device)

secondary_rays, object_hit = tracer(rays)
tertiary_rays, light_blocked = tracer(secondary_rays)

image = object_hit * ~light_blocked
image = secondary_rays.color
image = image.cpu().numpy()



plt.imshow(image)
plt.show()