import torch
import numpy as np
import matplotlib.pyplot as plt
from primitives import Rays, Light, Sphere
from trace import Tracer

frustum_resolution = (800, 800)

rays = Rays.create(torch.tensor([0, 0, 0], dtype=torch.float32), frustum_resolution)
light = Light(
    pos=[3, 1, -2], 
    color=[1, 1, 1])

spheres = []
spheres.append(Sphere(
    pos=[0, 0, 3], 
    radius=[1], 
    color=[1.0, 0.0, 0.0], 
    specular=0.1, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=10.0,
    refractive_index=1.5,
    reflection=0.9,
    transparency=.0))

spheres.append(Sphere(
    pos=[0, 0.6, 2], 
    radius=[0.2], 
    color=[0.0, 0.0, 1.0], 
    specular=0.1, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=15.0,
    refractive_index=1.2,
    reflection=0.9,
    transparency=0))

spheres.append(Sphere(
    pos=[0.3, -0.4, 2], 
    radius=[0.5], 
    color=[0.0, 1.0, 1.0], 
    specular=0.7, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=15.0,
    refractive_index=0.9,
    reflection=0.,
    transparency=0))

# gpu acceleration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
torch.cuda.empty_cache()

tracer = Tracer(spheres, light, ambient=[1.0, 1.0, 1.0], device=device, max_recursion_depth=2)

# 1st pass for blinn phong shading
color = tracer(rays)

image = color
image = image.view(frustum_resolution[0], frustum_resolution[1], -1)
image = image.cpu().numpy()




plt.imshow(image)
plt.show()