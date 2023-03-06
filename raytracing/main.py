import torch
import numpy as np
import matplotlib.pyplot as plt
from primitives import Rays, Light, Sphere
from trace import Tracer

frustum_resolution = (800, 800)

spheres = []
spheres.append(Sphere(
    pos=[0, 0, 2], 
    radius=[0.2], 
    color=[1.0, 0.0, 0.0], 
    specular=0.1, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=10.0,
    refractive_index=1.2,
    reflection=1.0,
    transparency=0.))

spheres.append(Sphere(
    pos=[0, 0.3, 2], 
    radius=[0.1], 
    color=[0.0, 0.0, 1.0], 
    specular=0.1, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=15.0,
    refractive_index=1.2,
    reflection=1.0,
    transparency=0.5))

spheres.append(Sphere(
    pos=[0.3, 0.1, 2], 
    radius=[0.1], 
    color=[0.0, 1.0, 1.0], 
    specular=0.1, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=15.0,
    refractive_index=1.2,
    reflection=1.0,
    transparency=0.5))

spheres.append(Sphere(
    pos=[0.2, -0.2, 1.5], 
    radius=[0.2], 
    color=[0.5, 0.5, 1.0], 
    specular=0.7, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=15.0,
    refractive_index=1.1,
    reflection=1.0,
    transparency=0.))

spheres.append(Sphere(
    pos=[-0.1, 0, 1.5], 
    radius=[0.07], 
    color=[1.0, 1.0, 1.0], 
    specular=0.9, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=18.0,
    refractive_index=1.001,
    reflection=1.0,
    transparency=1.0))

spheres.append(Sphere(
    pos=[0.1, 0.1, 1.5], 
    radius=[0.07], 
    color=[1.0, 1.0, 1.0], 
    specular=0.9, 
    diffuse=1.0, 
    ambient=0.1, 
    shininess=18.0,
    refractive_index=0.9,
    reflection=1.0,
    transparency=1.0))

rays = Rays.create(torch.tensor([0, 0, 0], dtype=torch.float32), frustum_resolution, len(spheres))
light = Light(
    pos=[3, 1, -2], 
    color=[1, 1, 1])

# gpu acceleration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
torch.cuda.empty_cache()

tracer = Tracer(spheres, light, ambient=[1.0, 1.0, 1.0], device=device, max_recursion_depth=5)

# 1st pass for blinn phong shading
color = tracer(rays)

image = color
image = image.view(frustum_resolution[0], frustum_resolution[1], -1)
image = image.cpu().numpy()




plt.imshow(image)
plt.show()