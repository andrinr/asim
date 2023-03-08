import torch
import matplotlib.pyplot as plt
import Trace as tr

frustum_resolution = (800, 800)

mat_1 = tr.Material(
    color=[1.0, 34/255., 0.0],
    specular=1.0,
    diffuse=1.0,
    ambient=0.1,
    shininess=30.0,
    refractive_index=1.2,
    reflection=1.0,
    transparency=0.0)

mat_2 = tr.Material(
    color=[1.0, 1.0, 1.0],
    specular=0.1,
    diffuse=0.3,
    ambient=0.1,
    shininess=2.0,
    refractive_index=1.2,
    reflection=1.0,
    transparency=0.1)

mat_3 = tr.Material(
    color=[0, 204/255, 255/255],
    specular=0.1,
    diffuse=1.0,
    ambient=0.01,
    shininess=2.0,
    refractive_index=1.2,
    reflection=1.0,
    transparency=0.1)

sphere_1 = tr.Sphere(
    position=[0, 0, 2.0],
    radius=0.4)

sphere_2 = tr.Sphere(
    position=[500.0, 0, 2.0],
    radius=499.5)

sphere_3 = tr.Sphere(
    position=[0, 0.7, 2.0],
    radius=0.3)

polygon_object = tr.PolygonObject(
    position=[0, 0, 2.0],
    vertices=[
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0]
    ],
    faces=[
        [0, 1, 2]
    ])

meshes : list[tr.Mesh] = []
meshes.append(tr.Mesh(
    object=sphere_1,
    material=mat_1))

meshes.append(tr.Mesh(
    object=sphere_2,
    material=mat_2))

meshes.append(tr.Mesh(
    object=sphere_3,
    material=mat_3))

meshes.append(tr.Mesh(
    object=polygon_object,
    material=mat_3))

rays = tr.Rays.create(torch.tensor([0, 0, 0], dtype=torch.float32), frustum_resolution, 1.0)
light = tr.Light(
    position=[-1.5, 2.0, 2], 
    color=[1, 1, 1],
    intensity=1.0)

tracer = tr.Tracer(meshes=meshes, light=light, ambient=[1.0, 1.0, 1.0], max_recursion_depth=3)
image = tracer(rays)

image = image.view(frustum_resolution[0], frustum_resolution[1], -1)
image = image.cpu().numpy()

plt.imshow(image)
plt.show()