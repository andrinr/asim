import Trace as tr

def create_scene_1():
    mat_1 = tr.Material(
        color=[1.0, 34/255., 0.0],
        specular=1.0,
        diffuse=1.0,
        ambient=0.1,
        shininess=30.0,
        refractive_index=1.2,
        reflection=0.0,
        transparency=0.0)

    mat_2 = tr.Material(
        color=[1.0, 1.0, 1.0],
        specular=0.1,
        diffuse=0.3,
        ambient=0.1,
        shininess=20,
        refractive_index=1.2,
        reflection=1.0,
        transparency=0)

    mat_3 = tr.Material(
        color=[0, 204/255, 255/255],
        specular=0.1,
        diffuse=1.0,
        ambient=0.01,
        shininess=2.0,
        refractive_index=1.2,
        reflection=1.0,
        transparency=0)

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
        position=[-1.0, -1.0, 2.0],
        vertices=[
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0]
        ],
        faces=[
            [0, 1, 2]
        ],
        scale=0.4)

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
        material=mat_1))
    
    light = tr.Light(
        position=[-1.5, 2.0, 2], 
        color=[1, 1, 1],
        intensity=1.0)
    
    return meshes, light