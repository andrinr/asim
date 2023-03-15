import Trace as tr

def create_scene_2():
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
        shininess=20,
        refractive_index=1.2,
        reflection=1.0,
        transparency=0)

    sphere_2 = tr.Sphere(
        position=[500.0, 0, 2.0],
        radius=499.5)

    meshes : list[tr.Mesh] = []

    meshes.append(tr.Mesh(
        object=sphere_2,
        material=mat_2))

    light = tr.Light(
        position=[-1.5, 2.0, 0.5], 
        color=[1, 1, 1],
        intensity=1.0)
    
    bunny = tr.Polygon.load(
        [0, 0, 2.0],
        0.2,
        "raytracing/teapot/vertices.txt",
        "raytracing/teapot/faces.txt",
    )

    meshes.append(tr.Mesh(
        object=bunny,
        material=mat_1))
    
    return meshes, light