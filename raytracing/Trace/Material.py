import torch 

class Material:
    def __init__(
            self, 
            color : list, 
            diffuse : float,
            ambient : float, 
            specular : float,
            shininess : float,
            refractive_index : float,
            transparency : float,
            reflection : float):
        self.color = torch.tensor(color, dtype=torch.float32)
        self.diffuse = diffuse
        self.ambient = ambient
        self.specular = specular
        self.shininess = shininess
        self.refractive_index = refractive_index
        self.transparency = transparency
        self.reflection = reflection