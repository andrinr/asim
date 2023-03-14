import Trace as tr
import torch

class PolygonObject(tr.Object):
    def __init__(
            self, 
            position : list, 
            scale : float,
            vertices : list,
            faces : list
    ):
        super().__init__(position)
        self.vertices = []
        self.vertices = torch.tensor(vertices, dtype=torch.float32)
        self.faces = torch.tensor(faces, dtype=torch.int32)
        self.scale = scale

    def intersect(self, rays : tr.Rays, horizon : float = 2^20, tolerance : float = 1e-6):

        nm = rays.n * rays.m
        t = torch.full((nm, 1), horizon)

        for face in self.faces:
            v0 = self.vertices[face[0]] * self.scale + self.position
            v1 = self.vertices[face[1]] * self.scale + self.position
            v2 = self.vertices[face[2]] * self.scale + self.position

            normal = torch.cross((v1 - v0), (v2 - v0))
            normal = normal / torch.norm(normal)
            print((normal * rays.direction).shape)
            normal = torch.where(torch.sum(normal * rays.direction, axis=1) > 0, normal * -1, normal)
            print(normal)
            D = torch.sum(-normal * (v0), axis=1)
            
            divisor = -torch.sum(normal * rays.origin, axis=1) - D
            dividend = torch.sum(normal * rays.direction, axis=1)
            mask = torch.abs(divisor) == 0

            t_0 = torch.where(
                mask, 
                horizon + 1, 
                torch.div(dividend, divisor))
            t_0 = t_0.unsqueeze(1)

            # check plane intersection

            index = torch.argmax(normal)
            normal = torch.abs(normal)

            v0[index] = 0
            v1[index] = 0
            v2[index] = 0

            point = rays.origin + t_0 * rays.direction
            point[:, index] = 0

            # check if point is inside triangle
            area0 = torch.sum(torch.cross((v0 - point), (v2 - point)), axis=1)
            area1 = torch.sum(torch.cross((v1 - point), (v0 - point)), axis=1)
            area2 = torch.sum(torch.cross((v2 - point), (v1 - point)), axis=1)

            mask = (area0 >= 0) & (area1 >= 0) & (area2 >= 0)
            mask = mask | (area0 <= 0) & (area1 <= 0) & (area2 <= 0)
            t_0 = t_0.squeeze()
           
            t_0 = torch.where(
                mask,
                t_0, 
                horizon + 1
            )

            t_0 = t_0.unsqueeze(1)

            t = torch.min(t, t_0)
        
        return t