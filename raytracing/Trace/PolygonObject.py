import Trace as tr
import torch

class PolygonObject(tr.Object):
    def __init__(
            self, 
            position : list, 
            vertices : list,
            faces : list
    ):
        super().__init__(position)
        self.vertices = []
        self.vertices = torch.tensor(vertices, dtype=torch.float32)
        self.faces = torch.tensor(faces, dtype=torch.int32)

    def intersect(self, rays : tr.Rays, horizon : float = 2^20, tolerance : float = 1e-5):

        nm = rays.n * rays.m
        t = torch.full((nm, 1), float('inf'))

        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]

            e1 = v1 - v0
            e2 = v2 - v0

            normal = torch.cross(e1, e2)
            # denom = torch.sum(rays.direction * normal, axis=1)
            # mask = torch.abs(denom) < tolerance
            # t_0 = torch.where(mask, torch.full((nm, 1), float('inf')), t)
            # t = torch.where(mask & torch.sum(rays.origin * normal, axis=1) / denom

            # # check plane intersection

            # t = torch.matmul(normal, v0) - torch.matmul(normal, rays.origin)
            # # check if ray hits inside the triangle
            # max_index = torch.argmax(torch.abs(normal))

            # #project to 2D
            # e1 = torch.cat((e1[0:max_index], e1[max_index+1:]))
            # e2 = torch.cat((e2[0:max_index], e2[max_index+1:]))

        
        return t

