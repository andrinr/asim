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

            edge1 = v1 - v0
            edge2 = v2 - v0

            h = torch.cross(rays.direction, edge2)
            a = torch.sum(edge1 * h, dim=1)
            mask = a > tolerance
            mask = mask.unsqueeze(1)

            s = rays.origin - v0
            u = torch.div(torch.sum(s * h, dim=1), a).unsqueeze(1)
            u[(u < 0) | ~mask] = horizon + 1

            q = torch.cross(s, edge1)
            v = torch.div(torch.sum(rays.direction * q, dim=1), a).unsqueeze(1)
            v[(v < 0) | (u + v > 1) | ~mask] = horizon + 1

            t = torch.div(torch.sum(edge2 * q, dim=1), a).unsqueeze(1)
            t[(t < 0) | ~mask] = horizon + 1

            return t
        pass

