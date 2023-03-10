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

    def intersect(self, rays : tr.Rays, horizon : float = 2^20, tolerance : float = 1e-6):

        nm = rays.n * rays.m
        t = torch.full((nm, 1), horizon)

        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]

            print(v0)
            print(v1)
            print(v2)

            e1 = v1 - v0
            e2 = v2 - v0

            normal = torch.cross(e1, e2)
            denom = torch.sum(rays.direction * (v0 - rays.origin), axis=1)
            mask = torch.abs(denom) == 0

            print(normal)

            t_0 = torch.where(
                mask, 
                torch.full((nm, 1), horizon + 1), 
                torch.div(torch.sum(normal * (rays.direction), axis=1), denom))
            
            # check plane intersection

            min_index = torch.argmin(normal)
            print(min_index)

            v0_2d = torch.cat((v0[(min_index + 1) % 3], v0[(min_index + 2) % 3]), axis=1)
            v1_2d = torch.cat((v1[(min_index + 1) % 3], v1[(min_index + 2) % 3]), axis=1)
            v2_2d = torch.cat((v2[(min_index + 1) % 3], v2[(min_index + 2) % 3]), axis=1)

            point_3d = rays.origin + t_0 * rays.direction
            point_2d = torch.cat((point_3d[:,(min_index + 1) % 3], point_3d[:,(min_index + 2) % 3]), axis=1)

            # check if point is inside triangle
            area0 = torch.sum((v0_2d - point_2d) * (v2_2d - point_2d), axis=1)
            area1 = torch.sum((v1_2d - point_2d) * (v0_2d - point_2d), axis=1)
            area2 = torch.sum((v2_2d - point_2d) * (v1_2d - point_2d), axis=1)

            t0 = torch.where(
                (area0 >= 0) & (area1 >= 0) & (area2 >= 0),
                t_0,
                torch.full((nm, 1), horizon + 1)
            )

            t = torch.min(t, t0)
        
        return t

