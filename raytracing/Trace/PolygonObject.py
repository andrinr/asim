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
            divisor = torch.sum(rays.direction * (v0 - rays.origin), axis=1)
            dividend = torch.sum(normal * (rays.direction), axis=1)
            mask = torch.abs(divisor) == 0

            t_0 = torch.where(
                mask, 
                horizon + 1, 
                torch.div(dividend, divisor))
            t_0 = t_0.unsqueeze(1)

            # check plane intersection

            index = torch.argmax(normal)
            mask = torch.tensor([True, True, True])
            mask[index] = False

            v0_2d = v0[mask]
            v1_2d = v1[mask]
            v2_2d = v2[mask]

            point_3d = rays.origin + t_0 * rays.direction
            point_2d = point_3d[:,mask]

            # check if point is inside triangle
            area0 = torch.sum((v0_2d - point_2d) * (v2_2d - point_2d), axis=1)
            area1 = torch.sum((v1_2d - point_2d) * (v0_2d - point_2d), axis=1)
            area2 = torch.sum((v2_2d - point_2d) * (v1_2d - point_2d), axis=1)

            mask = (area0 >= 0) & (area1 >= 0) & (area2 >= 0)
            t_0 = t_0.squeeze()
           
            t_0 = torch.where(
                mask,
                horizon + 1,
                t_0,
            )

            t_0 = t_0.unsqueeze(1)

            t = torch.min(t, t_0)
        
        
        return t

