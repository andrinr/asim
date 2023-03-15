import Trace as tr
import torch

class Polygon(tr.Object):
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
        self.normal = torch.cross(
            self.vertices[self.faces[1]] - self.vertices[self.faces[0]],
            self.vertices[self.faces[2]] - self.vertices[self.faces[0]]
        )
        self.normal = self.normal / torch.norm(self.normal)


    def intersect(self, rays : tr.Rays, horizon : float = 2^20, tolerance : float = 1e-6):

        nm = rays.n * rays.m
        t = torch.full((nm, 1), horizon)

        v0 = self.vertices[self.faces[0]] * self.scale + self.position
        v1 = self.vertices[self.faces[1]] * self.scale + self.position
        v2 = self.vertices[self.faces[2]] * self.scale + self.position

        normals = self.normal
        normals = normals.unsqueeze(0)
        normals = normals.repeat(nm, 1)

        dotp = torch.sum(normals * rays.direction, axis=1)
        dotp = dotp.unsqueeze(1)

        normals = torch.where(dotp > 0, normals * -1, normals)
        D = torch.sum(-normals * (v0), axis=1)
        
        divisor = -torch.sum(normals * rays.origin, axis=1) - D
        dividend = torch.sum(normals * rays.direction, axis=1)
        mask = torch.abs(divisor) == 0

        t_0 = torch.where(
            mask, 
            horizon + 1, 
            torch.div(dividend, divisor))
        t_0 = t_0.unsqueeze(1)

        # check plane intersection
        index = torch.argmax(torch.abs(normals), axis=1, keepdim=True)
        mask = torch.scatter(torch.zeros_like(normals), 1, index, 1)
        mask = mask.type(torch.BoolTensor)

        v0 = v0.unsqueeze(0)
        v1 = v1.unsqueeze(0)
        v2 = v2.unsqueeze(0)

        v0 = v0.repeat(nm, 1)
        v1 = v1.repeat(nm, 1)
        v2 = v2.repeat(nm, 1)

        v0[mask] = 0
        v1[mask] = 0
        v2[mask] = 0

        point = rays.origin + t_0 * rays.direction
        point[mask] = 0

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

        return t_0, normals