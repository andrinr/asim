import Trace as tr
import torch
from torch.nn.functional import normalize
import pandas as pd
import numpy as np

class Polygon(tr.Object):
    def __init__(
            self, 
            position : list, 
            scale : float,
            vertices,
            faces
    ):
        super().__init__(position)

        vertices = np.array(vertices)
        faces = np.array(faces)

        self.vertices = torch.tensor(vertices, dtype=torch.float32)
        self.faces = torch.from_numpy(faces)
        self.scale = scale
        self.n_faces = self.faces.shape[0]
        self.n_vertices = self.vertices.shape[0]
        self.normals = torch.zeros((self.n_faces, 3), dtype=torch.float32)

        for i in range(self.n_faces):
            self.normals[i, :] = torch.cross(
                self.vertices[self.faces[i, 1]] - self.vertices[self.faces[i, 0]],
                self.vertices[self.faces[i, 2]] - self.vertices[self.faces[i, 0]]
            )

        self.normals = normalize(self.normals, dim=1)

    def load(position : list, scale : float, vertices_path : str, faces_path : str):
        vertices = pd.read_csv(vertices_path, header=None, sep=' ')
        faces = pd.read_csv(faces_path, header=None, sep=' ')
        vertices = vertices.drop([0], axis=1)
        faces = faces.drop([0], axis=1)

        vertices = vertices.to_numpy()
        faces = faces.to_numpy()

        return Polygon(
            position=position,
            scale=scale,
            vertices=vertices,
            faces=faces - 1)

    def intersect(self, rays : tr.Rays, horizon : float = 2^20, tolerance : float = 1e-6):

        t = torch.full((rays.n * rays.m, 1), horizon, dtype=torch.float32)
        normals = torch.zeros((rays.n * rays.m, 3), dtype=torch.float32)

        for i in range(self.n_faces):
            nm = rays.n * rays.m

            v0 = self.vertices[self.faces[i, 0]] * self.scale + self.position
            v1 = self.vertices[self.faces[i, 1]] * self.scale + self.position
            v2 = self.vertices[self.faces[i, 2]] * self.scale + self.position

            face_normals = self.normals[i, :]
            face_normals = face_normals.unsqueeze(0)
            face_normals = face_normals.repeat(nm, 1)

            dotp = torch.sum(face_normals * rays.direction, axis=1)
            dotp = dotp.unsqueeze(1)

            face_normals = torch.where(dotp > 0, face_normals * -1, face_normals)
            D = torch.sum(-face_normals * (v0), axis=1)
            
            divisor = -torch.sum(face_normals * rays.origin, axis=1) - D
            dividend = torch.sum(face_normals * rays.direction, axis=1)
            mask = torch.abs(dividend) == 0

            t_0 = torch.where(
                mask, 
                horizon + 1, 
                torch.div(divisor, dividend))
            t_0 = t_0.unsqueeze(1)

            # check plane intersection
            index = torch.argmax(torch.abs(face_normals), axis=1, keepdim=True)
            mask = torch.scatter(torch.zeros_like(face_normals), 1, index, 1)
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

            t_0[(t_0 < 0)] = horizon + 1

            t_0 = t_0.unsqueeze(1)

            t = torch.min(t_0, t)

            t_0 = t_0.squeeze()
            t = t.squeeze()
            normals[t == t_0, :] = face_normals[t == t_0, :]
            t = t.unsqueeze(1)
            t_0 = t_0.unsqueeze(1)

        return t, normals