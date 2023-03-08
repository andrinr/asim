import Trace as tr

class RaysGrid:
    def __init__(
            self, 
            n : int,
            m : int,
            rays : list):
        self.n = n
        self.m = m
        self.rays = rays
        
    def create(origin, bottom_left : tuple = (-1, -1), top_right : tuple = (1, 1), grid_size : tuple = (16, 16), resolution : tuple = (64, 64), distance = 1):
        n = resolution[0]
        m = resolution[1]

        rays = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                _bottom_left = (bottom_left[0] + i * (top_right[0] - bottom_left[0]) / grid_size[0], bottom_left[1] + j * (top_right[1] - bottom_left[1]) / grid_size[1])
                _top_right = (bottom_left[0] + (i + 1) * (top_right[0] - bottom_left[0]) / grid_size[0], bottom_left[1] + (j + 1) * (top_right[1] - bottom_left[1]) / grid_size[1])

                rays.append(tr.Rays.create(origin, _bottom_left, _top_right, resolution, distance))

        return RaysGrid(grid_size[0], grid_size[1], rays)