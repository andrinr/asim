import Trace as tr

class Triangle(tr.Object):
    def __init__(
            self, 
            position : list, 
            v0 : list, 
            v1 : list, 
            v2 : list
    ):
        super().__init__(position)
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def intersect(self, rays : tr.Rays):
        pass