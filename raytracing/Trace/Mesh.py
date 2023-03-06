import Trace as tr

class Mesh():
    def __init__(self, object : tr.Object, material : tr.Material):
        self.object : tr.Object = object
        self.material : tr.Material = material