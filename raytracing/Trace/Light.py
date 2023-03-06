import Trace as tr

class Light(tr.Object):
    def __init__(
            self,
            position : list,
            color : list,
            intensity : float
        ):
        super().__init__(position)
        self.color = color
        self.intensity = intensity