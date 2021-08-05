
def get_rect(boundaries):
    x = boundaries[0]
    y = boundaries[1]

    width = boundaries[2] - x
    height = boundaries[7] - y

    return [x,y,width, height]

class BoundingBox:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.boundaries = []

    def __init__(self, boundaries):
        self.set_boundaries(boundaries)

    def set_boundaries(self, boundaries):
        self.boundaries = boundaries

        rect = get_rect(boundaries)

        self.x = rect[0]
        self.y = rect[1]
        self.width = rect[2]
        self.height = rect[3]
    def __repr__(self) -> str:
        return '{0}X, {1}Y, {2}W, {3}H'.format(self.x, self.y, self.width, self.height)

