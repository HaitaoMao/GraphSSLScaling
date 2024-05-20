from libgptb.augmentors.augmentor import Graph, Augmentor
from libgptb.augmentors.functional import drop_feature


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
    
class FeatureMaskingDGL():
    def __init__(self, pf: float):
        super(FeatureMaskingDGL, self).__init__()
        self.pf = pf

    def augment(self, x):
        x = drop_feature(x, self.pf)
        return x