from libgptb.augmentors.augmentor import Graph, Augmentor
from libgptb.augmentors.functional import dropout_adj
import torch
import dgl
import numpy as np

class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class EdgeRemovingDGL():
    def __init__(self, pe: float):
        super(EdgeRemovingDGL, self).__init__()
        self.pe = pe

    def augment(self, graph: Graph) -> Graph:
        
        num_edges = graph.num_edges()
        mask_rates = torch.FloatTensor(np.ones(num_edges) * self.pe)
        masks = torch.bernoulli(1 - mask_rates)
        edge_mask = masks.nonzero().squeeze(1)
     
        num_nodes = graph.num_nodes()
        ng = dgl.graph([])
        ng.add_nodes(num_nodes)
        src = graph.edges()[0]
        dst = graph.edges()[1]

        nsrc = src[edge_mask]
        ndst = dst[edge_mask]
        ng.add_edges(nsrc, ndst)
        return ng
