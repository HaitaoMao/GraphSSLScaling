import torch
import os.path as osp
import libgptb.losses as L

from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from libgptb.evaluators import get_split, LREvaluator
from libgptb.models import HomoContrast
import dgl
from dgl.nn import GraphConv
from libgptb.model.abstract_gcl_model import AbstractGCLModel

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers):
        super().__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if num_layers > 1:
            for i in range(num_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))

    def forward(self, graph, x):

        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)

        return x


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder

    def forward(self, graph1, graph2, feat1, feat2, graph, feat):
        z1 = self.encoder(graph1, feat1)
        z2 = self.encoder(graph2, feat2)
        z = self.encoder(graph, feat)
        return z1, z2, z, graph1, graph2, graph.number_of_nodes()
    
class HomoGCL(AbstractGCLModel):
    def __init__(self, config, data_feature):
        self.nhid = config.get('nhid', 32)
        self.layers = config.get('layers', 3)
        self.device = config.get('device', torch.device('cpu'))
        self.alpha = config.get('alpha', 1)
        self.nclusters = config.get('nclusters',5)
        self.niter = config.get('niter',20)
        self.sigma = config.get('sigma',1e-3)
        self.alpha = config.get('alpha',1)
        self.tau = config.get('tau',0.5)
        self.input_dim = data_feature.get('input_dim', 2)
        super().__init__(config, data_feature)

        self.gconv = GCN(in_dim=self.input_dim, hid_dim=self.nhid, num_layers=self.layers).to(self.device)
        self.encoder_model = Encoder(encoder=self.gconv, hidden_dim=self.nhid).to(self.device)
        self.contrast_model = HomoContrast(loss=L.HomoLoss(self.nclusters, self.niter, self.sigma,  self.alpha, self.tau, self.device)).to(self.device)
