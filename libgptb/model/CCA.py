import torch
import os.path as osp
import libgptb.losses as L

from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from libgptb.evaluators import get_split, LREvaluator
from libgptb.models import CCAContrast
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

    def forward(self, graph1, graph2, feat1, feat2):
        h1 = self.encoder(graph1, feat1)
        h2 = self.encoder(graph2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        return z1, z2
    
class CCA(AbstractGCLModel):
    def __init__(self, config, data_feature):
        self.nhid = config.get('nhid', 32)
        self.layers = config.get('layers', 3)
        self.device = config.get('device', torch.device('cpu'))
        self.lambd = config.get('lambd', 1e-3)
        self.input_dim = data_feature.get('input_dim', 2)
        super().__init__(config, data_feature)

        self.gconv = GCN(in_dim=self.input_dim, hid_dim=self.nhid, num_layers=self.layers).to(self.device)
        self.encoder_model = Encoder(encoder=self.gconv, hidden_dim=self.nhid).to(self.device)
        self.contrast_model = CCAContrast(loss=L.CCALoss(self.lambd)).to(self.device)
