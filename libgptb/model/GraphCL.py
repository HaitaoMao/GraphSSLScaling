import torch
import os.path as osp
import libgptb.losses as L
import libgptb.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from libgptb.model.abstract_gcl_model import AbstractGCLModel
from libgptb.evaluators import get_split, SVMEvaluator
from libgptb.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2

class GraphCL(AbstractGCLModel):
    def __init__(self, config, data_feature):

        self.device = config.get("device",torch.device('cpu'))
        self.input_dim = data_feature.get('input_dim', 1)
        self.num_layers = config.get('num_layers', 2)
        self.prior = config.get('prior',0)
        self.hidden_dim=config.get("hidden_dim",32)
        self.num_features=data_feature.get("num_features",1)
        super().__init__(config, data_feature)
        aug1 = A.Identity()
        aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                            A.NodeDropping(pn=0.1),
                            A.FeatureMasking(pf=0.1),
                            A.EdgeRemoving(pe=0.1)], 1)
        gconv = GConv(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers).to(self.device)
        self.encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(self.device)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(self.device)