import torch
import os.path as osp
import libgptb.losses as L
import torch_geometric.transforms as T
import libgptb.augmentors as A
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from libgptb.evaluators import get_split, LREvaluator
from libgptb.models import SingleBranchContrast, DualBranchContrast
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.nn.inits import uniform
from libgptb.model.abstract_gcl_model import AbstractGCLModel

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, act in zip(self.layers, self.activations):
            z = conv(z, edge_index)
            z = act(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, g


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)

class Encoder(torch.nn.Module):
    def __init__(self, gcn1, gcn2, mlp1, mlp2, aug1, aug2):
        super(Encoder, self).__init__()
        self.gcn1 = gcn1
        self.gcn2 = gcn2
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.aug1 = aug1
        self.aug2 = aug2

    def forward(self, x, edge_index, batch):
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index)
        z1, g1 = self.gcn1(x1, edge_index1, batch)
        z2, g2 = self.gcn2(x2, edge_index2, batch)
        h1, h2 = [self.mlp1(h) for h in [z1, z2]]
        g1, g2 = [self.mlp2(g) for g in [g1, g2]]
        return h1, h2, g1, g2

class MVGRLg(AbstractGCLModel):
    def __init__(self, config, data_feature):
        
        self.nhid = config.get('nhid', 512)
        self.layers = config.get('layers', 2)
        self.device = config.get('device', torch.device('cpu'))
        self.input_dim = data_feature.get('input_dim', 2)
        super().__init__(config, data_feature)
        aug1 = A.Identity()
        aug2 = A.PPRDiffusion(alpha=0.2,use_cache=False)

        gcn1 = GConv(input_dim=self.input_dim, hidden_dim=self.nhid, num_layers=self.layers ).to(self.device)
        gcn2 = GConv(input_dim=self.input_dim, hidden_dim=self.nhid, num_layers=self.layers ).to(self.device)
        mlp1 = FC(input_dim=self.nhid, output_dim=self.nhid)
        mlp2 = FC(input_dim=self.nhid * self.layers , output_dim=self.nhid)
        self.encoder_model = Encoder(gcn1=gcn1, gcn2=gcn2, mlp1=mlp1, mlp2=mlp2, aug1=aug1, aug2=aug2).to(self.device)
        self.contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(self.device)

