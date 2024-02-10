import torch
import copy
import os.path as osp
import libgptb.losses as L
import libgptb.augmentors as A
import torch_geometric.transforms as T
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from libgptb.evaluators import get_split, LREvaluator
from libgptb.models import BootstrapContrast
from torch_geometric.nn import GCNConv
from libgptb.model.abstract_gcl_model import AbstractGCLModel

class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target

class BGRL(AbstractGCLModel):
    def __init__(self, config, data_feature):

        self.nhid = config.get('nhid', 32)
        self.layers = config.get('layers', 3)
        self.device = config.get('device', torch.device('cpu'))
        self.input_dim = data_feature.get('input_dim', 2)
        
        self.pe1 = config.get('drop_edge_rate1', 0.5)
        self.pf1 = config.get('drop_feature_rate1', 0.1)
        self.pe2 = config.get('drop_edge_rate2', 0.5)
        self.pf2 = config.get('drop_feature_rate2', 0.1)

        aug1 = A.Compose([A.EdgeRemoving(pe=self.pe1), A.FeatureMasking(pf=self.pf1)])
        aug2 = A.Compose([A.EdgeRemoving(pe=self.pe2), A.FeatureMasking(pf=self.pf2)])

        super().__init__(config, data_feature)

        self.gconv = GConv(input_dim=self.input_dim, hidden_dim=self.nhid, num_layers=self.layers).to(self.device)
        self.encoder_model = Encoder(encoder=self.gconv, augmentor=(aug1, aug2), hidden_dim=self.nhid).to(self.device)
        self.contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(self.device)
