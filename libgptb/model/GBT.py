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
from libgptb.models import  WithinEmbedContrast
from torch_geometric.nn import GCNConv
from libgptb.model.abstract_gcl_model import AbstractGCLModel


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GConv, self).__init__()
        self.act = torch.nn.PReLU()
        self.bn = torch.nn.BatchNorm1d(2 * hidden_dim, momentum=0.01)
        self.conv1 = GCNConv(input_dim, 2 * hidden_dim, cached=False)
        self.conv2 = GCNConv(2 * hidden_dim, hidden_dim, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.conv1(x, edge_index, edge_weight)
        z = self.bn(z)
        z = self.act(z)
        z = self.conv2(z, edge_index, edge_weight)
        return z
    
class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

class GBT(AbstractGCLModel):
    def __init__(self, config, data_feature):
        
        self.nhid = config.get('nhid', 32)
        self.device = config.get('device', torch.device('cpu'))
        self.input_dim = data_feature.get('input_dim', 2)
        
        self.pe1 = config.get('drop_edge_rate1', 0.5)
        self.pf1 = config.get('drop_feature_rate1', 0.1)
        self.pe2 = config.get('drop_edge_rate2', 0.5)
        self.pf2 = config.get('drop_feature_rate2', 0.1)

        aug1 = A.Compose([A.EdgeRemoving(pe=self.pe1), A.FeatureMasking(pf=self.pf1)])
        aug2 = A.Compose([A.EdgeRemoving(pe=self.pe2), A.FeatureMasking(pf=self.pf2)])

        super().__init__(config, data_feature)

        self.gconv = GConv(input_dim=self.input_dim, hidden_dim=self.nhid).to(self.device)
        self.encoder_model = Encoder(encoder=self.gconv, augmentor=(aug1, aug2)).to(self.device)
        self.contrast_model =  WithinEmbedContrast(loss=L.BarlowTwins(), mode='L2L').to(self.device)
