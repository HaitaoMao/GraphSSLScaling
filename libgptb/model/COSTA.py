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
from torch_geometric.nn import GCNConv
from libgptb.model.abstract_gcl_model import AbstractGCLModel

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class COSTAInfoNCE(object):
    def __init__(self, tau):
        super(COSTAInfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob.diag()
        return -loss.mean()

    def __call__(self, anchor, sample) -> torch.FloatTensor:
        loss = self.compute(anchor, sample)
        return loss
    
class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss, mode, intraview_negs=False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None):
        l1 = self.loss(anchor=h1, sample=h2)
        l2 = self.loss(anchor=h2, sample=h1)
        return (l1 + l2) * 0.5


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z
    

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim, ratio, device):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.device = device
        self.hidden_dim = hidden_dim
        self.ratio = ratio

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)


    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)

        k = torch.tensor(int(z.shape[0] * self.ratio))
        p = (1/torch.sqrt(k))*torch.randn(k, z.shape[0]).to(self.device)

        z1 = p @ z1
        z2 = p @ z2 
        h1, h2 = [self.project(x) for x in [z1, z2]]
        return z, h1, h2


class COSTA(AbstractGCLModel):
    def __init__(self, config, data_feature):
        
        self.nhid = config.get('nhid', 256)
        self.pnhid = config.get('pnhid', 256)
        self.layers = config.get('layers', 2)
        self.device = config.get('device', torch.device('cpu'))
        self.input_dim = data_feature.get('input_dim', 2)

        self.pe1 = config.get('drop_edge_rate1', 0.5)
        self.pf1 = config.get('drop_feature_rate1', 0.1)
        self.pe2 = config.get('drop_edge_rate2', 0.5)
        self.pf2 = config.get('drop_feature_rate2', 0.1)

        self.ratio = config.get('ratio', 0.5)
        self.tau = config.get('tau',0.1)

        aug1 = A.Compose([A.EdgeRemoving(pe=self.pe1), A.FeatureMasking(pf=self.pf1)])
        aug2 = A.Compose([A.EdgeRemoving(pe=self.pe2), A.FeatureMasking(pf=self.pf2)])

        super().__init__(config, data_feature)

        self.gconv = GConv(input_dim=self.input_dim, hidden_dim=self.nhid, \
                            activation=torch.nn.ReLU, num_layers=self.layers).to(self.device)
        self.encoder_model = Encoder(encoder=self.gconv,augmentor=(aug1, aug2),\
                                      hidden_dim=self.nhid, proj_dim = self.pnhid,\
                                        ratio =self.ratio, device=self.device).to(self.device)
        self.contrast_model = DualBranchContrast(loss=COSTAInfoNCE(\
            tau=self.tau), mode='L2L', intraview_negs=True).to(self.device)

