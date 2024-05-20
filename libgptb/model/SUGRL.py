import torch
from torch import nn
import torch.nn.functional as F
from libgptb.model.abstract_gcl_model import AbstractGCLModel


def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num = len(cfg)
    for i, v in enumerate(cfg):
        out_channels = v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)


class SUGRL(AbstractGCLModel):
    def __init__(self, config, data_feature):
        super(SUGRL, self).__init__(config, data_feature)
        self.nhid = config.get('nhid', 32)
        self.layers = config.get('layers', 3)
        self.dropout = config.get('dropout', 0.2)
        self.device = config.get('device', torch.device('cpu'))
        self.input_dim = data_feature.get('input_dim', 2)
        self.encoder_model = make_mlplayers(self.input_dim, [self.nhid] * self.layers)
        self.act = nn.ReLU()
        self.A = None
        self.sparse = True
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq_a, adj=None):
        if self.A is None:
            self.A = adj
        seq_a = F.dropout(seq_a, self.dropout, training=self.training)

        h_a = self.encoder_model(seq_a)
        h_p_0 = F.dropout(h_a, 0.2, training=self.training)
        if self.sparse:
            h_p = torch.spmm(adj, h_p_0)
        else:
            h_p = torch.mm(adj, h_p_0)
        return h_a, h_p

    def embed(self,  seq_a, adj=None):
        h_a = self.encoder_model(seq_a)
        if self.sparse:
            h_p = torch.spmm(adj, h_a)
        else:
            h_p = torch.mm(adj, h_a)
        return h_a.detach(), h_p.detach()
