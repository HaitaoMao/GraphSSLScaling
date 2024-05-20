import torch
import numpy as np
import torch.nn.functional as F

from libgptb.losses.abstract_losses import Loss


class CCALoss(Loss):
    def __init__(self, lambd):
        super(CCALoss, self).__init__()
        self.lambd = lambd

    def compute(self, z1, z2) -> torch.FloatTensor:
        c = torch.mm(z1.T, z2) / z1.size(0)
        c1 = torch.mm(z1.T, z1) / z1.size(0)
        c2 = torch.mm(z2.T, z2) / z1.size(0)

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.size(0)), dtype=torch.float32).to(z1.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + self.lambd * (loss_dec1 + loss_dec2)

        return loss