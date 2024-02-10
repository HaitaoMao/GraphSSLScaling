import torch
import numpy as np
import torch.nn.functional as F

from libgptb.losses.abstract_losses import Loss
SIGMA = 1e-10

class InfoNCE_RFF(Loss):
    def __init__(self, tau, rff_dim = 4096, mode = 'infonce'):
        super(InfoNCE_RFF, self).__init__()
        self.tau = tau
        self.rff_dim = rff_dim
        self.mode = mode

    def approx_infonce(self, h1, h2):
        z1 = F.normalize(h1, dim=-1)
        z2 = F.normalize(h2, dim=-1)

        pos_score = torch.exp(torch.sum(z1 * z2, dim=1) / self.tau)

        z = torch.cat([z1, z2], dim = 0)

        if self.mode == 'infonce':
            neg_sim = torch.exp(torch.mm(z1, z.t().contiguous()) / self.tau)
            neg_score = neg_sim.sum(1)
       
        elif self.mode == 'rff':
            w = torch.randn(z.size(1), self.rff_dim).to(z.device) / np.sqrt(self.tau)
            rff_out = self.rff_transform(z, w)
            
            rff_1, rff_2 = rff_out.chunk(2, dim = 0)

            neg_sum = torch.sum(rff_out, dim=0, keepdim=True)
            neg_score = np.exp(1 / self.tau) * (torch.sum(rff_1 * neg_sum, dim=1))

        score = - torch.log((pos_score + SIGMA) / neg_score).mean()

        return score


    def rff_transform(self, embedding, w):
        D = w.size(1)
        out = torch.mm(embedding, w)
        d1 = torch.cos(out)
        d2 = torch.sin(out)
        return np.sqrt(1 / D) * torch.cat([d1, d2], dim=1)

    def compute(self, z1, z2) -> torch.FloatTensor:
        
        loss1 = self.approx_infonce(z1, z2)
        loss2 = self.approx_infonce(z2, z1)

        loss = (loss1 + loss2) / 2

        return loss