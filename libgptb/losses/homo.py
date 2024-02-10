import torch
import numpy as np
import torch.nn.functional as F
import faiss
from libgptb.losses.abstract_losses import Loss


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class InfoNCESP(Loss):
    """
    InfoNCE loss for single positive.
    """
    def __init__(self, tau):
        super(InfoNCESP, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        f = lambda x: torch.exp(x / self.tau)
        sim = f(_similarity(anchor, sample))  # anchor x sample
        assert sim.size() == pos_mask.size()  # sanity check

        neg_mask = 1 - pos_mask
        pos = (sim * pos_mask).sum(dim=1)
        neg = (sim * neg_mask).sum(dim=1)

        loss = pos / (pos + neg)
        loss = -torch.log(loss)

        return loss.mean()


class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()
def homo_loss(x, edge_index, nclusters, niter, sigma):
    kmeans = faiss.Kmeans(x.shape[1], nclusters, niter=niter) 
    kmeans.train(x.cpu().detach().numpy())
    centroids = torch.FloatTensor(kmeans.centroids).to(x.device)
    logits = []
    for c in centroids:
        logits.append((-torch.square(x - c).sum(1)/sigma).view(-1, 1))
    logits = torch.cat(logits, axis=1)
    probs = F.softmax(logits, dim=1)
    loss = F.mse_loss(probs[edge_index[0]], probs[edge_index[1]])
    return loss, probs
def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())
def semi_loss(z1, adj1, z2, adj2, confmatrix, tau):
        f = lambda x: torch.exp(x / tau)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))
        # if mean:
        pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1) / (adj1.sum(1)+0.01) 
        # else:
            # pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1)
        neg = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() - (refl_sim * adj1).sum(1) - (between_sim * adj2).sum(1)
        loss = -torch.log(pos / (pos + neg))

        return loss
def floss(h1, graph1, h2, graph2, confmatrix, tau):
    l1 = semi_loss(h1, graph1, h2, graph2, confmatrix, tau)
    l2 = semi_loss(h2, graph2, h1, graph1, confmatrix, tau)

    ret = (l1 + l2) * 0.5
    ret = ret.mean()
    return ret

class HomoLoss():
    def __init__(self, nclusters, niter, sigma,  alpha, tau,  device):
        super(HomoLoss, self).__init__()
        self.nclusters = nclusters
        self.niter = niter
        self.sigma = sigma
        self.alpha = alpha
        self.device = device
        self.tau = tau


    def compute(self, z1, z2, z, graph, graph1, graph2, N) -> torch.FloatTensor:
        adj1 = torch.zeros(N, N, dtype=torch.int).to(self.device)
        adj1[graph1.remove_self_loop().edges()] = 1
        adj2 = torch.zeros(N, N, dtype=torch.int).to(self.device)
        adj2[graph2.remove_self_loop().edges()] = 1

        homoloss, homoprobs = homo_loss(z, graph.remove_self_loop().add_self_loop().edges(), self.nclusters, self.niter, self.sigma)
        confmatrix = sim(homoprobs, homoprobs) # saliency matrix
        loss = floss(z1, adj1, z2, adj2, confmatrix, self.tau)
        loss = loss + self.alpha * homoloss

        return loss