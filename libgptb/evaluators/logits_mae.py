import torch
import time
from torch import nn, optim
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from libgptb.evaluators.base_logits_evaluator import BaseLogitsEvaluator

class Logits_GraphMAE(BaseLogitsEvaluator):
    def __init__(self, config, model, logger):
        super().__init__(config, model, logger)
        self.pooler = self.config.get('pooling','mean')

        self.answering =  torch.nn.Sequential(torch.nn.Linear( self.nhid, self.num_class),
                                            torch.nn.Softmax(dim=1)).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.encoder = self.model.encoder
        self.initialize_optimizer()

    def _get_embed(self,data):
        out = self.encoder(data.x, data.edge_index)
        if self.pooler == "mean":
            out = global_mean_pool(out, data.batch)
        elif self.pooler == "max":
            out = global_max_pool(out, data.batch)
        elif self.pooler == "sum":
            out = global_add_pool(out, data.batch)
        else:
            raise NotImplementedError
        return out

    def _train_epoch(self, train_loader):
        if self.downstream_model == "model":
            self.model.train()
        self.answering.train()
        total_loss = 0.0 
        for data in train_loader:  
            self.optimizer.zero_grad() 
            data = data.to(self.device)
            out = self._get_embed(data)
            out = self.answering(out)
            loss = self.criterion(out, data.y)  
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)  