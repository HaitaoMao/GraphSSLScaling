import torch
import time
from torch import nn, optim
from libgptb.evaluators.base_logits_evaluator import BaseLogitsEvaluator

class Logits_GraphCL(BaseLogitsEvaluator):
    def __init__(self, config, model, logger):
        super().__init__(config, model, logger)
        self.layers = self.config.get("num_layers",2)
        self.nhid = self.config.get("hidden_dim",32)
        self.answering =  torch.nn.Sequential(torch.nn.Linear( self.nhid*self.layers, self.num_class),
                                            torch.nn.Softmax(dim=1)).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.encoder = self.model.encoder_model.encoder
        self.initialize_optimizer()

    def _get_embed(self,data):
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        with torch.no_grad():
            _, g, _, _, _, _ = self.model.encoder_model(data.x, data.edge_index, data.batch)
        return g 

    def _train_epoch(self, train_loader):
        self.model.encoder_model.encoder.train()
        self.answering.train()
        total_loss = 0.0 
        for data in train_loader:  
            self.optimizer.zero_grad() 
            data = data.to(self.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            _, g, _, _, _, _ = self.model.encoder_model(data.x, data.edge_index, data.batch)
            out = self.answering(g)
            loss = self.criterion(out, data.y)  
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader) 