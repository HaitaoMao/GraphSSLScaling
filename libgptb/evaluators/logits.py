import torch
import time
from torch import nn, optim
class Logits():
    def __init__(self, config, model, logger):
        self.config = config
        self.device = self.config.get('device', torch.device('cpu'))
        self.model = model.to(self.device)
        self._logger = logger
        
        self.nhid = config.get('nhid', 32)
        self.layers = config.get('layers', 2)
        self.num_class = self.config.get('num_class',2)
        self.epochs = self.config.get('downstream_epoch',20)
        self.patience = self.config.get('downstream_patience',10)
        self.lr = self.config.get('downstream_lr',0.005)

        self.answering =  torch.nn.Sequential(torch.nn.Linear(self.nhid*self.layers, self.num_class),
                                            torch.nn.Softmax(dim=1)).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.initialize_optimizer()

    def initialize_optimizer(self):
        model_param_group = []
        model_param_group.append({"params": self.model.encoder_model.encoder.parameters()})
        model_param_group.append({"params": self.answering.parameters()})
        self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=5e-4)

    def eval(self,test_loader):
        self.model.encoder_model.eval()
        self.answering.eval()
        correct = 0
        for data in test_loader: 
            data = data.to(self.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device) 
            z, g = self.model.encoder_model(data.x, data.edge_index, data.batch)
            out = self.answering(g)  
            pred = out.argmax(dim=1)  
            correct += int((pred == data.y).sum())  
        acc = correct / len(test_loader.dataset)
        return acc  

    def train(self, train_loader):
        best = 1e9
        cnt_wait = 0
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            loss = self._train_epoch(train_loader)
            end_time = time.time()
            message = 'Downstream-Epoch-[{}/{}] train_loss: {:.4f}, {:.2f}s wait:[{}/{}] '.\
                    format(epoch, self.epochs, loss, (end_time - start_time),cnt_wait,self.patience)
            self._logger.info(message)
            
            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == self.patience:
                    self._logger.info(f'Early stopping at {epoch} eopch!')
                    break

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
            z, g = self.model.encoder_model(data.x, data.edge_index, data.batch)
            out = self.answering(g)
            loss = self.criterion(out, data.y)  
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)  