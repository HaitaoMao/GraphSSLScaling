import torch
import time
from torch import nn, optim
class BaseLogitsEvaluator():
    def __init__(self, config, model, logger):
        self.config = config
        self.device = self.config.get('device', torch.device('cpu'))
        self.model = model.to(self.device)
        self._logger = logger
        
        self.model_name = self.config.get('model')
        self.nhid = self.config.get('nhid', 32)
        self.layers = self.config.get('layers', 2)
        self.num_class = self.config.get('num_class',2)
        self.epochs = self.config.get('downstream_epoch',50)
        self.patience = self.config.get('downstream_patience',10)
        self.lr = self.config.get('downstream_lr',0.005)
        self.downstream_model = self.config.get('downstream_model',"model")
        self.criterion = torch.nn.CrossEntropyLoss()


    def initialize_optimizer(self):
        model_param_group = []
        if self.downstream_model == "model":
            model_param_group.append({"params": self.encoder.parameters()})
        model_param_group.append({"params": self.answering.parameters()})
        self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=5e-4)

    def eval(self,test_loader):
        self.encoder.eval()
        self.answering.eval()
        correct = 0
        for data in test_loader: 
            data = data.to(self.device)
            out = self._get_embed(data)
            out = self.answering(out)  
            pred = out.argmax(dim=1)  
            correct += int((pred == data.y).sum())  
        acc = correct / len(test_loader.dataset)
        return acc  

    def train(self, train_loader, valid_loader = None):
        best = 1e9
        cnt_wait = 0
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            loss = self._train_epoch(train_loader)
            valid_info = ''
            if valid_loader != None:
                val_acc, val_loss = self._valid_epoch(valid_loader)
                valid_info += f"valid_loss:{val_loss:.4f} valid_acc{val_acc:.4f}"
            end_time = time.time()
            training_what = "train" if self.downstream_model == 'model' else "header"
            message = 'Downstream-Epoch-[{}/{}] {}_loss: {:.4f}, {:.2f}s wait:[{}/{}] {}'.\
                    format(epoch, self.epochs, training_what, loss, (end_time - start_time),cnt_wait,self.patience,valid_info)
            self._logger.info(message)
            
            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == self.patience:
                    self._logger.info(f'Early stopping at {epoch} eopch!')
                    break

    def _valid_epoch(self, valid_loader):
        self.encoder.eval()
        self.answering.eval()
        total_loss = 0.0 
        correct = 0
        for data in valid_loader:  
            self.optimizer.zero_grad() 
            data = data.to(self.device)
            out = self._get_embed(data)
            out = self.answering(out)
            loss = self.criterion(out, data.y)
            pred = out.argmax(dim=1)  
            correct += int((pred == data.y).sum())    
            total_loss += loss.item()  
        return correct / len(valid_loader.dataset), total_loss / len(valid_loader) 

    def _train_epoch(train_loader):
        pass
    
    def _get_embed(self,data):
        pass
