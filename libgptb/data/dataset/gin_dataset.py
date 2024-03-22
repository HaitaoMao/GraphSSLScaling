import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
#from gnn import GNN
from libgptb.data.dataset.abstract_dataset import AbstractDataset
from tqdm import tqdm
import argparse
import time
import numpy as np
import sys
import random
### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator



class GINDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self._load_data()

    def _load_data(self):

        dataset = PygGraphPropPredDataset(name = self.datasetName)

        if self.config.get("feature") == 'full':
            pass 
        elif self.config.get("feature") == 'simple':
            print('using simple feature')
            # only retain the top two node/edge features
            dataset.x = dataset.x[:,:2]
            dataset.edge_attr = dataset.edge_attr[:,:2]

        self.split_idx = dataset.get_idx_split()
        
        self.data = dataset

    def get_data(self):
        train_index = self.split_idx["train"]
        train_loader = DataLoader(self.data[train_index[:int(self.config.get("training_ratio",0.2)*len(train_index))]], batch_size=self.config.get("batch_size",200), shuffle=True, num_workers = self.config.get("num_worders",5))
        valid_loader = DataLoader(self.data[self.split_idx["valid"]], batch_size=self.config.get("batch_size",200), shuffle=False, num_workers = self.config.get("num_worders",5))
        test_loader = DataLoader(self.data[self.split_idx["test"]], batch_size=self.config.get("batch_size",200), shuffle=False, num_workers = self.config.get("num_worders",5))
        return {'train':train_loader,'valid':valid_loader,'test':test_loader}
    
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"num_tasks": self.data.num_tasks,"task_type":self.data.task_type,"eval_metric":self.data.eval_metric}