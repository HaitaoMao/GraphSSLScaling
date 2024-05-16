import torch
import os.path as osp
import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger
import torch_geometric.transforms as T
from libgptb.data.dataset.abstract_dataset import AbstractDataset
from torch_geometric.loader import DataLoader
import importlib

class PyGDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.ratio = self.config.get('ratio', 0)
        self.train_ratio = self.config.get("train_ratio",0.8)
        self.valid_ratio = self.config.get("valid_ratio",0.1)
        self.test_ratio = self.config.get("test_ratio",0.1)
        self.downstream_ratio = self.config.get("downstream_ratio",0.1)
        self.downstream_task = self.config.get('downstream_task','orignal')
        self.task = self.config.get("task","GCL")
        self._load_data()

    def _load_data(self):
        device = torch.device('cuda')
        path = osp.join(os.getcwd(), 'raw_data')

        if self.datasetName in ["Cora", "CiteSeer", "PubMed"]:
            pyg = getattr(importlib.import_module('torch_geometric.datasets'), 'Planetoid')
        if self.datasetName in ["Computers", "Photo"]:
            pyg = getattr(importlib.import_module('torch_geometric.datasets'), 'Amazon')
        if self.datasetName in ["CS", "Physics"]:
            pyg = getattr(importlib.import_module('torch_geometric.datasets'), 'Coauthor')
        if self.datasetName in ["MUTAG", "MCF-7", "MOLT-4","P388","ZINC_full","reddit_threads","github_stargazers"]:   
            pyg = getattr(importlib.import_module('torch_geometric.datasets'), 'TUDataset')
        if self.datasetName in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-ppa", "ogbg-code2"]:
            pyg = getattr(importlib.import_module('ogb.graphproppred'), 'PygGraphPropPredDataset')
        
        if self.task == "SSGCL":
            self.dataset = pyg(root = path, name=self.datasetName)
        else:
            self.dataset = pyg(root = path, name=self.datasetName, transform=T.NormalizeFeatures())
        
        self.data = self.dataset[0].to(device)
        
    
    def get_data(self):
        if self.task == "SSGCL":
            return self.process()
        elif self.task == "GCL":
            return self.data
    
    def process(self):
        
        assert self.train_ratio + self.test_ratio + self.test_ratio <= 1
        if os.path.exists("./split/{}.pt".format(self.datasetName)):
            indices = torch.load("./split/{}.pt".format(self.datasetName))
        else:
            torch.manual_seed(0)
            indices = torch.randperm(len(self.dataset))
            torch.save(indices,"./split/{}.pt".format(self.datasetName))
            

        train_size = int(len(self.dataset) * self.train_ratio)
        valid_size = int(len(self.dataset) * self.valid_ratio)
        test_size = int(len(self.dataset) * self.test_ratio)
        partial_size = min(int(self.ratio*train_size),train_size)
        downstream_size = int(self.downstream_ratio*train_size)
        
        def transform_data(data):
            data.x = data.x.float()
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr.float()
            T.NormalizeFeatures()(data)
            return data

        train_set = [transform_data(self.dataset[i]) for i in indices[:partial_size]]
        valid_set = [transform_data(self.dataset[i])  for i in indices[train_size: train_size + valid_size]]
        test_set = [transform_data(self.dataset[i])  for i in indices[train_size + valid_size:]]
        full_set =  [transform_data(self.dataset[i])  for i in indices]
        if self.downstream_task == 'orignal':
            downstream_train = [transform_data(self.dataset[i])  for i in indices[:downstream_size]]
            downstream_set = downstream_train + valid_set + test_set
        else:
            downstream_set = test_set # may consider agregate valid+test
        return {
        'train': DataLoader(train_set, batch_size=self.batch_size),
        'valid': DataLoader(valid_set, batch_size=self.batch_size),
        'test': DataLoader(test_set, batch_size=self.batch_size),
        'full': DataLoader(full_set, batch_size=self.batch_size),
        'downstream':DataLoader(downstream_set, batch_size=self.batch_size)
        }

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            "input_dim": self.dataset.num_features,
            "num_samples": len(self.dataset)
        }
    
