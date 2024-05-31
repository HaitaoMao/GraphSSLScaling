import torch
import os.path as osp
import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger
from sklearn.utils import column_or_1d
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from libgptb.data.dataset.abstract_dataset import AbstractDataset
import importlib
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree
from collections import namedtuple, Counter
import torch.nn.functional as F

class MAEDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.train_ratio = self.config.get("train_ratio",0.8)
        self.valid_ratio = self.config.get("valid_ratio",0.1)
        self.test_ratio = self.config.get("test_ratio",0.1)
        self.downstream_ratio = self.config.get("downstream_ratio",0.1)
        self.downstream_task = self.config.get('downstream_task','original')
        
        self._load_data()
        self.ratio = self.config.get('ratio', 0)
        config['num_feature']={
            "input_dim": max(self.num_features, 1)
        }
        config['num_class']=self.num_classes
        
        
    def _load_data(self):
        device = torch.device('cuda')
        path = osp.join(os.getcwd(), 'raw_data')
        deg4feat=False
        if self.datasetName in ["REDDIT-BINARY" "PROTEINS"]:
            deg4feat=True
        # original paper choices of datasets.
        #if self.datasetName in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        if self.datasetName in ["MUTAG", "MCF-7", "MOLT-4","P388","ZINC_full","reddit_threads","github_stargazers"]:   
            tu_dataset = getattr(importlib.import_module('torch_geometric.datasets'), 'TUDataset')
            self.dataset = tu_dataset(root=path, name=self.datasetName)
        if self.datasetName in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-ppa"]:
            tu_dataset = getattr(importlib.import_module('ogb.graphproppred'), 'PygGraphPropPredDataset')
            self.dataset = tu_dataset(root=path, name=self.datasetName)
        
        self.dataset = list(self.dataset)
        graph = self.dataset[0]
        print(graph)
        if graph.x == None:
            if graph.y and not deg4feat:
                print("Use node label as node features")
                feature_dim = 0
                for g in self.dataset:
                    feature_dim = max(feature_dim, int(g.y.max().item()))
            
                feature_dim += 1
                for i, g in enumerate(self.dataset):
                    node_label = g.y.view(-1)
                    feat = F.one_hot(node_label, num_classes=int(feature_dim)).float()
                    self.dataset[i].x = feat
            else:
                print("Using degree as node features")
                feature_dim = 0
                degrees = []
                for g in self.dataset:
                    feature_dim = max(feature_dim, degree(g.edge_index[0]).max().item())
                    degrees.extend(degree(g.edge_index[0]).tolist())
                MAX_DEGREES = 400

                oversize = 0
                for d, n in Counter(degrees).items():
                    if d > MAX_DEGREES:
                        oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
                feature_dim = min(feature_dim, MAX_DEGREES)

                feature_dim += 1
                for i, g in enumerate(self.dataset):
                    degrees = degree(g.edge_index[0])
                    degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                    degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])
                    feat = F.one_hot(degrees.to(torch.long), num_classes=int(feature_dim)).float()
                    g.x = feat
                    self.dataset[i] = g

        else:
            print("******** Use `attr` as node features ********")
        
        feature_dim = int(graph.num_features)
        
        if self.datasetName in ["ogbg-molpcba"]:
            labels = torch.stack([torch.nan_to_num(x.y, nan=0.0) for x in self.dataset])
        elif self.datasetName in ["ogbg-code2"]:
            for i, g in enumerate(self.dataset):
                print(g.y)
            valid_labels = [x.y for x in self.dataset if isinstance(x.y, (torch.Tensor, list, tuple, int, float))]
            labels = torch.stack([torch.tensor(y) if not isinstance(y, torch.Tensor) else y for y in valid_labels])
        else:
            labels = torch.tensor([x.y for x in self.dataset]) 
        if self.datasetName in ["ogbg-molhiv", "ogbg-molpcba",  "ogbg-code2"]:
            for i, g in enumerate(self.dataset):
                g.x = g.x.float()
        
        num_classes = torch.max(labels).item() + 1
        for i, g in enumerate(self.dataset):
            self.dataset[i].edge_index = remove_self_loops(self.dataset[i].edge_index)[0]
            self.dataset[i].edge_index = add_self_loops(self.dataset[i].edge_index)[0]
        
        self.num_features=feature_dim
        self.num_classes=num_classes
        
        
    def get_data(self):
        
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

        train_set = [self.dataset[i] for i in indices[:partial_size]]
        valid_set = [self.dataset[i] for i in indices[train_size: train_size + valid_size]]
        test_set = [self.dataset[i] for i in indices[train_size + valid_size:]]
        full_set =  [self.dataset[i]  for i in indices]
        downstream_set = [self.dataset[i] for i in indices[:downstream_size]]

        train_loader = DataLoader(train_set, batch_size=self.batch_size, pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size, pin_memory=True)
        test_loader  = DataLoader(test_set, batch_size=self.batch_size, pin_memory=True)
        full_loader  = DataLoader(full_set, batch_size=self.batch_size, pin_memory=True)
        downstream_train = DataLoader(downstream_set, batch_size=self.batch_size, pin_memory=True)
        down_loader  = {}
        down_loader  = {}
        down_loader['full'] = full_loader
        down_loader['test'] = test_loader
        down_loader['downstream_train'] = downstream_train
        return {
        'train': train_loader,
        'valid': valid_loader,
        'test' : test_loader,
        'full' : full_loader,
        'downstream':down_loader
        }
        
    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        
        return {
            "input_dim": max(self.num_features, 1)
        }
