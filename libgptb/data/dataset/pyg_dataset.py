import torch
import os.path as osp
import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger
import torch_geometric.transforms as T
from libgptb.data.dataset.abstract_dataset import AbstractDataset
import importlib


class PyGDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
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
        self.dataset = pyg(path, name=self.datasetName, transform=T.NormalizeFeatures())
        self.data = self.dataset[0].to(device)
        
    
    def get_data(self):
        return self.data
    
    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"input_dim": self.dataset.num_features}
    
