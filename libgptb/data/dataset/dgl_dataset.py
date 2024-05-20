import torch
import os.path as osp
import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger
from libgptb.data.dataset.abstract_dataset import AbstractDataset
import importlib
import dgl


class DGLDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self._load_data()

    def _load_data(self):
        device = torch.device('cuda')
        path = osp.join(osp.expanduser('~'), 'datasets')


        if self.datasetName in ["Cora", "CiteSeer", "PubMed"]:
            dgl = getattr(importlib.import_module('dgl.data'), f'{self.datasetName.capitalize()}GraphDataset')
        if self.datasetName in ["Computers", "Photo"]:
            if self.datasetName == "Computers":
                dgl = getattr(importlib.import_module('dgl.data'), f'AmazonCoBuyComputerDataset')
            else:
                dgl = getattr(importlib.import_module('dgl.data'), f'AmazonCoBuy{self.datasetName}Dataset')
        if self.datasetName in ["CS", "Physics"]:
            dgl = getattr(importlib.import_module('dgl.data'), f'Coauthor{self.datasetName}Dataset')
        self.dataset = dgl(path)
        
        self.data = self.dataset[0]
    
    def get_data(self):
        return self.data
    
    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"input_dim": self.data.ndata['feat'].shape[1]}
    
