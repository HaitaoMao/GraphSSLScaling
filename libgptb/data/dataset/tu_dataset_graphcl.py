import torch
import os.path as osp
import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from libgptb.data.dataset.abstract_dataset import AbstractDataset
import importlib
from torch_geometric.loader import DataLoader

from copy import deepcopy
import pdb

class TUDataset_graphcl(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.train_ratio = self.config.get("train_ratio",0.8)
        self.valid_ratio = self.config.get("valid_ratio",0.1)
        self.test_ratio = self.config.get("test_ratio",0.1)
        
        self._load_data()
        self.ratio = self.config.get('ratio', 0)

        
    def _load_data(self):
        device = torch.device('cuda')
        path = ""
        aug=self.config.get("aug","random2")
        # orignal paper choices of datasets.
        #if self.datasetName in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        if self.datasetName in ["MUTAG", "MCF-7", "MOLT-4","P388","ZINC_full","reddit_threads","BZR"]:   
            tu_dataset_aug = getattr(importlib.import_module('libgptb.data.dataset'), 'TUDataset_aug')

        self.dataset = tu_dataset_aug(path, name=self.datasetName, aug=aug).shuffle()
        self.dataset_eval=tu_dataset_aug(path, name=self.datasetName, aug="none").shuffle()


    def get_data(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        dataloader_eval = DataLoader(self.dataset_eval, batch_size=self.batch_size)

        return{"train":dataloader,"valid":dataloader_eval}
    
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"num_features":self.dataset.get_num_feature()}
