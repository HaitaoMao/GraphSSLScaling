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
from libgptb.data.dataset.tu_dataset_aug import TUDataset_aug
import importlib
from torch_geometric.loader import DataLoader

from copy import deepcopy
import pdb

class TUDataset_graphcl(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 128)
        self.train_ratio = self.config.get("train_ratio",0.8)
        self.valid_ratio = self.config.get("valid_ratio",0.1)
        self.test_ratio = self.config.get("test_ratio",0.1)
        
        self._load_data()
        self.split_ratio = self.config.get('ratio', 1)
        if self.split_ratio != 0:
            self.split_for_train(self.split_ratio)
            print("split generated")

    def _load_data(self):
        device = torch.device('cuda')
        path = ""
        self.aug=self.config.get("aug","dnodes")
        # orignal paper choices of datasets.
        #if self.datasetName in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        if self.datasetName in ["MUTAG", "MCF-7", "MOLT-4","P388","ZINC_full","reddit_threads","BZR"]:   
            tu_dataset_aug = getattr(importlib.import_module('libgptb.data.dataset'), 'TUDataset_aug')

        if self.aug != 'minmax': 
            self.dataset = TUDataset_aug(path, name=self.datasetName, aug=self.aug).shuffle()
        else:
            self.dataset = []
            for augment in ["dnodes","pedges","subgraph","mask_nodes","minmax_none"]:
                self.dataset.append(TUDataset_aug(path, name=self.datasetName, aug=augment).shuffle())
        # self.dataset = TUDataset_aug(path, name=self.datasetName, aug=self.aug).shuffle()
        self.dataset_eval=TUDataset_aug(path, name=self.datasetName, aug="none").shuffle()

    def get_data(self):
        assert self.train_ratio + self.test_ratio + self.test_ratio <= 1
        indices = torch.load("./split/{}.pt".format(self.datasetName))
        
        
        if self.aug != 'minmax': 
            train_size = int(len(self.dataset) * self.train_ratio)
            partial_size = min(int(self.split_ratio*train_size),train_size)
            train_set = [self.dataset[i] for i in indices[:partial_size]]
            dataloader = DataLoader(train_set, batch_size=self.batch_size)
        else:
            dataloader=[]
            for i in range(5):
                train_size = int(len(self.dataset[i]) * self.train_ratio)
                partial_size = min(int(self.split_ratio*train_size),train_size)
                train_set = [self.dataset[i][j] for j in indices[:partial_size]]
                dataloader_aug = DataLoader(train_set, batch_size=self.batch_size)
                dataloader.append(dataloader_aug)
        # dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        # train_set = [self.dataset[i] for i in indices[:partial_size]]
        # dataloader = DataLoader(train_set, batch_size=self.batch_size)
        dataloader_eval = DataLoader(self.dataset_eval, batch_size=self.batch_size)

        return{"train":dataloader,"valid":dataloader_eval,"test":dataloader_eval,"full":dataloader_eval}
        
    def split_for_train(self,ratio):
        """
        @parameter (float ratio): ratio of the dataset
        @return: return a dataloader with splited dataset
        """
        assert self.train_ratio + self.test_ratio + self.test_ratio <= 1
        seed = self.config.get("seed",0)

        split_file_path = "./split/{}.pt".format(self.datasetName)
        if os.path.exists(split_file_path):
            indices = torch.load("./split/{}.pt".format(self.datasetName))
        else:
            torch.manual_seed(self.config.get("seed",0))
            indices = torch.randperm(len(self.dataset))
            torch.save(indices,"./split/{}.pt".format(self.datasetName))


    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        if self.aug=='minmax':
            for i in range(4):
                if(self.dataset[i].get_num_feature()!=self.dataset[i+1].get_num_feature()):
                    print("different num_feature")
                    assert False
            return {"num_features":self.dataset[0].get_num_feature()}
        else:
            return {"num_features":self.dataset.get_num_feature()}
