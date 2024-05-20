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
        self.downstream_ratio = self.config.get("downstream_ratio",0.1)
        self.downstream_task = self.config.get('downstream_task','original')
        
        self._load_data()
        self.split_ratio = self.config.get('ratio', 1)

    def _load_data(self):
        device = torch.device('cuda')
        path = ""
        self.aug=self.config.get("aug","dnodes")
        # original paper choices of datasets.
        #if self.datasetName in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        if self.datasetName in ["MUTAG", "MCF-7", "MOLT-4","P388","ZINC_full","reddit_threads","BZR","github_stargazers"]:   
            tu_dataset_aug = getattr(importlib.import_module('libgptb.data.dataset'), 'TUDataset_aug')

        if self.aug != 'minmax': 
            self.dataset = TUDataset_aug(path, name=self.datasetName, aug=self.aug)
        else:
            self.dataset = []
            for augment in ["dnodes","pedges","subgraph","mask_nodes","minmax_none"]:
                self.dataset.append(TUDataset_aug(path, name=self.datasetName, aug=augment))
        # self.dataset = TUDataset_aug(path, name=self.datasetName, aug=self.aug).shuffle()
        self.dataset_unaug=TUDataset_aug(path, name=self.datasetName, aug="none")

    def get_data(self):
        assert self.train_ratio + self.test_ratio + self.test_ratio <= 1
        if os.path.exists("./split/{}.pt".format(self.datasetName)):
            indices = torch.load("./split/{}.pt".format(self.datasetName))
        else:
            torch.manual_seed(0)
            indices = torch.randperm(len(self.dataset))
            torch.save(indices,"./split/{}.pt".format(self.datasetName))


        if self.aug != 'minmax': 
            train_size = int(len(self.dataset) * self.train_ratio)
            valid_size = int(len(self.dataset) * self.valid_ratio)
            test_size = int(len(self.dataset) * self.test_ratio)
            partial_size = min(int(self.split_ratio*train_size),train_size)
            downstream_size = min(int(self.downstream_ratio*train_size),train_size)
            valid_set = [self.dataset[i] for i in indices[train_size: train_size + valid_size]]
            test_set = [self.dataset[i] for i in indices[train_size + valid_size:]]
            train_set = [self.dataset[i] for i in indices[:partial_size]]
            dataloader_train = DataLoader(train_set, batch_size=self.batch_size)
        else:
            print("size:",len(self.dataset[0]))
            dataloader_train_set=[]
            dataloader_test_set=[]
            for i in range(5):
                train_size = int(len(self.dataset[i]) * self.train_ratio)
                valid_size = int(len(self.dataset[i]) * self.valid_ratio)
                test_size = int(len(self.dataset[i]) * self.test_ratio)
                partial_size = min(int(self.split_ratio*train_size),train_size)
                downstream_size = min(int(self.downstream_ratio*train_size),train_size)
                train_set = [self.dataset[i][j] for j in indices[:partial_size]]
                valid_set = [self.dataset[i][j] for j in indices[train_size: train_size + valid_size]]
                test_set = [self.dataset[i][j] for j in indices[train_size + valid_size:]]
                dataloader_train_aug = DataLoader(train_set, batch_size=self.batch_size)
                dataloader_test_aug = DataLoader(test_set, batch_size=self.batch_size)
                dataloader_train_set.append(dataloader_test_aug)
                dataloader_test_set.append(dataloader_test_aug)

            dataloader_train=dataloader_train_set

        if self.downstream_task == 'original':
            # downstream_train = [self.dataset[i] for i in indices[:downstream_size]]
            # downstream_set = downstream_train + valid_set + test_set
            downstream_set = self.dataset_unaug
            dataloader_downstream=DataLoader(downstream_set,batch_size=self.batch_size)
        else:
            if self.aug != 'minmax': 
                downstream_set = test_set # may consider agregate valid+test
                dataloader_downstream=DataLoader(downstream_set,batch_size=self.batch_size)
            else:
                dataloader_downstream=dataloader_test_set
        dataloader_unaug = DataLoader(self.dataset_unaug, batch_size=self.batch_size)
        return{
            "train":dataloader_train,
            "valid":dataloader_unaug,
            "test":dataloader_unaug,
            "full":dataloader_unaug,
            "downstream":dataloader_downstream
            }


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
