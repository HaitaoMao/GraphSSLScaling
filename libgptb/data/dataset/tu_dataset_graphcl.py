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
        self.split_ratio = self.config.get('ratio', 0)
        if self.split_ratio != 0:
            self.split_for_train(self.split_ratio)
            print("split generated")

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
        assert self.train_ratio + self.test_ratio + self.test_ratio <= 1
        indices = torch.load("./split/{}.pt".format(self.datasetName))
        
        if self.split_ratio==1:
            train_size = int(len(self.dataset) * self.train_ratio)
            # valid_size = int(len(self.dataset) * self.valid_ratio)
            # test_size = int(len(self.dataset) * self.test_ratio)
            partial_size = min(int(self.split_ratio*train_size),train_size)
            
            train_set = [self.dataset[i] for i in indices[:partial_size]]
            # valid_set = [self.dataset[i] for i in indices[train_size: train_size + valid_size]]
            # test_set = [self.dataset[i] for i in indices[train_size + valid_size:]]
        else:
            train_path = f"./split/{self.datasetName}/{self.datasetName}_train{self.train_ratio}_{self.split_ratio}.pt"
            train_indices = torch.load(train_path)
            train_set = [self.dataset[i] for i in train_indices]

        dataloader = DataLoader(train_set, batch_size=self.batch_size)
        # dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        dataloader_eval = DataLoader(self.dataset_eval, batch_size=self.batch_size)

        return{"train":dataloader,"valid":dataloader_eval,"test":dataloader_eval}
        
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

        print(f"split_for_train seed{seed}:{indices[0:10]}")
        train_size = int(len(self.dataset) * self.train_ratio)
        valid_size = int(len(self.dataset) * self.valid_ratio)
        test_size = int(len(self.dataset) * self.test_ratio)

        self.train = indices[:train_size]
        self.valid = indices[train_size: train_size + valid_size],
        self.test = indices[train_size + valid_size:]
        
        if not os.path.exists(f"./split/{self.datasetName}"):
                os.makedirs(f"./split/{self.datasetName}")
        cur_ratio = ratio
        while cur_ratio <= 1:
            cur_partial = min(int(cur_ratio*train_size),train_size)
            torch.save(indices[:cur_partial],f"./split/{self.datasetName}/{self.datasetName}_train{self.train_ratio}_{cur_ratio}.pt")
            cur_ratio = round(cur_ratio + ratio, 1)

        torch.save(self.train,f"./split/{self.datasetName}/{self.datasetName}_train{self.train_ratio}.pt")
        torch.save(self.valid,f"./split/{self.datasetName}/{self.datasetName}_valid{self.valid_ratio}.pt")
        torch.save(self.test,f"./split/{self.datasetName}/{self.datasetName}_test{self.test_ratio}.pt") 

    def load_split_data(self,ratio):
        # load indice 
        train_path = f"./split/{self.datasetName}/{self.datasetName}_train{self.train_ratio}_{ratio}.pt"
        #valid_path = f"./split/{self.datasetName}/{self.datasetName}_valid{self.valid_ratio}.pt"
        #test_path = f"./split/{self.datasetName}/{self.datasetName}_test{self.test_ratio}.pt"
        
        train_indices = torch.load(train_path)
        #valid_indices = torch.load(valid_path)
        #test_indices = torch.load(test_path)
        
        # creat subset dataloader 
        train_set = [self.dataset[i] for i in train_indices]
        valid_set = [self.dataset[i] for i in self.valid]
        test_set = [self.dataset[i] for i in self.test]

        return {
        'train': DataLoader(train_set, batch_size=self.batch_size),
        'valid': self.dataset_eval,
        'test': self.dataset_eval
        }
    
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"num_features":self.dataset.get_num_feature()}
