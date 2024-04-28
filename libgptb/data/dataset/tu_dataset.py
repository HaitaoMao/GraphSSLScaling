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

class TUDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self._load_data()
        self.split_ratio = self.config.get('split_ratio', 0)
        if self.split_ratio != 0:
            self.split_for_train(self.split_ratio)

    def _load_data(self):
        device = torch.device('cuda')
        path = osp.join(os.getcwd(), 'raw_data')

        # orignal paper choices of datasets.
        #if self.datasetName in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        if self.datasetName in ["MUTAG", "MCF-7", "MOLT-4","P388","ZINC_full","reddit_threads"]:   
            tu_dataset = getattr(importlib.import_module('torch_geometric.datasets'), 'TUDataset')
        self.dataset = tu_dataset(path, name=self.datasetName, transform=T.NormalizeFeatures())
    
    def get_data(self):
        return DataLoader(self.dataset, batch_size=self.config.get('batch_size', 64))

    def split_for_train(self,ratio):
        """
        @parameter (float ratio): ratio of the dataset
        @return: return a dataloader with splited dataset
        """
        torch.manual_seed(self.config.get("seed",0))
        indices = torch.randperm(len(self.dataset))
        #indices = torch.load("./split/{}_{}.pt".format(dataset)) 
        if not os.path.exists(f"./split/{self.datasetName}"):
                os.makedirs(f"./split/{self.datasetName}") 

        train_ratio = ratio
        while train_ratio <= 1:
            split_size = int(len(self.dataset) * train_ratio)
            torch.save(indices[:split_size],f"./split/{self.datasetName}/{self.datasetName}_{train_ratio}.pt")
            train_ratio = round(train_ratio + ratio, 1)

    def load_split_data(self,ratio):
        # load indice 
        indices_path = f"./split/{self.datasetName}/{self.datasetName}_{ratio}.pt"
        indices = torch.load(indices_path)
        # creat subset dataloader 
        subset  = [self.dataset[i] for i in indices]
        dataloader = DataLoader(subset, batch_size=self.config.get('batch_size', 64))

        return dataloader

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        
        return {
            "input_dim": max(self.dataset.num_features, 1)
        }


if __name__ == '__main__':
    for d in ['MCF-7', 'QM9']:   
        tu = TUDataset({"dataset":f"{d}"})
