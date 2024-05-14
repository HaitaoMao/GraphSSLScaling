import os
import json
import torch
import numpy as np
from logging import getLogger
from libgptb.executors.abstract_executor import AbstractExecutor
from libgptb.utils import get_evaluator, ensure_dir
from libgptb.evaluators.evaluate_embedding import evaluate_embedding
from functools import partial
from libgptb.evaluators import get_split,SVMEvaluator
from sklearn import preprocessing


class GraphCLExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.config=config
        self.data_feature=data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        self.model = model.GraphCL.to(self.device)
        self.learning_rate=self.config.get('learning_rate',0.001)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.epochs=self.config.get("epochs",20)
        self.exp_id = self.config.get('exp_id', None)

        self.cache_dir = './libgptb/cache/{}/model_cache'.format(self.exp_id)
        self.log_interval=self.config.get("log_interval",10)

        self.batch_size=self.config.get("batch_size",128)
        self.aug=self.config.get("augs","dnodes")
        self.accuracies= {'val':[], 'test':[]}
        self.dataset=self.config.get('dataset')
        self.local=self.config.get("local")=="True"
        self.prior=self.config.get("prior")=="True"
        self.DS=self.config.get("DS","MUTAG")
        self.num_gc_layers=model.num_gc_layers

        self._logger = getLogger()

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
    
    def save_model_with_epoch(self, epoch):

        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))
        
    def train(self, train_dataloader, eval_dataloader):
        self.model.eval()
        self.emb, self.y = self.model.encoder.get_embeddings(eval_dataloader)

        for epoch in range(1, self.epochs+1):
            loss_all = 0
            self.model.train()
            for data in train_dataloader:

                # print('start')
                data, data_aug = data
                self.optimizer.zero_grad()

                
                node_num, _ = data.x.size()
                data = data.to(self.device)
                x = self.model(data.x, data.edge_index, data.batch, data.num_graphs)

                if self.aug == 'dnodes' or self.aug == 'subgraph' or self.aug == 'random2' or self.aug == 'random3' or self.aug == 'random4':
                    # node_num_aug, _ = data_aug.x.size()
                    edge_idx = data_aug.edge_index.numpy()
                    _, edge_num = edge_idx.shape
                    idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                    node_num_aug = len(idx_not_missing)
                    data_aug.x = data_aug.x[idx_not_missing]

                    

                    data_aug.batch = data.batch[idx_not_missing]
                    idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                    edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                    data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

                data_aug = data_aug.to(self.device)

                '''
                print(data.edge_index)
                print(data.edge_index.size())
                print(data_aug.edge_index)
                print(data_aug.edge_index.size())
                print(data.x.size())
                print(data_aug.x.size())
                print(data.batch.size())
                print(data_aug.batch.size())
                pdb.set_trace()
                '''

                x_aug = self.model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

                # print(x)
                # print(x_aug)
                loss = self.model.loss_cal(x, x_aug)
                print(loss)
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                self.optimizer.step()
                # print('batch')
            self._logger.info('Epoch {}, Loss {}'.format(epoch, loss_all / len(train_dataloader)))

            if epoch % self.log_interval == 0:
                self.save_model_with_epoch(epoch)
                # print(accuracies['val'][-1], accuracies['test'][-1])

        tpe  = ('local' if self.local else '') + ('prior' if self.prior else '')
        with open('libgptb/log/TU-' + self.dataset + '_' + self.aug, 'a+') as f:
            s = json.dumps(self.accuracies)
            f.write('{},{},{},{},{},{},{}\n'.format(self.DS, tpe, self.num_gc_layers, self.epochs, self.log_interval, self.learning_rate, s))
            f.write('\n')

    def evaluate(self, test_dataloader):
        """
        use model to test data
        
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        # accuracies = {'val':[], 'test':[]}
        for epoch in range(1, self.epochs+1):
            if epoch % self.log_interval == 0:
                self.load_model_with_epoch(epoch)
                self.model.eval()
                emb, y = self.model.encoder.get_embeddings(test_dataloader)
                #evaluator original code used
                # acc_val, acc = evaluate_embedding(emb, y)
                # accuracies['val'].append(acc_val)
                # accuracies['test'].append(acc)
                split_ratio=self.config.get("ratio",1)
                split = get_split(num_samples=emb.shape[0], train_ratio=0.8, test_ratio=0.1,split_ratio=split_ratio,dataset=self.config['dataset'])
                
                labels = preprocessing.LabelEncoder().fit_transform(y)
                x, y = np.array(emb), np.array(labels)

                x = torch.from_numpy(x)
                y = torch.from_numpy(y)

                result=SVMEvaluator()(x,y,split)
                self._logger.info(f'(E): Best test F1Mi={result["micro_f1"]:.4f}, F1Ma={result["macro_f1"]:.4f}')
        
