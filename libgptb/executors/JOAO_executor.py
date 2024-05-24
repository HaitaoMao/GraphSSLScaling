import os
import time
import json
import torch
import datetime
import numpy as np

import libgptb.losses as L
import libgptb.augmentors as A
import torch.nn.functional as F
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from libgptb.executors.abstract_executor import AbstractExecutor
from libgptb.utils import get_evaluator, ensure_dir
from functools import partial
from libgptb.evaluators import get_split,SVMEvaluator,RocAucEvaluator,PyTorchEvaluator
from libgptb.models import DualBranchContrast
from sklearn import preprocessing


class JOAOExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.config=config
        self.data_feature=data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        self.model = model.to(self.device)
        self.exp_id = self.config.get('exp_id', None)

        self.cache_dir = './libgptb/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libgptb/cache/{}/evaluate_cache'.format(self.exp_id)
        self.summary_writer_dir = './libgptb/cache/{}/'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))
        self.log_interval=self.config.get("log_interval",10)

        self.epochs=self.config.get("epochs",100)
        self.batch_size=self.config.get("batch_size",128)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.learner = self.config.get('learner', 'adam')
        self.weight_decay = self.config.get('weight_decay', 0)
        self.lr_decay = self.config.get('lr_decay', True)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', False)
        self.hyper_tune = self.config.get('hyper_tune', False)


        self.aug=self.config.get("augs","dnodes")
        self.dataset=self.config.get('dataset')
        self.local=self.config.get("local")=="True"
        self.prior=self.config.get("prior")=="True"
        self.DS=self.config.get("DS","MUTAG")
        self.num_layers=model.num_layers
        self.downstream_task=config.get("downstream_task","original")
        self.train_ratio = self.config.get("train_ratio",0.8)
        self.valid_ratio = self.config.get("valid_ratio",0.1)
        self.test_ratio = self.config.get("test_ratio",0.1)
        self.downstream_ratio=self.config.get("downstream_ratio",0.1)
        self.mode=self.config.get("mode","fast")
        self.gamma=config.get("gamma",0.01)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = self._build_lr_scheduler()
        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)
        self.loss_func = None

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

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.encoder_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.encoder_model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def evaluate(self, test_dataloader):
        """
        use model to test data
        
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        #for epoch_idx in [50-1, 100-1, 500-1, 1000-1, 10000-1]:
        for epoch_idx in [3-1,10-1,20-1,40-1,60-1,80-1,100-1]:
                if epoch_idx+1 > self.epochs:
                    break
                self.load_model_with_epoch(epoch_idx)
                if self.downstream_task == 'original':
                    self.model.encoder_model.eval()
                    x = []
                    y = []
                    for data in test_dataloader:
                        data = data.to(self.device)
                        if data.x is None:
                            num_nodes = data.batch.size(0)
                            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
                        with torch.no_grad():
                            _, g, _, _, _, _ = self.model.encoder_model(data.x, data.edge_index, data.batch)
                            x.append(g)
                            y.append(data.y)
                    x = torch.cat(x, dim=0)
                    y = torch.cat(y, dim=0)

                    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1,dataset=self.dataset)
                    if self.config['dataset'] == 'ogbg-molhiv': 
                        result = RocAucEvaluator()(x, y, split)
                        self._logger.info(f'(E): Roc-Auc={result["roc_auc"]:.4f}')
                    elif self.config['dataset'] == 'ogbg-ppa':
                        unique_classes = torch.unique(y)
                        nclasses = unique_classes.size(0)
                        result = PyTorchEvaluator(n_features=x.shape[1],n_classes=nclasses)(x, y, split)
                        self._logger.info(f'(E): Acc={result["accuracy"]:.4f}')
                    else:
                        result = SVMEvaluator(linear=True)(x, y, split)
                        self._logger.info(f'(E): Best test F1Mi={result["micro_f1"]:.4f}, F1Ma={result["macro_f1"]:.4f}')
                elif self.downstream_task == 'loss':
                    losses = self._train_epoch(test_dataloader,epoch_idx, self.loss_func,train = False)
                    result = np.mean(losses) 
                    
                self._logger.info('Evaluate result is ' + json.dumps(result))
                filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                            self.config['model'] + '_' + self.config['dataset']
                save_path = self.evaluate_res_dir
                with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                    json.dump(result, f)
                    self._logger.info('Evaluate result is saved at ' + os.path.join(save_path, '{}.json'.format(filename)))
        
    def train(self, train_dataloader, eval_dataloader):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = np.mean(losses)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.\
                    format(epoch_idx, self.epochs, np.mean(losses),  log_lr, (end_time - start_time))
                self._logger.info(message)

            #if epoch_idx+1 in [50, 100, 500, 1000, 10000]:
            if epoch_idx+1 in [3,10,20,40,60,80,100]:
                model_file_name = self.save_model_with_epoch(epoch_idx)
                self._logger.info('saving to {}'.format(model_file_name))
            
            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss


    def _train_epoch(self, train_dataloader,epoch_idx,loss_func,train=True):
        loss_all = 0
        if train:
            self.model.train()
        else:
            self.model.eval()
        for data in train_dataloader:
            self.model.encoder_model.train()
        epoch_loss = 0
        for data in train_dataloader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

            _, _, _, _, g1, g2 = self.model.encoder_model(data.x, data.edge_index, data.batch)
            g1, g2 = [self.model.encoder_model.encoder.project(g) for g in [g1, g2]]
            loss = self.model.contrast_model(g1=g1, g2=g2, batch=data.batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        #minimax 
        loss_aug = np.zeros(5)
        for n in range(5):
            _aug_P = np.zeros(5)
            _aug_P[n] = 1
            dataset_aug_P = _aug_P
            count, count_stop = 0, len(train_dataloader)//5+1
            with torch.no_grad():
                 for data in train_dataloader:
                    data = data.to(self.device)
                    self.optimizer.zero_grad()

                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

                    _, _, _, _, g1, g2 = self.model.encoder_model(data.x, data.edge_index, data.batch)
                    g1, g2 = [self.model.encoder_model.encoder.project(g) for g in [g1, g2]]
                    loss = self.model.contrast_model(g1=g1, g2=g2, batch=data.batch)
                    loss_aug[n] += loss.item()*data.num_graphs
                    if self.mode == 'fast':
                            count += 1
                            if count == count_stop:
                                break
            if self.mode == 'fast':
                loss_aug[n] /= (count_stop*self.batch_size)
            else:
                loss_aug[n] /= len(train_dataloader.dataset)

        gamma = float(self.gamma)
        beta = 1
        b = self.model.aug_P + beta * (loss_aug - gamma * (self.model.aug_P - 1/5))

        mu_min, mu_max = b.min()-1/5, b.max()-1/5
        mu = (mu_min + mu_max) / 2
        # bisection method
        while abs(np.maximum(b-mu, 0).sum() - 1) > 1e-2:
            if np.maximum(b-mu, 0).sum() > 1:
                mu_min = mu
            else:
                mu_max = mu
            mu = (mu_min + mu_max) / 2

        self.model.aug_P = np.maximum(b-mu, 0)
        self.model.aug_P /= np.sum(self.model.aug_P)
        self.model._update_aug2()
        return epoch_loss
        