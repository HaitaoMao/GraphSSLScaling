import os
import time
import json
import numpy as np
import datetime
import torch
import sys
import torch.optim as optim
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from libgptb.executors.abstract_executor import AbstractExecutor
from libgptb.utils import get_evaluator, ensure_dir
from libgptb.evaluators import get_split, LREvaluator
from ogb.graphproppred import Evaluator
from functools import partial



class GINExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.config=config
        self.evaluator=Evaluator(self.config.get('dataset'))
        self.data_feature=data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        self.model=model.gnn.to(self.device)
        self.exp_id = self.config.get('exp_id', None)

        self.num_params=sum(p.numel() for p in self.model.parameters())
        self.learning_rate=self.config.get('learning_rate',0.001)
        self.dataset=self.config.get('dataset')
        self.epochs=config.get('max_epoch',100)
        self.drop_ratio=config.get('drop_ratio',0)
        self.batch_size=config.get('batch_size',128)
        self.training_ratio=config.get('training_ratio',0.2)
        self.random=config.get('random_seed',7)
        self.task_type=data_feature.get("task_type")
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.learning_rate)

            
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
        # for name, param in self.model.named_parameters():
        #     self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
        #                       str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum(p.numel() for p in self.model.parameters())
        self._logger.info('Total parameter numbers: {}'.format(total_num))
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
        """
        保存某个epoch的模型

        Args:
            epoch(int): 轮数
        """
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
        valid_curve = []
        test_curve = []
        val_loss_curve = []
        test_loss_curve = []
        min_loss=1.0
        self._logger.info('Start training ...')
        for epoch in range(1, self.epochs + 1):

            train_loss=_train_epoch(self.model, self.device, train_dataloader, self.optimizer, self.task_type)

            # self._writer.add_scalar('training loss', np.mean(train_loss), epoch)
            self._logger.info("epoch complete!")
            self._logger.info("evaluating now!")
            valid_perf,val_loss = _eval_epoch(self.model, self.device, eval_dataloader, self.evaluator, self.task_type)
            # test_perf, test_loss = _eval_epoch(self.model, self.device, test_dataloader, self.evaluator, self.task_type)
            message = 'Epoch [{}/{}] train_loss: {:.4f}'.\
                    format(epoch, self.epochs, train_loss)
            self._logger.info(message)
            if min_loss>val_loss:
                self.save_model_with_epoch(epoch)
            
            valid_curve.append(valid_perf[self.data_feature.get('eval_metric')])
            val_loss_curve.append(val_loss)
            # test_curve.append(test_perf[self.data_feature.get('eval_metric')])
            # test_loss_curve.append(test_loss)

        if 'classification' in self.data_feature.get('task_type'):
            best_val_epoch = np.argmax(np.array(valid_curve))
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
        best_val_loss_epoch = np.argmin(np.array(val_loss_curve))

        self.best_val_epoch=best_val_epoch
        self.best_val_loss_epoch=best_val_loss_epoch

        self._logger.info('Finished training!')
        self._logger.info('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        self._logger.info('Best val loss: {}'.format(val_loss_curve[best_val_loss_epoch]))
        # self._logger.info('Test score: {}'.format(test_curve[best_val_epoch]))
        # self._logger.info('Best test loss: {}'.format(test_loss_curve[best_val_loss_epoch]))
        # message = 'best val score: {},best val loss:{}'.\
        #             format(valid_curve[best_val_epoch], val_loss_curve[best_val_loss_epoch])
        # self._logger.info(message)
    def evaluate(self, test_dataloader):
        """
        use model to test data
        
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        test_curve = []
        test_loss_curve = []
        self.load_model_with_epoch(self.best_val_epoch)
        #self._logger.info("=====Epoch {}".format(epoch))
        test_perf, test_loss = _eval_epoch(self.model, self.device, test_dataloader, self.evaluator, self.task_type)
        #test_curve.append(test_perf[self.data_feature.get('eval_metric')])
        #    test_loss_curve.append(test_loss)
        #for epoch in range(1, self.epochs + 1):
        print('test_pref: {},  test_loss: {}'.format(test_perf,test_loss))    
        #    self._logger.info("=====Epoch {}".format(epoch))
        #    test_perf, test_loss = _eval_epoch(self.model, self.device, test_dataloader, self.evaluator, self.task_type)
        #    test_curve.append(test_perf[self.data_feature.get('eval_metric')])
        #    test_loss_curve.append(test_loss)
        #message = 'best test score : {},  best test loss : {}'.\
        #            format(test_curve[self.best_val_epoch], test_loss_curve[self.best_val_loss_epoch])
        
        #self._logger.info(message)

def _train_epoch(model, device, loader, optimizer, task_type):
    model.train()
    loss_accum = 0
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        # if(step%100==0):
        #     print("step",step)
        #     sys.stdout.flush()
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss_accum += loss.item()
            loss.backward()
            optimizer.step()
    train_loss = loss_accum/(step+1)
    return train_loss

def _eval_epoch(model, device, loader, evaluator,task_type):
    model.eval()
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()
    y_true = []
    y_pred = []
    loss_accum = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss_accum += loss.item()

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict),loss_accum/(step+1)

