import os
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import time
import json
import numpy as np
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from logging import getLogger
from torch import optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from torch.utils.tensorboard import SummaryWriter
from libgptb.executors.abstract_executor import AbstractExecutor
from libgptb.utils import get_evaluator, ensure_dir
from libgptb.evaluators import get_split, LREvaluator, SVMEvaluator, PyTorchEvaluator, RocAucEvaluator, Logits_GraphMAE, APEvaluator,OGBLSCEvaluator,MLPRegressionModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


class GraphMAEExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        
        self.config=config
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = './libgptb/cache/{}/model_cache'.format(self.exp_id)
        self.summary_writer_dir = './libgptb/cache/{}/'.format(self.exp_id)
        self.config=config
        self._logger = getLogger()
        
        self.device = self.config.get('device', torch.device('cpu'))
        self.model=model.to(self.device)
        
        self.dataset_name = config['dataset']
        self.epochs = config['max_epoch']
        self.evaluate_res_dir = './libgptb/cache/{}/evaluate_cache'.format(self.exp_id)
        self.epochs_f =config['max_epoch_f']
        self.num_hidden = config['nhid']
        self.num_layers = config['layers']
        self.encoder_type = config['encoder']
        self.decoder_type = config['decoder']
        self.loss_fn = config['loss_fn']
        self.learner = self.config.get('learner', 'adam')
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.weight_decay_f = config['weight_decay_f']
        self.linear_prob = config['linear_prob']
        self.scheduler = config['scheduler']
        self.pooler = config['pooling']
        self.deg4feat = config['deg4feat']
        self.batch_size = config['batch_size']
        self.num_class = self.config.get('num_class',2)
        self.hidden_dim = self.config.get('nhid')
        self.num_layers = self.config.get('layers')
        self.label_dim = data_feature.get('label_dim')
        
        self.load_best_epoch = self.config.get('load_best_epoch', False)
        self.patience = self.config.get('patience', 50)
        self.saved = self.config.get('saved_model', True)
        self.log_every = self.config.get('log_every', 1)
        
        self._writer = SummaryWriter(self.summary_writer_dir)
        self.lr_decay = self.config.get('lr_decay', True)
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.optimizer = self._build_optimizer()
        
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        # TODO
        self.lr_scheduler = self._build_lr_scheduler()
        self.downstream_ratio = self.config.get('downstream_ratio', 0.1)
        self.downstream_task=config.get("downstream_task","original")
        self.use_early_stop = self.config.get('use_early_stop', False)
        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)
        self.loss_func = None
        
        
    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
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
        #self.pooler.load_state_dict(check)
        self._logger.info("Loaded model at {}".format(epoch))
        
    def compute_metrics(self, predictions, targets):
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        mse = mean_squared_error(targets, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(targets, predictions)
        mape = mean_absolute_percentage_error(targets, predictions)
        return rmse, mae, mape

    def downstream_regressor(self,dataloader):
        # 初始化模型和回归器
        input_dim = self.hidden_dim   # 这个需要在第一次获得embedding后确定
        nhid = 128  # 你可以根据需要调整这个值
        output_dim = 1    # 回归任务的输出维度为1

        regressor = None
        optimizer = None
        criterion = torch.nn.MSELoss()

        # 按照指定比例划分数据集
        downstream_ratio = self.downstream_ratio  # 下游任务训练集比例
        test_ratio = 0.1  # 测试集比例

        # 获取数据集的大小
        num_samples = len(dataloader['full'])
        print(f'num_samples is {num_samples}')
        num_train = int(num_samples * downstream_ratio)
        print(f'num_train is {num_train}')
        num_test = int(num_samples * (1-test_ratio))
        print(f'num_test is {num_test}')

        num_epochs = 20
        best_test_rmse = float('inf')
        best_test_mae = float('inf')
        best_test_mape = float('inf')
        
        regressor = MLPRegressionModel(input_dim, nhid, output_dim).to(self.device)
        optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            train_loss = 0
            test_loss = 0
            correct = 0
            
            for i, batch_g in enumerate(dataloader['full']):
                data = batch_g.to(self.device)
                feat = data.x
                labels = data.y
                out = self.model.embed(feat, batch_g.edge_index)
                if self.pooler == "mean":
                    out = global_mean_pool(out, batch_g.batch)
                elif self.pooler == "max":
                    out = global_max_pool(out, batch_g.batch)
                elif self.pooler == "sum":
                    out = global_add_pool(out, batch_g.batch)
                else:
                    raise NotImplementedError
                
                if i < num_train:
                    regressor.train()
                    optimizer.zero_grad()
                    output = regressor(out)
                    loss = criterion(output, labels.to(self.device).unsqueeze(1))  # 调整维度
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    # print(train_loss)
                else:
                    print(f'i is {i}')
                    break
        
        self._logger.info(f'Downstream Epoch: {epoch+1}, Training Loss: {train_loss:.4f}')
        with torch.no_grad():
            regressor.eval()
            all_predictions = []
            all_labels = []
            for j, test_batch in enumerate(dataloader['full']):
                # print(f'j is {j}')
                if j >= num_test:
                    self._logger.debug(f'Processing batch: {j}')
                    test_batch = test_batch.to(self.device)
                    feat = test_batch.x
                    labels = test_batch.y
                    self._logger.debug('Batch moved to device')
                    out = self.model.embed(feat, test_batch.edge_index)
                    if self.pooler == "mean":
                        test_out = global_mean_pool(out, test_batch.batch)
                    elif self.pooler == "max":
                        test_out = global_max_pool(out, test_batch.batch)
                    elif self.pooler == "sum":
                        test_out = global_add_pool(out, test_batch.batch)
                    else:
                        raise NotImplementedError
                    self._logger.debug(f'Encoder model output: {test_out}')
                    test_output = regressor(test_out)
                    self._logger.debug(f'Regressor output: {test_output}')
                    all_predictions.append(test_output.cpu())
                    self._logger.debug(f'Predictions appended: {test_output.cpu()}')
                    all_labels.append(test_batch.y.cpu().float().unsqueeze(1))
                    self._logger.debug(f'Labels appended: {test_batch.y.cpu().float().unsqueeze(1)}')
                    
                
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            rmse, mae, mape = self.compute_metrics(all_predictions, all_labels)

            if mae < best_test_mae:
                best_test_rmse = rmse
                best_test_mae = mae
                best_test_mape = mape
                
        print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {mae:.4f}')
            
        result = {
        'best_test_rmse': float(best_test_rmse),
        'best_test_mae': float(best_test_mae),
        'best_test_mape': float(best_test_mape)
        }
        
        filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                    self.config['model'] + '_' + self.config['dataset']
        save_path = self.evaluate_res_dir
        if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
            json.dump(result, f)
            self._logger.info('Evaluate result is saved at ' + os.path.join(save_path, '{}.json'.format(filename)))
        return result
    
    # 定义回归模型



    def evaluate(self, dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        if self.config['dataset'] in ['PCQM4Mv2']:
            epoch_idx_list = [100-1]
        else:
            epoch_idx_list = [10-1,20-1,40-1,60-1,80-1,100-1]
            
            
        for epoch_idx in epoch_idx_list:
            self.load_model_with_epoch(epoch_idx)
            if self.downstream_task in ['original','both']:
                if self.config['dataset'] in ['PCQM4Mv2','ZINC_full']:
                    self.model.eval()
                    result=self.downstream_regressor(dataloader)
                    self._logger.info(f'(E): Best test RMSE={result["best_test_rmse"]:.4f}, MAE={result["best_test_mae"]:.4f}, MAPE={result["best_test_mape"]:.4f}')
                    
                else:
                    self.model.eval()
                    x_list = []
                    y_list = []
                    with torch.no_grad():
                        for i, batch_g in enumerate(dataloader['full']):
                            batch_g = batch_g.to(self.device)
                            feat = batch_g.x
                            labels = batch_g.y.cpu()
                            out = self.model.embed(feat, batch_g.edge_index)
                            if self.pooler == "mean":
                                out = global_mean_pool(out, batch_g.batch)
                            elif self.pooler == "max":
                                out = global_max_pool(out, batch_g.batch)
                            elif self.pooler == "sum":
                                out = global_add_pool(out, batch_g.batch)
                            else:
                                raise NotImplementedError

                            y_list.append(labels)
                            x_list.append(out)
                    x = torch.cat(x_list, dim=0)
                    y = torch.cat(y_list, dim=0)
                    split = get_split(num_samples=x.shape[0], train_ratio=0.8, test_ratio=0.1,dataset=self.config['dataset'])
                    if self.config['dataset'] == 'ogbg-molhiv': 
                        result = RocAucEvaluator()(x, y, split)
                        print(f'(E): Roc-Auc={result["roc_auc"]:.4f}')
                    elif self.dataset_name == 'ogbg-ppa':
                        self._logger.info('nclasses is {}'.format(self.num_class))
                        result = PyTorchEvaluator(n_features=x.shape[1],n_classes=self.num_class)(x, y, split)
                    elif self.config['dataset'] == 'ogbg-molpcba':
                        result = APEvaluator(self.hidden_dim, self.label_dim)(x, y, split)
                        self._logger.info(f'(E): ap={result["ap"]:.4f}')
                    # elif self.config['dataset'] == 'PCQM4Mv2':
                    #     result = OGBLSCEvaluator()(x, y, split)
                    #     self._logger.info(f'(E): Best test RMSE={result["best_test_rmse"]:.4f}, MAE={result["best_test_mae"]:.4f}, MAPE={result["best_test_mape"]:.4f}')
                    else:
                        result = SVMEvaluator(linear=True)(x, y, split)
                        print(f'(E): Best test F1Mi={result["micro_f1"]:.4f}, F1Ma={result["macro_f1"]:.4f}')
                    self._logger.info('Evaluate result is ' + json.dumps(result))
                
            if self.downstream_task == 'loss' or self.downstream_task == 'both':
                losses = self._train_epoch(dataloader['test'], epoch_idx, self.loss_func,train = False)
                result = np.mean(losses) 
                self._logger.info('Evaluate loss is ' + json.dumps(result))

            if self.downstream_task == 'logits':
                logits = Logits_GraphMAE(self.config, self.model, self._logger)
                self._logger.info("-----Start Downstream Fine Tuning-----")
                logits.train(dataloader['downstream_train'],dataloader['valid'])
                self._logger.info("-----Fine Tuning Done, Start Eval-----")
                result = logits.eval(dataloader['test'])
                self._logger.info('Evaluate acc is ' + json.dumps(result))

            filename = 'epoch'+str(epoch_idx)+"_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                            self.config['model'] + '_' + self.config['dataset']
            save_path = self.evaluate_res_dir
            file_path = os.path.join(save_path, '{}.json'.format(filename))
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(result, f)
                self._logger.info('Evaluate result is saved at ' + os.path.join(save_path, '{}.json'.format(filename)))
    
    
    
    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        self.model.to(self.device)
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))
        
        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader,self.loss_func ,epoch_idx)
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
            if epoch_idx+1 in range(5,101,5): #[10,20,30,40,50,60,70,80,90,100]:
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

        for batch in train_dataloader:
            #print(batch)
            batch_g = batch
            batch_g = batch_g.to(self.device)

            feat = batch_g.x
            loss, loss_dict = self.model(feat, batch_g.edge_index)
            #print(f"loss:{loss} ")
            self.optimizer.zero_grad()
            loss_all+=loss.item()
            # original_parameters = {name: param.clone() for name, param in self.model.named_parameters()}
            if train:
                loss.backward()
                self.optimizer.step()
            # for name, param in self.model.named_parameters():
            #         original_param = original_parameters[name]
            #         if not torch.equal(original_param, param):
            #             print(f"Parameter {name} has changed.")
          
        return loss_all /len(train_dataloader)