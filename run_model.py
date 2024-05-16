"""
训练并评估单一模型的脚本
"""

import argparse

from libgptb.pipeline import run_model
from libgptb.utils import str2bool, add_general_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--task', type=str,
                        default='GCL', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='DGI', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='Planetoid', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--saved_model', type=str2bool,
                        default=False, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='whether re-train model if the model is trained before')
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--ratio', type=float, default=1, help='pretrain split ratio')
    parser.add_argument('--downstream_ratio', type=float, default=0.1, help='downstream train ratio')
    parser.add_argument('--downstream_task', type=str, default='orignal', help='downstream evaluation metric; default orginal')
    # 增加其他可选的参数
    add_general_args(parser)
    # 解析参数
    args = parser.parse_args()

 
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
       'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
       val is not None}
    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
             config_file=args.config_file, saved_model=args.saved_model,
            train=args.train, other_args=other_args)
