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
                        default='SSGCL', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='InfoGraph', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='ogbg-molhiv', help='the name of dataset')
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
    parser.add_argument('--downstream_task', type=str, default='original', help='downstream evaluation metric; default orginal')
    # Add general arguments
    add_general_args(parser)
    # Parse arguments
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
       'task', 'model', 'dataset', 'config_file', 'saved_model', 'train', 'exp_id', 'seed', 'ratio', 'downstream_ratio', 'downstream_task'] and
       val is not None}
    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
              config_file=args.config_file, saved_model=args.saved_model,
              train=args.train, other_args=other_args)