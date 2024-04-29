import os
import json
import torch
import random
from libgptb.config import ConfigParser
from libgptb.data import get_dataset
from libgptb.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, config_file={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(config_file), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)
    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    # split_ratio
    split_ratio = config.get('split_ratio', 0)
    # load dataset
    dataset = get_dataset(config)
    # transform the dataset and split
    data = dataset.get_data()
    # train_data, valid_data, test_data = data
    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']
    full_data = data['full']
  
    data_feature = dataset.get_data_feature()
    # load executor
    model_cache_file = './libgptb/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    model = get_model(config, data_feature)
    print(config['model'])
    executor = get_executor(config, model, data_feature)
    # train without split

    if train and split_ratio != 0:
        train_ratio = split_ratio
        while train_ratio <= 1:
            logger.info(f'Training With Split Data Ratio of {train_ratio}')
            data = dataset.load_split_data(train_ratio)
            train_data = data['train']
            valid_data = data['valid']
            test_data = data['test']
            executor.train(train_data, valid_data)
            if saved_model:
                executor.save_model(model_cache_file)
            train_ratio = round(train_ratio + split_ratio, 1)
            executor.evaluate(full_data)
        
    elif (train and split_ratio == 0) or not os.path.exists(model_cache_file):
        logger.info('Training With Full Data ')
        executor.train(train_data, valid_data)
        if saved_model:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    # evaluate and the result will be under cache/evaluate_cache
    executor.evaluate(test_data)
    


