import torch
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV


def get_split(num_samples: int, train_ratio: float = 0.8, test_ratio: float = 0.1, split_ratio: float = 1, dataset = 'Cora'):
    torch.manual_seed(0)
    assert train_ratio + test_ratio < 1
    
    train_size_full = int(num_samples * train_ratio)
    train_size_split = int(num_samples * train_ratio * split_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.load("./split/{}.pt".format(dataset))
    #indices = torch.randperm(num_samples)
    print(f"get_split:{indices[0:10]}")
    #torch.save(indices,"./split/{}.pt".format(dataset)) #tensor([ 772,  728, 1741,  688, 1511, 2555, 1895, 1662, 2205,  380])
    return {
        'train': indices[:train_size_split],
        'valid': indices[train_size_full: test_size + train_size_full],
        'test': indices[test_size + train_size_full:]
    }


def from_predefined_split(data):
    assert all([mask is not None for mask in [data.train_mask, data.test_mask, data.val_mask]])
    num_samples = data.num_nodes
    indices = torch.arange(num_samples)
    return {
        'train': indices[data.train_mask],
        'valid': indices[data.val_mask],
        'test': indices[data.test_mask]
    }


def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


# class BaseEvaluator(ABC):
#     @abstractmethod
#     def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
#         pass

#     def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
#         for key in ['train', 'test', 'valid']:
#             assert key in split

#         result = self.evaluate(x, y, split)
#         return result


# class BaseSKLearnEvaluator(BaseEvaluator):
#     def __init__(self, evaluator, params):
#         self.evaluator = evaluator
#         self.params = params

#     def evaluate(self, x, y, split):
#         x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
#         ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
#         classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
#         classifier.fit(x_train, y_train)
#         test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
#         test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')

#         return {
#             'micro_f1': test_micro,
#             'macro_f1': test_macro,
#         }
