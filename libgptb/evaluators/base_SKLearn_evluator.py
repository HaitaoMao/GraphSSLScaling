import torch
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from libgptb.evaluators.base_evaluator import BaseEvaluator
from libgptb.evaluators.eval import split_to_numpy,get_predefined_split

class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params):
        self.evaluator = evaluator
        self.params = params

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
        test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')

        return {
            'micro_f1': test_micro,
            'macro_f1': test_macro,
        }