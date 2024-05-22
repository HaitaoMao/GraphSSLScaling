import torch
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV, StratifiedKFold
from libgptb.evaluators.base_evaluator import BaseEvaluator
from libgptb.evaluators.eval import split_to_numpy,get_predefined_split

class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params, scoring = "accuracy"):
        self.evaluator = evaluator
        self.params = params
        self.scoring = scoring

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring= self.scoring, verbose=0)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        if self.scoring == 'accuracy':
            test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
            test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')

            return {
                'micro_f1': test_micro,
                'macro_f1': test_macro,
            }

        elif self.scoring == 'roc_auc':
            y_proba = classifier.predict_proba(x_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            return {'roc_auc' : roc_auc }

