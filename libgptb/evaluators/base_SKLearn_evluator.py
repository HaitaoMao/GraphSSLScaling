import torch
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV, StratifiedKFold
from libgptb.evaluators.base_evaluator import BaseEvaluator
from libgptb.evaluators.eval import split_to_numpy,get_predefined_split

class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params, scoring = "accuracy"):
    def __init__(self, evaluator, params, scoring = "accuracy"):
        self.evaluator = evaluator
        self.params = params
        self.scoring = scoring
        self.scoring = scoring

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring= self.scoring, verbose=0)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring= self.scoring, verbose=0)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        if self.scoring == 'accuracy':
            test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
            test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')
        y_pred = classifier.predict(x_test)
        if self.scoring == 'accuracy':
            test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
            test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')

            return {
                'micro_f1': test_micro,
                'macro_f1': test_macro,
            }

        elif self.scoring == 'roc_auc':
            rocauc_list = []
            y_pred = classifier.predict_proba(x_test)
            for i in range(y_test.shape[1]):
            #AUC is only defined when there is at least one positive data.
                if np.sum(y_test[:,i] == 1) > 0 and np.sum(y_test[:,i] == 0) > 0:
                    # ignore nan values
                    is_labeled = y_test[:,i] == y_test[:,i]
                    rocauc_list.append(roc_auc_score(y_test[is_labeled,i], y_pred[is_labeled,i]))
    
            if len(rocauc_list) == 0:
                raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')
    
            return {'roc_auc': sum(rocauc_list)/len(rocauc_list)}

