from sklearn.svm import SVC, LinearSVC
from libgptb.evaluators.base_SKLearn_evluator import BaseSKLearnEvaluator
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

class RocAucEvaluator(BaseSKLearnEvaluator):
    def __init__(self, linear=True, params=None):
        self.evaluator = SVC(probability=True)
        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        super(RocAucEvaluator, self).__init__(self.evaluator, params, scoring='roc_auc')
