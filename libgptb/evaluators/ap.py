import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from libgptb.evaluators.base_evaluator import BaseEvaluator
from libgptb.evaluators.eval import split_to_numpy, get_predefined_split
from sklearn.metrics import f1_score, average_precision_score
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

class MultiLabelClassifier(nn.Module):
    def __init__(self, n_features, n_label):
        super(MultiLabelClassifier, self).__init__()
        self.fc = nn.Linear(n_features, n_label) 

    def forward(self, x):
        return self.fc(x)  

class APEvaluator(BaseEvaluator):
    def __init__(self, n_features, n_classes, lr=0.001, epochs=0):
        self.model = MultiLabelClassifier(n_features, n_classes)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.criterion = nn.BCEWithLogitsLoss()
        # Cross-entropy loss for multi-class classification
        # Binary classification should also be fine

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
        x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x_train)
            is_labeled = y_train == y_train
            loss = self.criterion(outputs.to(torch.float32)[is_labeled], y_train.to(torch.float32)[is_labeled])
            loss.backward()
            self.optimizer.step()

        # Evaluation phase
        evaluator = Evaluator("ogbg-molpcba")
        print(evaluator.expected_input_format)
        print(evaluator.expected_output_format)
        pred_label = []

        for x in x_test:
            with torch.no_grad():
                pred = self.model(x).detach().cpu()  # Get the prediction and move it to CPU
                pred_label.append(pred)

        # Concatenate all predictions
        pred_label = torch.cat(pred_label, dim=0).view(y_test.shape).numpy()
        y_test = y_test.view(pred_label.shape).numpy()
        y_pred=pred_label
        y_true=y_test
        # input_dict = {"y_true": y_test, "y_pred": pred_label}
        # return evaluator.eval(input_dict)
        ap_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                y_true_col = y_test[is_labeled, i].reshape(-1, 1)
                y_pred_col = pred_label[is_labeled, i].reshape(-1, 1)
                
                ap = average_precision_score(y_true_col, y_pred_col)

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')
        print(sum(ap_list)/len(ap_list))
        return {'ap': sum(ap_list)/len(ap_list)}



