import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from libgptb.evaluators.base_evaluator import BaseEvaluator
from libgptb.evaluators.eval import split_to_numpy, get_predefined_split
from sklearn.metrics import f1_score

class MultiClassClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MultiClassClassifier, self).__init__()
        self.fc = nn.Linear(n_features, n_classes) 

    def forward(self, x):
        return self.fc(x)  

class PyTorchEvaluator(BaseEvaluator):
    def __init__(self, n_features, n_classes, lr=0.01, epochs=100):
        self.model = MultiClassClassifier(n_features, n_classes)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()  
        # Cross-entropy loss for multi-class classification
        # Binary classification should also be fine

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
        x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x_train)
            loss = self.criterion(outputs, y_train.squeeze())
            loss.backward()
            self.optimizer.step()

        # Evaluation phase
        with torch.no_grad():
            outputs_test = self.model(x_test)
            _, predictions = torch.max(outputs_test, 1)
            predictions = predictions.numpy()  # Convert predictions to a numpy array for f1_score calculation
            y_test_np = y_test.numpy().squeeze()  # Convert y_test to numpy and remove extra dimensions if any

            # Calculate F1 Scores
            micro_f1 = f1_score(y_test_np, predictions, average='micro')
            macro_f1 = f1_score(y_test_np, predictions, average='macro')

            accuracy = (predictions == y_test_np).mean()

        # Print the accuracy for checking
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Micro F1 Score: {micro_f1:.4f}')
        print(f'Macro F1 Score: {macro_f1:.4f}')

        return {'accuracy': accuracy, 'micro_f1': micro_f1, 'macro_f1': macro_f1}