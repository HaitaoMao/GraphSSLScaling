import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score

from libgptb.evaluators.base_evaluator import BaseEvaluator


import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from tqdm import tqdm

class MLPRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x




class OGBLSCEvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval
    
    
    def evaluate(self, x: torch.FloatTensor, y: torch.FloatTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        hidden_dim = 64  # You can change the hidden layer size as needed
        output_dim = 1  # Assuming a single output regression task
        regressor = MLPRegressionModel(input_dim, hidden_dim, output_dim).to(device)
        optimizer = Adam(regressor.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_val_rmse = float('inf')
        best_test_rmse = float('inf')
        best_val_mae = float('inf')
        best_test_mae = float('inf')
        best_test_mape = float('inf')
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(Regression)',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                regressor.train()
                optimizer.zero_grad()

                output = regressor(x[split['train']])
                loss = criterion(output, y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    regressor.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred_test = regressor(x[split['test']]).detach().cpu().numpy()
                    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred_val = regressor(x[split['valid']]).detach().cpu().numpy()
                    val_rmse = mean_squared_error(y_val, y_pred_val, squared=False)
                    val_mae = mean_absolute_error(y_val, y_pred_val)

                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_test_rmse = test_rmse
                        best_val_mae = val_mae
                        best_test_mae = test_mae
                        best_test_mape = test_mape
                        best_epoch = epoch

                    pbar.set_postfix({'best test RMSE': best_test_rmse, 'best test MAE': best_test_mae})
                    pbar.update(self.test_interval)

        return {
        'best_test_rmse': float(best_test_rmse),
        'best_test_mae': float(best_test_mae),
        'best_test_mape': float(best_test_mape)
        }
        
