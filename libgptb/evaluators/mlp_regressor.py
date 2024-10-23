import torch
class MLPRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPRegressionModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x