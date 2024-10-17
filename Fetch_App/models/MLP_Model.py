from torch import nn

class MLP_Model(nn.Module):
    '''
        This is a simple implementation of a multi layer perceptron model
        to predict time series data.
    '''
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.LazyLinear(self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.LazyLinear(self.input_size)
    def forward(self, inp):
        return self.fc2(self.relu(self.fc1(inp)))


            