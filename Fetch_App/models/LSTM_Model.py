import torch
from torch import nn

class LSTM_Model(nn.Module):
    '''
        This is a simple implementation of the Long Short Term Memory (LSTM) model
        to predict time series data.
    '''
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Input gate
        self.Wx_i = nn.Parameter(torch.randn((input_size, hidden_size)))
        self.Wh_i = nn.Parameter(torch.randn((hidden_size, hidden_size)))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        # Forget gate
        self.Wx_f = nn.Parameter(torch.randn((input_size, hidden_size)))
        self.Wh_f = nn.Parameter(torch.randn((hidden_size, hidden_size)))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        # Output gate
        self.Wx_o = nn.Parameter(torch.randn((input_size, hidden_size)))
        self.Wh_o = nn.Parameter(torch.randn((hidden_size, hidden_size)))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        # Cell gate
        self.Wx_c = nn.Parameter(torch.randn((input_size, hidden_size)))
        self.Wh_c = nn.Parameter(torch.randn((hidden_size, hidden_size)))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, inp, H = None, C = None):
        outputs = []
        if H is None: 
            # Initialize Hidden and Cell state
            H = torch.zeros((inp.shape[0], self.hidden_size),
                      device=inp.device)
            C = torch.zeros((inp.shape[0], self.hidden_size),
                      device=inp.device)
        for i in range(inp.shape[1]): # Iterate through sequence
            X = inp[:,i,:]
            # I_t = sigmoid(X_t @ Wx_i + H_{t-1} @ Wh_i + b_i)
            I = torch.sigmoid(X @ self.Wx_i + H @ self.Wh_i + self.b_i)
            # F_t = sigmoid(X_t @ Wx_f + H_{t-1} @ Wh_f + b_f)
            F = torch.sigmoid(X @ self.Wx_f + H @ self.Wh_f + self.b_f)
            # O_t = sigmoid(X_t @ Wx_o + H_{t-1} @ Wh_o + b_o)
            O = torch.sigmoid(X @ self.Wx_o + H @ self.Wh_o + self.b_o)
            # C_til = tanh(X_t @ Wx_c + H_{t-1} @ Wh_c + b_c)
            C_til = torch.tanh(X @ self.Wx_c + H @ self.Wh_c + self.b_c)
            # C_t = F_t * C_{t-1} + I_t * C_til
            C =  F * C + I * C_til
            H = O * torch.tanh(C)
            outputs.append(self.fc(H))
        return torch.stack(outputs, dim=1), H, C


            