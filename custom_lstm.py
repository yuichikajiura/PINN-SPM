# ref:https://qiita.com/windfall/items/72b303867b174875f1b3
import torch.nn as nn
import torch


class CustomLSTM(nn.Module):
    def __init__(self, cfg, fc_layers, input_size, max_values, min_values=0.0):
        super().__init__()
        self.max_values = max_values  # for normalization for the outputs to roughly range [-1 1]
        self.min_values = min_values  # for normalization for the outputs to roughly range [-1 1]

        'Setting up LSTM layer'
        # defining some parameters
        self.cfg = cfg
        # defining layers
        self.lstm = nn.LSTM(input_size, cfg['hidden_lstm'], cfg['layer_lstm'], batch_first=True)

        'Setting up FC layer after the LSTM layer'
        # activation  function
        if cfg['activation'] == 'relu':
            self.activation_fc = nn.ReLU()
        elif cfg['activation'] == 'tanh':
            self.activation_fc = nn.Tanh()
        else:
            raise ValueError("activation"+cfg['activation']+"is not defined")

        # Initialize FC layer as a list using nn.Modulelist
        self.fc = nn.ModuleList([nn.Linear(fc_layers[i], fc_layers[i + 1]) for i in range(len(fc_layers) - 1)])

    def forward(self, u_seq):
        batch_size = u_seq.size(0)

        # Initializing hidden state for first input using method defined below
        h_0 = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and
        # outputs
        out, (h_n, c_n) = self.lstm(u_seq, (h_0, h_0))

        last_out = out[:, -1, :]

        for i in range(len(self.fc) - 1):
            last_out = self.fc[i](last_out)
            last_out = self.activation_fc(last_out)

        x = self.fc[-1](last_out)  # x = normalized z0 or vc0
        x = (x + 1)/2 * (self.max_values - self.min_values) + self.min_values  # Normalize back
        return x

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.cfg['layer_lstm'], batch_size, self.cfg['hidden_lstm'])
        return hidden
