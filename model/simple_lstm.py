import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        """
        Initialize the LSTM model.

        Parameters:
        - input_size: The number of expected features in the input `x`
        - hidden_size: The number of features in the hidden state `h`
        - num_layers: Number of recurrent layers.
        - dropout: If non-zero, introduces a `Dropout` layer on the outputs of each LSTM layer except the last layer.
        """
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output
