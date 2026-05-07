#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
class CentroidPredictionLSTM(nn.Module):
    """Long-short-term-memory network for supervised prediction of centroid positions. 

    Args:
        input_features (int): number of input features at each time step
        hidden_features (int): number of features of LSTM hidden state
        output_features (int): number of output features at each time step
        num_layers (int): number of LSTM hidden layers
        batch_size (int): number of data patterns to be fed to network simultaneously
        seq_len_in (int): length of input window
        seq_len_out (int): length of predicted window
        device (torch.device): torch cuda device
        dropout (float, optional): probability of dropout [0,1] in dropout layer  
        bi (bool, optional): if True, becomes a bidirectional LSTM
    """
    def __init__(self, input_features, hidden_features, output_features,
                 num_layers, seq_len_out, 
                 device, dropout=0, bi=False):
        super(CentroidPredictionLSTM, self).__init__()
        
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.seq_len_out = seq_len_out
        self.output_features = output_features
        self.bi = bi
        self.device = device
        
        # construct lstm 
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_features,
                            num_layers=num_layers, dropout=dropout, 
                            bidirectional=self.bi, batch_first=True)
        
        # construct fully-connected layer
        if self.bi is False:
            self.fc = nn.Linear(in_features=hidden_features, 
                                out_features=seq_len_out * output_features)
        if self.bi:
            self.fc = nn.Linear(in_features=hidden_features * 2, 
                                out_features=seq_len_out * output_features)        
        
    def reset_h_c_states(self, batch_size=1):
        "Reset the hidden state and the cell state of the LSTM."
        
        # tensors containing the initial hidden state and initial cell state
        # with shape (num_layers * num_directions, batch_size, hidden_size)
        if self.bi is False:
            self.h_c = (torch.zeros(self.num_layers, batch_size, self.hidden_features),
                        torch.zeros(self.num_layers, batch_size, self.hidden_features))
        if self.bi:
            self.h_c = (torch.zeros(self.num_layers * 2, batch_size, self.hidden_features),
                        torch.zeros(self.num_layers * 2, batch_size, self.hidden_features))           

        # move states to cuda device
        self.h_c = (self.h_c[0].to(self.device), self.h_c[1].to(self.device))
            

    def forward(self, input_batch):
        """Compute forward pass through network.

        Args:
            input_batch (array): input array with shape (batch_size, seq_len_in, input_features)   

        Returns:
            predictions (array): output array with shape (batch_size, seq_len_out, output_features) 
                                    containing predicted time sequence
        """
        # get batch size from current input batch
        batch_size = input_batch.shape[0]
        
        # reset hidden state and cell state for current batch of data
        self.reset_h_c_states(batch_size=batch_size)
        # print(f'Shape of input_batch: {input_batch.shape} ')  # (batch_size, seq_len_in, input_features) 
               
        # propagate input of shape=(batch, seq_len, input_size) through LSTM
        lstm_out, self.h_c = self.lstm(input_batch, self.h_c)
        # print(f'Shape of lstm_out: {lstm_out.shape} ')  # (batch_size, seq_len_in, hidden_features)

        # only take the output of the last LSTM module
        # (can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction)
        predictions = self.fc(lstm_out[:, -1, :])
        
        # reshape to explicitly get output_features dimension back
        predictions = predictions.reshape(batch_size, self.seq_len_out, self.output_features)
        # print(f'Shape of predictions: {predictions.shape} ')   # (batch_size, seq_len_out, output_features)
        
        # return output windows of batch
        return predictions
 
