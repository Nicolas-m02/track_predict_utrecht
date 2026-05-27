import torch
import torch.nn as nn

class LSTM(nn.Module):
    """Long-short-term-memory network for supervised prediction of centroid positions. 

    Args:
        input_features (int): number of input features at each time step = 2 for SI and AP
        hidden_features (int): number of features of LSTM hidden state = hyperparameter
        output_features (int): number of output features at each time step = 2 for SI and AP
        num_layers (int): number of LSTM hidden layers
        batch_size (int): number of data patterns to be fed to network simultaneously
        seq_len_out (int): length of predicted window = 3 points
        device (torch.device): torch cuda device
        dropout (float, optional): probability of dropout [0,1] in dropout layer  
    """
    def __init__(self, input_features, hidden_features, output_features,
                 num_layers, seq_len_out, 
                 device, dropout=0, bi=False):
        super(LSTM, self).__init__()
        
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.seq_len_out = seq_len_out
        self.output_features = output_features
        self.device = device
        
        # construct lstm 
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_features,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        
        # construct fully-connected layer
        self.fc = nn.Linear(in_features=hidden_features, 
                                out_features=seq_len_out * output_features)
      
        
    def reset_h_c_states(self, batch_size=1):
        "Reset the hidden state and the cell state of the LSTM."
        
        # tensors containing the initial hidden state and initial cell state
        # with shape (num_layers * num_directions, batch_size, hidden_size)
        self.h_c = (torch.zeros(self.num_layers, batch_size, self.hidden_features),
                        torch.zeros(self.num_layers, batch_size, self.hidden_features))

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
        #print(f'Shape of input_batch: {input_batch.shape} ')  # (batch_size, seq_len_in, input_features) 
        #input_batch = input_batch [:,:,None]
        #print(input_batch.shape)
        # propagate input of shape=(batch, seq_len, input_size) through LSTM
        self.lstm.flatten_parameters()
        
        lstm_out, self.h_c = self.lstm(input_batch, self.h_c)
        #lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1).float(), self.hidden_cell)

        # print(f'Shape of lstm_out: {lstm_out.shape} ')  # (batch_size, seq_len_in, hidden_features)

        # only take the output of the last LSTM module
        # (can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction)
        predictions = self.fc(lstm_out[:,-1, :])
        
        # reshape to explicitly get output_features dimension back
        predictions = predictions.reshape(batch_size, self.seq_len_out, self.output_features)
        # print(f'Shape of predictions: {predictions.shape} ')   # (batch_size, seq_len_out, output_features)
        
        # return output windows of batch
        #print(f'Shape of predictions: {predictions.shape} ')   # (batch_size, seq_len_out, output_features)
        return predictions



class TimeSeriesTransformer2(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dim_feedforward, output_dim, output_length=3, dropout=0.1):
        super(TimeSeriesTransformer2, self).__init__()
        
        self.embedding = nn.Linear(input_dim, model_dim)
        self.output_length = output_length
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(model_dim, output_dim)
        
    def forward(self, src,output_length=3):
        src = self.embedding(src)
        encoded = self.transformer_encoder(src)
        output = self.fc_out(encoded[:, -self.output_length:, :])  # Take the last output for prediction
        return output

class LSTMTransformer(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, n_heads, num_transformer_layers, output_dim):
        super(LSTMTransformer, self).__init__()
        
        # LSTM
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_layers, batch_first=True)
        
        # Transformer Encoder
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_transformer_layers)

        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x,output_length=3):
        lstm_out, _ = self.lstm(x)  # LSTM output: (batch, seq_len, hidden_dim)
        #print(f"LSTM out shape {lstm_out.shape}")
        transformer_out = self.transformer_encoder(lstm_out) 
        #print(f"Transformer output {transformer_out.shape}") # Transformer encoder
        output = self.fc(transformer_out[:, -output_length:, :])  # Take the last timestep output
        #print(f"Output shape {output.shape}")
        return output


class LSTMTransformer2(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, n_heads, num_transformer_layers, output_dim, output_length,device):
        super(LSTMTransformer2, self).__init__()

        self.num_layers = lstm_layers
        self.hidden_features = lstm_hidden_dim
        self.device = device

        # LSTM
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_layers, batch_first=True)
        
        # Transformer Encoder
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_transformer_layers)

        
        # Fully connected layer
        self.output_dim = output_dim
        self.output_length = output_length
        predictions = output_dim*output_length
        self.fc = nn.Linear(lstm_hidden_dim, predictions)
    
    def flush_hc(self, batch_size):

        self.h_c = (torch.zeros(self.num_layers, batch_size, self.hidden_features),
                        torch.zeros(self.num_layers, batch_size, self.hidden_features))

        # move states to cuda device
        self.h_c = (self.h_c[0].to(self.device), self.h_c[1].to(self.device))

    def forward(self, x):
        batch_size = x.shape[0]
        self.flush_hc(batch_size)
        lstm_out, self.h_c = self.lstm(x,self.h_c)  # LSTM output: (batch, seq_len, hidden_dim)
        #print(f"LSTM out shape {lstm_out.shape}")
        transformer_out = self.transformer_encoder(lstm_out)  # Transformer output: (batch, seq_len, hidden_dim)
        #print(f"Transformer output {transformer_out.shape}") # Transformer encoder

        # CHANGED THIS PART TO GET ALL LSTM outs

        output = self.fc(transformer_out)[:,-1,:].reshape(batch_size,self.output_length,self.output_dim)  # Take the last timestep output
        #print(f"Output shape {output.shape}")
        return output


class GRU(nn.Module): # unused and untested
    """Gated Recurrent Unit network for supervised prediction of centroid positions.

    Args:
        input_features (int): number of input features at each time step = 2 for SI and AP
        hidden_features (int): number of features of GRU hidden state = hyperparameter
        output_features (int): number of output features at each time step = 2 for SI and AP
        num_layers (int): number of GRU hidden layers
        batch_size (int): number of data patterns to be fed to network simultaneously
        seq_len_out (int): length of predicted window = 3 points
        device (torch.device): torch cuda device
        dropout (float, optional): probability of dropout [0,1] in dropout layer  
    """
    def __init__(self, input_features, hidden_features, output_features,
                    num_layers, seq_len_out, 
                    device, dropout=0):
        super(GRU, self).__init__()
        
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.seq_len_out = seq_len_out
        self.output_features = output_features
        self.device = device
        
        # construct GRU 
        self.gru = nn.GRU(input_size=input_features, hidden_size=hidden_features,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        
        # construct fully-connected layer
        self.fc = nn.Linear(in_features=hidden_features, 
                            out_features=seq_len_out * output_features)
        
        
    def reset_h_state(self, batch_size=1):
        "Reset the hidden state of the GRU."
        
        # tensor containing the initial hidden state
        # with shape (num_layers, batch_size, hidden_size)
        self.h = torch.zeros(self.num_layers, batch_size, self.hidden_features).to(self.device)
            

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
        
        # reset hidden state for current batch of data
        self.reset_h_state(batch_size=batch_size)
        
        # propagate input of shape=(batch, seq_len, input_size) through GRU
        gru_out, self.h = self.gru(input_batch, self.h)

        # only take the output of the last GRU module
        predictions = self.fc(gru_out[:, -1, :])
        
        # reshape to explicitly get output_features dimension back
        predictions = predictions.reshape(batch_size, self.seq_len_out, self.output_features)
        
        # return output windows of batch
        return predictions
    


class LSTM_Heads(nn.Module):

    def __init__(self, input_features, hidden_features, output_features,
                 num_layers, seq_len_out, 
                 device, dropout=0,n_heads=0):
        super(LSTM_Heads, self).__init__()
        
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.seq_len_out = seq_len_out
        self.output_features = output_features
        self.n_heads = n_heads
        self.device = device
        
        # construct lstm 
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_features,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        
        # construct fully-connected layer
        self.fc = nn.Linear(in_features=hidden_features, 
                                out_features=seq_len_out * output_features)

        if self.n_heads != 0:
            self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_features, num_heads=n_heads)
        
    def reset_h_c_states(self, batch_size=1):
        "Reset the hidden state and the cell state of the LSTM."
        
        # tensors containing the initial hidden state and initial cell state
        # with shape (num_layers * num_directions, batch_size, hidden_size)
        self.h_c = (torch.zeros(self.num_layers, batch_size, self.hidden_features),
                        torch.zeros(self.num_layers, batch_size, self.hidden_features))

        # move states to cuda device
        self.h_c = (self.h_c[0].to(self.device), self.h_c[1].to(self.device))
            

    def forward(self, input_batch):

        batch_size = input_batch.shape[0]
        
        self.reset_h_c_states(batch_size=batch_size)

        self.lstm.flatten_parameters()
        
        lstm_out, self.h_c = self.lstm(input_batch, self.h_c)
        
        if self.n_heads != 0:
            print(f'lstm out {lstm_out.shape}')
            lstm_out = lstm_out.permute(1,0,2)
            print(f'permuted {lstm_out.shape}')
            attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
            print(f'attention out {attn_output.shape}')
            predictions = self.fc(attn_output[-1,:,:])
            print(f'predictions shape: {predictions.shape}')
        else:
            predictions = self.fc(lstm_out[:,-1, :])
        
        predictions = predictions.reshape(batch_size, self.seq_len_out, self.output_features)

        return predictions

