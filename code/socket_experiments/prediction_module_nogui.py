#%%
import os

import asyncio
import socket
import torch
import torch.nn as nn
import struct
import numpy as np

host = "0.0.0.0"
port = 9002  # change to 1221 for COM from SAM2
testing = False

lstm_model_path = '/utrecht_exp/all_arcs_all_sectors_raw_pretrained.pth'
os.chdir('/utrecht_exp/code/socket_experiments')
from socket_model import LSTM

# device = torch.device('cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cpu')

device = torch.device("cuda:0")

def better_manual_scaler(input_data,scale_range,backward=False):
    try:
        data = np.array(input_data.cpu())
    except:
        data = input_data

    if data.shape[1] == 1:
        return ((data - np.min(data)) / (np.max(data) - np.min(data))) * (scale_range[1] - scale_range[0]) + scale_range[0], np.array((np.max(data),np.min(data)))
    elif data.shape[1] > 1 and not backward:
        orig_scale = np.zeros((data.shape[1],2))
        for i in range(data.shape[1]):
            orig_scale[i,:] = (np.max(data[:,i],axis=0),np.min(data[:,i],axis=0))
            data[:,i] = ((data[:,i] - np.min(data[:,i],axis=0)) / (np.max(data[:,i],axis=0) - np.min(data[:,i],axis=0))) * (scale_range[1] - scale_range[0]) + scale_range[0]
        return data,np.array(orig_scale)
    elif data.shape[1] > 1 and backward:
        for i in range(data.shape[1]):
            data[:,i] = ((data[:,i]) + 1) * (scale_range[1,i] - scale_range[0,i])/2 + scale_range[0,i]
            
        return data


def undo_sliding_window_norm(inputarray,sliding_window_range, dimension=1):
    """Undo the sliding window normalization

    Args:
        inputarray (array): Normalized array
        sliding_window_range (tuple): Tuple of maximum and minimum values of the sliding window

    Returns:
        output (array): original array
    """
    #set scaling
    if dimension == 1:
        maximum = sliding_window_range[0]
        minimum = sliding_window_range[1]
        #print(type(inputarray))
        output = (inputarray+1)*(maximum-minimum)/2 + minimum
        return output

    elif dimension == 2:
        max1 = sliding_window_range[0][0]
        min1 = sliding_window_range[0][1]
        max2 = sliding_window_range[1][0]
        min2 = sliding_window_range[1][1]

        #print(f"max1: {max1}, min1: {min1}, max2: {max2}, min2: {min2}")
        output = np.zeros((np.shape(inputarray)[0],2))
        output[:,0] = (inputarray[:,0]+1)*(max1-min1)/2 + min1
        output[:,1] = (inputarray[:,1]+1)*(max2-min2)/2 + min2
        return output

import time
import matplotlib.pyplot as plt
import datetime

class predictor:
    
    def __init__(self):
        self.input_size = 100
        self.output_size = 4
        self.input_dim = 2
        self.output_dim = 2
        self.input_features = 2
        self.hidden_features = 15
        self.num_layers = 5

        self.new_data_queue = asyncio.Queue()
        

        self.seen_data = torch.zeros((0,self.input_dim)).float().to(device)
        self.received_first_data_point = False
        self.stopping_point = 850

        self.prediction_history = []

        self.true_history = []

        # initialize LSTM model
        self.lstm_model = LSTM(input_features=self.input_features, hidden_features=self.hidden_features, output_features=self.output_dim,
                 num_layers=self.num_layers, seq_len_out=self.output_size, device=device, dropout=0, bi=False).to(device)

        self.lstm_model.load_state_dict(torch.load(lstm_model_path))

        print('Intialized LSTM model.')

        print(f'Device: {torch.cuda.get_device_name()}')
        print(f'Cuda version: {torch.version.cuda}')
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())

        self.current_data_point = 0
        self.current_prediction_point = 0

        self.optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=1e-5)
        self.online_epochs = 4
        self.online_batch_size = 200
        self.online_batched_data = torch.zeros((1,self.input_size,self.input_dim)).float().to(device)
        self.online_input = torch.zeros((1,self.input_size-self.output_dim,self.input_dim)).float().to(device)
        self.online_target = torch.zeros((1,self.output_size,self.output_dim)).float().to(device)

        self.logging = True
        self.first_data_point_received = False

        # Params for GUI
        self.interpolation_point = 2.5 # latency ms/4 * Frequency ms
        self.prediction_queue = asyncio.Queue()

        for i in range(3):
            with torch.no_grad():
                self.lstm_model.eval()
                dummy_input = torch.rand((1,self.input_size,self.input_dim)).float().to(device)
                dummy_output = self.lstm_model(dummy_input)
                #print(f"Dummy output shape: {dummy_output.shape}")

            for epoch in range(1):
                self.lstm_model.train()
                self.optimizer.zero_grad()
                dummy_training = torch.rand((200,self.input_size,self.input_dim)).float().to(device)
                print(dummy_training.shape)
                output = self.lstm_model(dummy_training)
                loss = nn.MSELoss()(output, torch.rand((200,self.output_size,self.input_dim)).float().to(device))
                loss.backward()
                self.optimizer.step()

        print('Warmed up LSTM model with dummy data.')

        if self.logging:
            self.log_file = '/utrecht_exp/logs/online_log.txt'
            with open(self.log_file, 'w') as f:
                f.write('Online optimization log\n')
                f.write('=======================\n')

            with open('/utrecht_exp/logs/online_received.txt', 'w') as f:
                f.write('Online received data log\n')
                f.write('=======================\n')

    def connect(self,host=host, port=port): # receiver
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((host, port))
        self.s.listen(1)
        print("Server listening...")
        self.conn, self.addr = self.s.accept()
        print("Connected by", self.addr)


    async def predict(self):
        while True:

            prediction = await self.prediction_queue.get()
                

            if prediction is not None:
                self.current_prediction_point += 1
                # print("Prediction:", prediction)
                # print("Seen data:", self.seen_data)
                # print(data)
                self.prediction_history.append(prediction)
                #print(prediction)
                if self.logging:
                    with open(self.log_file, 'a') as f:
                        f.write(f"Prediction: {prediction[-1,:]} at {datetime.datetime.now()}\n")
                                
                if len(self.prediction_history) > self.stopping_point:
                    print("Stopping point reached. Closing connection.")
                    self.close_connection()
                    import matplotlib.pyplot as plt
                    
                    print(len(self.prediction_history))

                    plt.figure(figsize=(10,5))
                    plt.plot(np.array(self.prediction_history)[-104:-4,3,1],'b-o',label='Predicted SI')
                    plt.plot(np.array(self.true_history)[-100:,1], 'r-o',label='Actual SI')
                    plt.legend()
                    plt.title('Predicted vs Actual SI')

                    import os
                    os.makedirs('/utrecht_exp/results', exist_ok=True)
                    print(np.array(self.prediction_history).shape)
                    np.savetxt('/utrecht_exp/results/prediction_history.npy', np.array(self.prediction_history)[:,3,:])
                    np.savetxt('/utrecht_exp/results/true_history.npy', np.array(self.true_history))

                    break

                await self.optimize_online()
            await asyncio.sleep(0.001)  # Adjust sleep time as needed to control loop frequency
                
            


    async def receive_data(self):
        while True:
            try:
                data = torch.tensor(struct.unpack('2f', self.conn.recv(1024))).float().to(device)
                #print(data.shape)
                self.new_data_queue.put_nowait(data)
            except:
                pass


            await asyncio.sleep(0.001)  # Adjust sleep time as needed to control receiving frequency
                

    async def make_prediction(self):
        # Simulate making a prediction based on the received data
        while True:
            if testing:
                return [0]
            else:
                data = await self.new_data_queue.get()
                self.true_history.append(data.cpu().numpy())

                if self.logging:
                    with open('/utrecht_exp/logs/online_received.txt', 'a') as f:
                        f.write(f"Received data: {data.cpu().numpy()} at {datetime.datetime.now()}\n")
                self.current_data_point += 1

                if self.first_data_point_received == False:
                    self.first_data_point_received = True
                    #print(data.shape)
                    self.seen_data = torch.tensor(data[None,:]).float().to(device)
                    print("First data point received. Starting prediction loop...")
                    print(self.seen_data.shape)
                else:
                    self.seen_data = torch.cat((self.seen_data, data.unsqueeze(0)), dim=0).to(device)
                
                if self.seen_data.any() == None:
                    print('NONE DETECTED')

                if len(self.seen_data) >= self.input_size:
                    #print("Seen data:", self.seen_data)
                    self.lstm_model.eval()


                    
                    input_data,orig_scale = better_manual_scaler(self.seen_data, [-1,1])
                    
                    input_tensor = torch.tensor(input_data).float().to(device)  # Placeholder input
                    
                    # delete the first value in seen_data to make room for new data
                    self.seen_data = self.seen_data[1:,:]

                    # append new data to online_batched_data
                    self.online_batched_data = torch.cat((self.online_batched_data, input_tensor.unsqueeze(0)), dim=0).to(device)

                    with torch.no_grad():
                        # Convert input data to tensor and make prediction
                        # print("Input tensor:", input_tensor.shape)
                        output = self.lstm_model(input_tensor[None,:,:])[0,:,:]

                    output = undo_sliding_window_norm(output.cpu().numpy(),orig_scale,dimension=2)
                    if output is not None:
                        self.prediction_queue.put_nowait(output)
                    
                        #print("Queue size:", self.prediction_queue.qsize())
            
                await asyncio.sleep(0.001) 
                
                


    async def optimize_online(self):
        # Simulate online optimization of the model based on the received data and predictions
        # split online_batched_data into input and target
        if len(self.online_batched_data) >= self.online_batch_size:
            start_time = time.time()
            
            # Splitting into input and target            
            self.online_input = self.online_batched_data[:,-self.input_size:-self.output_size,:].unsqueeze(0)
            self.online_target = self.online_batched_data[:,-self.output_size:,:].unsqueeze(0)

            # print(self.online_input[0,:,:].shape)
            # print(self.online_target.squeeze(0).shape)
            self.lstm_model.train()
            for epoch in range(self.online_epochs):
                self.optimizer.zero_grad()
                output = self.lstm_model(self.online_input[0,:,:])
                loss = nn.MSELoss()(output, self.online_target.squeeze(0))
                loss.backward()
                self.optimizer.step()
            
            # crop out last point
            self.online_batched_data = self.online_batched_data[1:,:,:]
            end_time = time.time()
            print(f"Online optimization took {end_time - start_time:.4f} seconds.")

        
        else:
            pass

    def interpolate_prediction(self, prediction):
        # prediction is shape (4,2)
        new_prediction = np.zeros((1,2))
        new_prediction[:,0] = np.interp(self.interpolation_point, np.arange(1, 5, 1), prediction[:,1])
        new_prediction[:,1] = np.interp(self.interpolation_point, np.arange(1, 5, 1), prediction[:,0])
        return new_prediction


   
    async def close_connection(self):
        self.conn.close()
        self.s.close()

# combine coroutines into single future
async def main():
    prediction_instance = predictor()
    prediction_instance.connect(host,port)
    print("Starting prediction loop...")
    await asyncio.gather(
                                prediction_instance.receive_data(),
                                prediction_instance.predict(),
                                prediction_instance.make_prediction(),
                                )


# run concurrent script
import asyncio

asyncio.run(main())

# aysncio.run(main())

# %%




