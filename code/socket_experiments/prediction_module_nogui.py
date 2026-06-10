#%%
import os
import threading
import asyncio
import socket
import torch
import torch.nn as nn
import struct
import numpy as np
import datetime
import PositionServer_pb2 as ps

host = "0.0.0.0"
port = 9002  
testing = False

host_send = "0.0.0.0"
port_send = 9003

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
    
    def __init__(self,receive_timestamps=False,send_frequency=None):
        self.input_size = 100
        self.output_size = 4
        self.input_dim = 2
        self.output_dim = 2
        self.input_features = 2
        self.hidden_features = 15
        self.num_layers = 5

        self.new_data_queue = asyncio.Queue()
        self.receive_timestamps = receive_timestamps

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
        self.current_optimization_point = 0

        # Online optimization parameters
        self.optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=1e-5)
        self.online_epochs = 4
        self.online_batch_size = 200
        self.online_batched_data = torch.zeros((1,self.input_size,self.input_dim)).float().to(device)
        self.online_input = torch.zeros((1,self.input_size-self.output_dim,self.input_dim)).float().to(device)
        self.online_target = torch.zeros((1,self.output_size,self.output_dim)).float().to(device)

        self.logging = True
        self.first_data_point_received = False


        # Params for sending predictions
        self.send_frequency = send_frequency
        import threading

        self.latest_prediction = None
        self.latest_timestamp = None
        self.lookahead_time = 250 #ms
        self.prediction_lock = threading.Lock()
       

        # LSTM Startup
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

    def connect_sender(self,host=host_send, port=port_send): # sender
        if self.send_frequency is not None:
            self.send_host = host_send
            self.send_port = port_send
            import zmq
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://{host}:{port}")
            print(f"Tracking module waiting for ZMQ connection on {host}:{port}...")
            print(self.socket.getsockopt(zmq.LAST_ENDPOINT))
            
            self.conn_send = self.socket

    async def receive_data(self):
        while True:
            try:
                if self.receive_timestamps:
                    
                    SIZE = struct.calcsize('2fQ')

                    buf = b''
                    while len(buf) < SIZE:
                        chunk = self.conn.recv(SIZE - len(buf))
                        if not chunk:
                            raise ConnectionError("socket closed")
                        buf += chunk

                    x, y, timestamp_ns = struct.unpack('2fQ', buf)
                    #print(f"Received data: x={x}, y={y}, timestamp={timestamp_ns}")
                    data = torch.tensor([x,y]).to(device)
                    self.new_data_queue.put_nowait((data, timestamp_ns))

                else:
                    data = torch.tensor(struct.unpack('2f', self.conn.recv(1024))).to(device)
                    self.new_data_queue.put_nowait((data, timestamp_ns))

            except Exception as e:
                print(f"Error occurred: {e}")
                pass


            await asyncio.sleep(0.001)  # Adjust sleep time as needed to control receiving frequency
                

    async def make_prediction(self):
        # Simulate making a prediction based on the received data
        while True:
            if testing:
                return [0]
            else:
                if self.receive_timestamps:
                    data, timestamp = await self.new_data_queue.get()
                    
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
                        if self.receive_timestamps:
                            with self.prediction_lock:
                                self.latest_prediction = output.copy()
                                self.latest_timestamp = timestamp
                        else:
                            with self.prediction_lock:
                                self.latest_prediction = output.copy()                        
                        self.current_prediction_point += 1

                        if self.logging:
                            with open(self.log_file, 'a') as f:
                                f.write(f"Prediction: {output[-1,:]} at {datetime.datetime.now()}\n")
                                

                        #print("Queue size:", self.prediction_queue.qsize())
            
                await asyncio.sleep(0.001) 
    

    async def optimize_online(self):

        while True:
            await asyncio.sleep(0.005)

            if self.current_prediction_point > self.current_optimization_point and len(self.online_batched_data) >= self.online_batch_size:
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
                #print(f"Online optimization took {end_time - start_time:.4f} seconds.")
                self.current_optimization_point += 1
            
            else:
                pass

    def interpolate_prediction(self, prediction,interpolation_point):
        # prediction is shape (4,2)
        new_prediction = np.zeros(3)
        new_prediction[0] = np.interp(interpolation_point, np.arange(1, 5, 1), prediction[:,1])     ########### 1?
        new_prediction[1] = np.interp(interpolation_point, np.arange(1, 5, 1), prediction[:,0])     ########### 0?
        new_prediction[2] = 64
        return new_prediction

    # async def send_prediction_loop(self):
    #     if self.send_frequency is not None:
    #         print(f"Starting to send interpolated predictions at {self.send_frequency} Hz to {self.send_host}:{self.send_port}...")
    #         while True:
    #             if self.first_prediction_performed == False:
    #                 await asyncio.sleep(0.001)
    #                 continue
    #             if self.prediction_queue.qsize() > 0:
    #                 if self.receive_timestamps:
    #                     prediction, timestamp = await self.prediction_queue.get()
    #                     print(f"Got prediction from queue with timestamp {timestamp}: {prediction}")
    #                 else:
    #                     prediction = await self.prediction_queue.get()
    #                     print(f"Got prediction from queue: {prediction}")
            
    #             # Convert timestamp to interpolation point
    #             latency = (datetime.datetime.now().timestamp() - timestamp/1e9)
    #             #print(f"Latency: {latency:.4f} seconds")
    #             interpolation_point = latency / 90
    #             interpolated_prediction = self.interpolate_prediction(prediction, interpolation_point)
    #             # Send the interpolated prediction to the tracking module
    #             self.conn_send.send(struct.pack('2f', *interpolated_prediction.flatten()))
            
            
    #             await asyncio.sleep(1/self.send_frequency)  # Adjust sleep time to control sending frequency

    # async def send_prediction_loop(self):

       
    #     latest_prediction = None
    #     latest_timestamp = None

    #     period = 1.0 / self.send_frequency

    #     print(
    #         f"Starting to send interpolated predictions at "
    #         f"{self.send_frequency} Hz to {self.send_host}:{self.send_port}..."
    #     )

    #     while True:
    #         # Drain queue and keep only the newest prediction
    #         while not self.prediction_queue.empty():
    #             if self.receive_timestamps:
    #                 latest_prediction, latest_timestamp = await self.prediction_queue.get()
    #             else:
    #                 latest_prediction = await self.prediction_queue.get()

    #         if latest_prediction is not None:
    #             latency = (
    #                 datetime.datetime.now().timestamp()
    #                 - latest_timestamp / 1e9
    #             )

    #             interpolation_point = latency / 90

    #             interpolated_prediction = self.interpolate_prediction(
    #                 latest_prediction,
    #                 interpolation_point,
    #             )

    #             self.conn_send.send(
    #                 struct.pack(
    #                     "2f",
    #                     *interpolated_prediction.flatten(),
    #                 )
    #             )

    #         await asyncio.sleep(period)

    def send_prediction_loop(self):

        period = 1.0 / self.send_frequency

        print(
            f"Starting to send interpolated predictions at "
            f"{self.send_frequency} Hz"
        )

        next_time = time.perf_counter()

        while True:
            #print("Waiting for message from tracking module...")
            msg = self.conn_send.recv()
            #print(f"Received message: {msg}")

            with self.prediction_lock:
                prediction = self.latest_prediction
                timestamp = self.latest_timestamp

            if prediction is None:
                # sending zeros until we have a prediction
                vec = ps.Vector()
                vec.x = 0.0
                vec.y = 0.0
                vec.z = 0.0

                print(datetime.datetime.now(),' x y z ',vec.x,' ',vec.y,' ',vec.z)
                rep = ps.LetterRep()
                rep.payload = vec.SerializeToString()
                rep.message_type = ps.Letter.POSITION_VECTOR
                env = ps.Envelope()
                env.payload = rep.SerializeToString()
                env.message_type = ps.Envelope.LETTER_REP
                self.conn_send.send(env.SerializeToString())
                continue

            latency = (
                datetime.datetime.now().timestamp()
                - timestamp / 1e9 + self.lookahead_time/1000
            )
            print(f"Latency: {latency:.4f} seconds")
            interpolation_point = latency*1000 / 100
            print(f"Interpolation point: {interpolation_point:.4f}")
            if interpolation_point > 4:
                print("Interpolation point is greater than 4. Taking last predicion.")    
            interpolated_prediction = self.interpolate_prediction(
                prediction,
                interpolation_point,
            )

            # Convert positions from pixel space to real space
            # MLC takes center of the image as (0,0)

            interpolated_prediction = (interpolated_prediction-64)*1.95 #mm




            # create and send message
            vec = ps.Vector()
            arr = np.asarray(interpolated_prediction).reshape(-1)

            vec.x = float(arr[0])
            vec.y = float(arr[1])
            vec.z = float(arr[2])

            print(datetime.datetime.now(),' replying x y z ',vec.x,' ',vec.y,' ',vec.z)
            rep = ps.LetterRep()
            rep.payload = vec.SerializeToString()
            rep.message_type = ps.Letter.POSITION_VECTOR
            env = ps.Envelope()
            env.payload = rep.SerializeToString()
            env.message_type = ps.Envelope.LETTER_REP
            self.conn_send.send(env.SerializeToString())


            next_time += period
            sleep_time = next_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()


    
# combine coroutines into single future
async def main():
    prediction_instance = predictor(receive_timestamps=True, send_frequency=50)
    prediction_instance.connect_sender(host_send, port_send)
    prediction_instance.connect(host,port)
    sender_thread = threading.Thread(
        target=prediction_instance.send_prediction_loop,
        daemon=True
    )
    sender_thread.start()
    print("Starting prediction loop...")
    await asyncio.gather(
                                prediction_instance.receive_data(),
                                prediction_instance.make_prediction(),
                                prediction_instance.optimize_online(),
                                )


# run concurrent script
import asyncio

asyncio.run(main())

# aysncio.run(main())

# %%




