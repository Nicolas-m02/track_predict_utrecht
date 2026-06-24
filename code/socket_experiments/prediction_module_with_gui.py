#%%
import os
import sys
import threading
import asyncio
import socket
import torch
import torch.nn as nn
import struct
import numpy as np
import datetime
import PositionServer_pb2 as ps
import pyyaml

with open('/utrecht_exp/config.yaml', 'r') as f:
    config = pyyaml.safe_load(f)

host = "0.0.0.0"
port = 9002  # change to 1221 for COM from SAM2
testing = False


#port = 6055 
# ######################## watch out: 
# if directly connect to mrtc then port 4005 should be exposed 
# (right now done so with SAM container, need to be changed!) 

# change in MRTC to send positions:   sock_.connect("tcp://0.0.0.0:" + std::to_string(port)); (in zmqpub.cpp)


host_send = config['ports']['host_receive_com']
port_send = config['ports']['port_receive_com']

# host_gui = 'gui_container'
host_gui = config['ports']['host_gui']
port_gui = config['ports']['port_gui_predictions']


lstm_model_path = config['predictor']['lstm_model_path']

os.chdir('/utrecht_exp/code/socket_experiments')
from socket_model import LSTM

device = torch.device('cuda:0')
print(device)

def better_manual_scaler(input_data, scale_range, backward=False):
    try:
        data = np.array(input_data.cpu())
    except:
        data = input_data

    if data.shape[1] == 1:
        mn = np.min(data)
        mx = np.max(data)
        den = mx - mn
        den = den if den != 0 else 1e-8

        scaled = ((data - mn) / den) * (scale_range[1] - scale_range[0]) + scale_range[0]
        return scaled, np.array((mx, mn))

    elif data.shape[1] > 1 and not backward:
        orig_scale = np.zeros((data.shape[1], 2))

        for i in range(data.shape[1]):
            mx = np.max(data[:, i], axis=0)
            mn = np.min(data[:, i], axis=0)
            den = mx - mn
            den = den if den != 0 else 1e-8

            orig_scale[i, :] = (mx, mn)

            data[:, i] = ((data[:, i] - mn) / den) * (
                scale_range[1] - scale_range[0]
            ) + scale_range[0]

        return data, np.array(orig_scale)

    elif data.shape[1] > 1 and backward:
        for i in range(data.shape[1]):
            data[:, i] = ((data[:, i]) + 1) * (
                scale_range[1, i] - scale_range[0, i]
            ) / 2 + scale_range[0, i]

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

        self.no_prediction = config['predictor']['no_prediction']
        self.connect_to_mrtc = config['predictor']['connect_to_mrtc']
        self.logging = True
        self.first_data_point_received = False

        # Scaling params
        self.img_frequency = config['predictor']['framerate']  # Hz

        
        # Params for GUI
        self.prediction_queue = asyncio.Queue()
        self.gui_queue = asyncio.Queue()

        # Params for sending predictions
        self.send_frequency = send_frequency
        import threading

        self.latest_prediction = None
        self.previous_prediction = None
        self.latest_timestamp_recv_mri = None
        self.previous_timestamp_recv_mri = None
        self.prediction_done_timestamp = None
        self.lookahead_time = config['predictor']['lookahead_time']
        self.prediction_lock = threading.Lock()


        # LSTM Startup
        for i in range(3):
            with torch.no_grad():
                self.lstm_model.eval()
                dummy_input = torch.rand((1,self.input_size,self.input_dim)).float().to(device)
                dummy_output = self.lstm_model(dummy_input)

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
            self.time_logging = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"/utrecht_exp/logs/online_log_{self.time_logging}.txt", 'w') as f:
                f.write('Online optimization log\n')
                f.write('=======================\n')

            with open(f'/utrecht_exp/logs/online_received_{self.time_logging}.txt', 'w') as f:
                f.write('Online received data log\n')
                f.write('=======================\n')

            with open(f'/utrecht_exp/logs/pred_log_file_{self.time_logging}.txt', 'w') as f:
                f.write('Predictions \n')
                f.write('=======================\n')

        if self.no_prediction: 
            print("Running with no prediction")

    def connect(self,host=host, port=port): # receiver

        if self.connect_to_mrtc:
            import zmq
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.bind(f"tcp://{host}:{port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            print(f"Tracking module waiting for ZMQ connection on {host}:{port}...")
            self.conn = self.socket
        else:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.s.bind((host, port))
            self.s.listen(1)
            print("Server listening...")
            self.conn, self.addr = self.s.accept()
            print("Connected by", self.addr)   

    def connect_to_gui(self, host=host_gui, port=port_gui): # sender
        self.gui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gui_socket.connect((host, port))
        print(f"Connected to GUI at {host}:{port}")

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
                if  self.connect_to_mrtc:

                    msg = self.conn.recv()

                    env = ps.Envelope()
                    env.ParseFromString(msg)

                    rep = ps.LetterRep()
                    rep.ParseFromString(env.payload)

                    v = ps.Vector()
                    v.ParseFromString(rep.payload)

                    data = torch.tensor([v.x, v.y], device=device)
                    print(data)
                    timestamp_ns = time.time_ns()                        # timestamp not sent in msg therefore create timestamp when receiving motion
                    self.new_data_queue.put_nowait((data, timestamp_ns))       

                else:
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
                    data, timestamp_recv_mri = await self.new_data_queue.get()
                    
                else:
                    data = await self.new_data_queue.get()
                self.true_history.append(data.cpu().numpy())

                if self.logging:
                    with open(f'/utrecht_exp/logs/online_received_{self.time_logging}.txt', 'a') as f:
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
                                if self.no_prediction:
                                    self.previous_prediction = self.latest_prediction
                                    self.latest_prediction = data.cpu().numpy().copy()
                                    self.previous_timestamp_recv_mri = self.latest_timestamp_recv_mri
                                    self.latest_timestamp_recv_mri = timestamp_recv_mri
                                else:
                                    self.previous_prediction = self.latest_prediction
                                    self.latest_prediction = output.copy()
                                    self.previous_timestamp_recv_mri = self.latest_timestamp_recv_mri
                                    self.latest_timestamp_recv_mri = timestamp_recv_mri
                                
                        else:
                            with self.prediction_lock:
                                if self.no_prediction:
                                    self.previous_prediction = self.latest_prediction
                                    self.latest_prediction = data.cpu().numpy().copy()
                                else:
                                    self.previous_prediction = self.latest_prediction
                                    self.latest_prediction = output.copy()                        
                        self.current_prediction_point += 1
                        self.gui_queue.put_nowait(output)
                        self.prediction_done_timestamp = datetime.datetime.now().timestamp()

                        if self.logging:
                            with open(f"/utrecht_exp/logs/online_log_{self.time_logging}.txt", 'a') as f:
                                f.write(f"Prediction: {output[-1,:]} at {datetime.datetime.now()}\n")
                        #print("Queue size:", self.prediction_queue.qsize())
            
                await asyncio.sleep(0.001) 

    async def optimize_online(self):

        while True:
            await asyncio.sleep(0.001)

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
                print(f"Online optimization took {end_time - start_time:.4f} seconds.")
                self.current_optimization_point += 1
            
            else:
                pass

    def interpolate_prediction(self, prediction,interpolation_point):
        # prediction is shape (4,2), no prediction is shape (2,)
        new_prediction = np.zeros(2)
        new_prediction[0] = np.interp(interpolation_point, np.arange(1, 5, 1), prediction[:,0])
        new_prediction[1] = np.interp(interpolation_point, np.arange(1, 5, 1), prediction[:,1])  
        return new_prediction


    async def send_to_gui(self):
        print("Starting send_to_gui loop...")
        # Interpolate points to match expected latency
        while True:
            new_prediction = await self.gui_queue.get()
            #print(f"New prediction from queue: {new_prediction.shape}")
            while not self.gui_queue.empty():
                new_prediction = await self.gui_queue.get()
            # interpolate to match expected latency
            #print(new_prediction.shape)

            interpolated_prediction = self.interpolate_prediction(new_prediction,3)
            #print(f"Interpolated prediction: {interpolated_prediction.shape}")


            try:
                self.gui_socket.send(struct.pack('2f', interpolated_prediction[0], interpolated_prediction[1]))
                print(f"Sent prediction {interpolated_prediction} to GUI.")
            except Exception as e:
                print(f"Error sending to GUI: {e}")
            await asyncio.sleep(0.001)  # Adjust sleep time as needed to control sending frequency
            
    def send_prediction_loop(self):

        period = 1.0 / self.send_frequency

        print(
            f"Starting to send interpolated predictions at "
            f"{self.send_frequency} Hz"
        )

        next_time = time.perf_counter()

        while True:
            msg = self.conn_send.recv()

            with self.prediction_lock:
                prediction = self.latest_prediction
                timestamp_recv_mri = self.latest_timestamp_recv_mri

            if prediction is None or np.isnan(prediction).any() and np.isnan(self.previous_prediction).any():
                print("No prediction ready or NAN values for pred and also previous pred. Sending zeros.")
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


            if np.isnan(prediction).any():
                print("NAN values detected for prediction output.")
                if not np.isnan(self.previous_prediction).any():
                    print("Sending predictions of previous point")
                    
                    timestamp_recv_mri = self.previous_timestamp_recv_mri
                    prediction = self.previous_prediction


            # Variant A: Get time between MRI and current, and add the time that the MLCs need to get to this position
            latency_sam_lstm_ms = (
                datetime.datetime.now().timestamp() * 1000
                - timestamp_recv_mri / 1e6
                + self.lookahead_time  # this should be MLC adaptation latency
            )

            # Variant B: deterine end-to-end latency and add time when prediction is ready to when it is requestedd by miniplan
            #latency_sam_lstm_ms = (
            #    datetime.datetime.now().timestamp() * 1000
            #    - self.prediction_done_timestamp * 1000
            #    + self.lookahead_time   # this should be end-to-end latency
            #)

            print(f"Current latency for sam and lstm: {latency_sam_lstm_ms:.4f} ms")
            interpolation_point = latency_sam_lstm_ms/ 1000 * self.img_frequency # ms -> prediction steps
            if interpolation_point > 4:
                print("Interpolation horizon is greater than Predictions * MRI frequency. Taking last predicion.")    
            
            
            if self.no_prediction:
                interpolated_prediction_mm = prediction
            else:
                interpolated_prediction_mm = self.interpolate_prediction(prediction, interpolation_point)

            # create and send message
            vec = ps.Vector()
            arr = np.asarray(interpolated_prediction_mm).reshape(-1)

            vec.x = float(arr[0])
            vec.y = float(arr[1])
            vec.z = float(0)


            print(datetime.datetime.now(),' current predcition x y z ',vec.x,' ',vec.y,' ',vec.z)
            rep = ps.LetterRep()
            rep.payload = vec.SerializeToString()
            rep.message_type = ps.Letter.POSITION_VECTOR
            env = ps.Envelope()
            env.payload = rep.SerializeToString()
            env.message_type = ps.Envelope.LETTER_REP
            self.conn_send.send(env.SerializeToString())
                            
            
            if self.logging:
                with open(f'/utrecht_exp/logs/pred_log_file_{self.time_logging}.txt', 'a') as f:
                    f.write(f"Sent prediction: {interpolated_prediction_mm} mm at {datetime.datetime.now()}\n")

            next_time += period
            sleep_time = next_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()




    async def close_connection(self):
        self.conn.close()
        self.s.close()

# combine coroutines into single future
async def main():
    prediction_instance = predictor(receive_timestamps=config['settings']['timestamps'],send_frequency=25)
    prediction_instance.connect_sender(host_send, port_send)
    prediction_instance.connect(host,port)
    prediction_instance.connect_to_gui(host_gui,port_gui)
    sender_thread = threading.Thread(
        target=prediction_instance.send_prediction_loop,
        daemon=True
    )
    sender_thread.start()
    print("Starting prediction loop...")
    await asyncio.gather(
                                prediction_instance.send_to_gui(),
                                prediction_instance.receive_data(),
                                prediction_instance.optimize_online(),
                                prediction_instance.make_prediction(),
                                )


# run concurrent script
import asyncio

asyncio.run(main())

# aysncio.run(main())

