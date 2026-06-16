#%%
#!/usr/bin/env python3

import asyncio
import argparse
import sys
import time
from datetime import datetime
import socket
import struct
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from auxiliary.plotting import true_vs_pred_timestamped_plot
from auxiliary.utils import perform_prediction, perform_interpolation, perform_online_training
from auxiliary.architectures import CentroidPredictionLSTM
from auxiliary.aioudp import open_local_endpoint

#%%

# command line parsing instantiation
parser = argparse.ArgumentParser()

# command line options
parser.add_argument("--host_receive", type=str, default='0.0.0.0',
                    help="the UDP receive IP address, defaults to 0.0.0.0 used with --host option when running container to access Ubuntu localhost")
parser.add_argument("--host_send", type=str, default='192.168.2.5',
                    help="the UDP send IP address, defaults to 192.168.2.5, i.e. the tracking Windows computer IP adress")
parser.add_argument("--port_receive", type=int, default=5005,
                    help="the port used by the receiver, defaults to 5005, non-privileged ports are > 1023")
parser.add_argument("--port_send", type=int, default=5004,
                    help="the port used by the sender, defaults to 5004, non-privileged ports are > 1023")
parser.add_argument("--code_version", type=str, default='2022_12_14_motion_prediction',
                    help="code version folder name")
parser.add_argument("--gpu", action="store_true",
                    help="if called, use GPU 0th device")
parser.add_argument("--model", type=str, default='LSTM',
                    help="model used for motion prediction, either LSTM or LR or sinusoid or no-prediction, defaults to LSTM")
parser.add_argument("--noisy_training", action="store_true",
                    help="if called, use LSTM trained on input data on which no moving average filter was applied")
parser.add_argument("--online", type=int, default=2,
                    help="if > 0, perform real-time optimzation of LSTM for specified nr of epochs. Defaults to 2 epochs. For the LR, any number > 0 will analytically solve the model ")
parser.add_argument("--smooth", action="store_true",
                    help="if called, use moving average filter on input sequence")
parser.add_argument("--T_imaging", type=float, default=0.25,
                    help="imaging rate in s, defaults to 0.25")
parser.add_argument("--T_prediction", type=float, default=0.01,
                    help="model prediction rate in s, defaults to 0.01")
parser.add_argument("--T_interpolation", type=float, default=0.02,
                    help="output interpolation rate in s, defaults to 0.02")
parser.add_argument("--latency", type=float, default=0.390,
                    help="end-to-end latency of the system in s to be predicted, defaults to 0.39")
parser.add_argument("--save", type=int, default=91,
                    help="if non-zero, the specified nr of seconds of input and output data will be plotted and the logs saved under /home/results/simulations")
parser.add_argument("--verbose", type=int, default=0,
                    help="if > 0, print info to console")
args = parser.parse_args()


# some combinations are not allowed
if (args.model == 'LR') and (args.online == False):
    raise ValueError('Attention: offline LR not implemented! Change input args.')
if (args.model == 'LR') and (args.gpu == True):
    raise ValueError('Attention: LR must be run on CPU! Run script without --gpu.')
if (args.model == 'sinusoidal') and (args.gpu == True):
    raise ValueError('Attention: sinusoidal must be run on CPU! Run script without --gpu.')
if (args.model == 'no-prediction') and (args.gpu == True):
    raise ValueError('Attention: no-prediction must be run on CPU! Run script without --gpu.')
if (args.model == 'sinusoidal') and (args.online == True):
    raise ValueError('Attention: sinusoidal fit has to be run with --online 0! Change input args.')
if (args.model == 'no-prediction') and (args.online == True):
    raise ValueError('Attention: no-prediction has to be run with --online 0! Change input args.')

#%%

class ConcurrentMotionPrediction:
    def __init__(self):
        print('\n')
        # time for plot file name
        self.start_time_string = time.strftime("%Y-%m-%d-%H-%M-%S")  

        # GPU settings
        dev_nr = 0  # 0,1 
        if args.gpu:
            if torch.cuda.is_available():  
                self.device = torch.device(f'cuda:{dev_nr}') 
                # set device nr to standard GPU
                torch.cuda.set_device(dev_nr)   
                print('Name of GPU device used: ', torch.cuda.get_device_name(dev_nr))
            else:  
                self.device = torch.device('cpu') 
                print('Using CPU!')
        else:
            self.device = torch.device('cpu')
            print('Using CPU!')

        # LSTM settings
        self.wdw_duration_i = 8  # in s, duration of input sequence the model was trained on
        self.wdw_duration_o = 0.5  # in s, duration of input sequence the model was trained on
        self.wdw_size_i = int(round(self.wdw_duration_i/args.T_imaging))  # nr of input data points
        self.wdw_size_o = int(round(self.wdw_duration_o/args.T_imaging))  # nr of output data points
        self.wdw_size_i_4hz = 32 # nr of input data points at 4Hz imaging
        self.wdw_size_o_4hz = 2  # nr of predicted data points the model was trained on
        input_features = 2
        hidden_features = 15
        output_features = 2
        num_layers = 5
        dropout = 0     
        bi = False

        # input sequence which is streamed
        self.input_seq_SI = []

        # pred vs true motion for plots
        self.t_true = []
        self.y_true_SI = [] 
        self.t_pred = []
        self.t_pred_log = []
        self.y_pred_SI = []
        self.y_pred_log_SI = []
        
        # training set of sequences for online LSTM
        self.train_inputs_SI = []
        self.train_labels_SI = []
        self.train_data_length_4hz = 80  # 20 s of data for online optimization
        
        # comaprison with log
        self.t_true_log = []
        self.y_true_log_SI = []

        # latency minus imaging period, needed to ease computations below
        self.time_diff = args.latency - args.T_imaging
        self.time_diff_4hz = args.latency - 0.25

        # time axes for interpolation
        self.input_t = np.arange(self.wdw_size_i)
        self.output_t = np.arange(self.wdw_size_i + self.wdw_size_o)
        # make sure last interpolation point is an acquired point by shifting the array
        self.input_t_4hz = np.arange(0, self.wdw_size_i, 0.25/args.T_imaging) - (np.arange(0, self.wdw_size_i, 0.25/args.T_imaging)[-1] - np.arange(self.wdw_size_i)[-1])
        self.output_t_4hz = np.arange(self.wdw_size_i_4hz + self.wdw_size_o_4hz)

        if args.model == 'LSTM':
            # load lstm model and parameters
            if args.noisy_training:
                path_to_model = os.path.join('/', 'home', 'code', args.code_version, 'model', 'or', 'best_model_epoch_205_MSE_val_loss_0.136172.pth')            
            else:
                path_to_model = os.path.join('/', 'home', 'code', args.code_version, 'model', 'f_or', 'best_model_epoch_545_MSE_val_loss_0.047172.pth')
            self.model = CentroidPredictionLSTM(input_features=input_features, 
                                            hidden_features=hidden_features, 
                                            output_features=output_features,
                                            num_layers=num_layers, 
                                            seq_len_out=self.wdw_size_o_4hz,
                                            dropout=dropout, bi=bi,
                                            device=self.device)
            if args.gpu:
                self.model.load_state_dict(torch.load(path_to_model))  
            else:
                self.model.load_state_dict(torch.load(path_to_model, map_location=self.device))  
            self.model.to(self.device)
            print('\n')
            print('Loaded the following model:')
            print(path_to_model)
            print(self.model)
            print('\n')
            
            # set Adam optimizer for potential online training
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6, weight_decay=1e-6)
            
            # as first backpropagation is slow, do it with zeros during model init 
            # clear stored gradients
            self.optimizer.zero_grad()
            # forward pass
            self.model.train()
            train_outputs_norm = self.model(torch.zeros(size=(1,32,2), dtype=torch.float32).to(self.device))
            # compute the loss for current windows
            train_loss_online = torch.nn.MSELoss(reduction="mean")(train_outputs_norm, torch.zeros(size=(1,2,2)).to(self.device))
            # backpropagate the errors and update the weights for current window
            train_loss_online.backward()
            self.optimizer.step() 
            self.model.eval()
            
        elif args.model == 'LR':
            self.model = Ridge(alpha=1e-4, fit_intercept=True, solver='cholesky')
            print('\n')
            print('Loaded the following model:')
            print(self.model)
            print('\n')       
            
            self.optimizer = None  # LR has closed form solution, no iterative optimizer needed
            
            # until set of windows of 20 s is reached, use a LR fitted on zeros... that data is excluded from evaluation
            self.model.fit(torch.zeros(size=(1,32), dtype=torch.float32), torch.zeros(size=(1,2), dtype=torch.float32)) 
            
        elif args.model == 'sinusoid':
            print('Using sinusoid model.')
            self.model = 'sinusoid'
        
        elif args.model == 'no-prediction':
            # if there is no prediction just run code as if you fit a sinusoid 
            # but the UDP send will send the input positions
            print('Using no-prediction (pass-through).')
            self.model = 'sinusoid'
            
        else:
            raise ValueError('Unknown model specified! Known models are LSTM, LR, sinusoid and no-prediction.')  
        

        # create UDP socket for sending
        self.sock_send = socket.socket(socket.AF_INET,  # internet
                            socket.SOCK_DGRAM)  # UDP
        
        # to store udp input
        self.current_data = []
        
        # status whether prediciton has started and whether training is required
        self.training = False
        self.predicting = False
            # store online optimization duration
        self.online_duration = []
        
        
    async def receive_udp(self, sock_receive):
        """Coroutine to await for a UDP data packet.

        Args:
            sock_receive: asynchronous receiver socket based on aioudp module.
        """
        data_receive_bytes, _ = await sock_receive.receive() # buffer size e.g. 4 bytes = 32 bit = 1 float, should be more!
        if args.verbose:
            print('\n')
            print(f'Received data: {data_receive_bytes}')  #  b'B&\x00\x00\xc2&\x00\x00'
            # print(len(data_bytes))
            # print(struct.calcsize('!2f')) 

        data = struct.unpack('<4d2?', data_receive_bytes)  # unpack using network byte order (!) or little endian (<)
        if args.verbose:
            print(f'Received and unpacked data x: {data[0]}')
            print(f'Received and unpacked data -z: {data[1]}')
            print(f'Received and unpacked data y: {data[2]}')
        
        return data
    

    async def get_data(self):
        """Coroutine to create receiver socket, get tumor positions over UDP and store them in current_data array."""
        sock_receive = await open_local_endpoint(host=args.host_receive, port=args.port_receive)
        print(f'The UDP server is running on port {args.port_receive} and waiting for data...')
         
        while True:
            self.current_data = await self.receive_udp(sock_receive)
            
                
    async def predict(self):
        """Coroutine to accumulate 8 s of input data and perform motion prediciton."""
        while True:
            if len(self.current_data) != 0:
                if len(self.input_seq_SI) == 0:
                    print('First data point added to the input sequence.')
                    # add very first point to input sequence
                    self.input_seq_SI.extend([self.current_data[2]])
                    #  print(len(input_seq_SI))
                    
                    # for comparison with log
                    self.t_true_log.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
                    self.y_true_log_SI.append(self.input_seq_SI[-1])
                    
                else:
                    if (self.input_seq_SI[-1] != self.current_data[2]):
                        # if point is different from last point, add it to input sequence --> this approach avoids UDP data loss
                        self.input_seq_SI.extend([self.current_data[2]])
                        #  print(len(input_seq_SI))
                        
                        # for comparison with log, actual data which is input
                        self.t_true_log.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
                        self.y_true_log_SI.append(self.input_seq_SI[-1])
                        
                        # accumulate 8 s of data for LSTM
                        if len(self.input_seq_SI) >= (self.wdw_size_i + self.wdw_size_o):
                            t_i = time.time()
                            
                            # perform prediction and store timestamps
                            self.input_seq_fixed_SI, self.y_true_SI, self.t_true, \
                            self.output_seq_SI, self.predicting = perform_prediction(self.input_seq_SI, 
                                                                                        self.y_true_SI, self.t_true,
                                                                                        self.wdw_size_i, self.wdw_size_i_4hz, 
                                                                                        self.input_t, self.input_t_4hz, 
                                                                                        args.T_imaging, args.smooth, args.save, 
                                                                                        self.model, self.device, args.verbose)
                            
                            # self.y_pred_log_SI.append(self.output_seq_SI[-1])
                            # self.t_pred_log.append(time.time())

                            t_f = time.time()
                            if args.verbose:
                                print(f'Time needed for prediction: {round((t_f - t_i)*1000,3)} ms')
                
                    else:
                        # if point is equal to previous point, no action has to be taken
                        pass  
            
            # LSTM input data sampling is faster than T_imaging to avoid data loss
            await asyncio.sleep(args.T_prediction - 0.005)  

                
    async def interpolate(self):
        """Coroutine to interpolate prediction of model to specified latency and send back to tracking computer over UDP."""
        while True:
            if self.predicting is False:
                if args.verbose:
                    print('Interpolator waiting for values...')
                await asyncio.sleep(args.T_interpolation) # interpolation interval 

            else:                
                self.y_pred_log_SI, self.t_pred_log = perform_interpolation(self.input_seq_fixed_SI, 
                                                                                self.y_pred_log_SI, self.t_pred_log,
                                                                                self.output_seq_SI,  
                                                                                self.wdw_size_i_4hz, self.wdw_size_o_4hz,
                                                                                self.time_diff_4hz, args.latency, 
                                                                                self.output_t_4hz, args.verbose)
                
                # send predictions to MLC-tracking software
                if self.input_seq_SI[-1] == 0.0:     
                    # if input is exactly zero, send a zero t oreset MLC positions   
                    data_send_bytes = struct.pack('<4d2?', self.current_data[0], self.current_data[1], float(0.0),
                                                self.current_data[3], self.current_data[4], self.current_data[5])
                else:
                    if args.model == 'no-prediction':
                        # sanity check, no prediction at all --> output the received input            
                        data_send_bytes = struct.pack('<4d2?', self.current_data[0], self.current_data[1], self.current_data[2],
                                                    self.current_data[3], self.current_data[4], self.current_data[5])  
                    else:
                        data_send_bytes = struct.pack('<4d2?', self.current_data[0], self.current_data[1], float(self.y_pred_log_SI[-1]),
                                                    self.current_data[3], self.current_data[4], self.current_data[5])   

      
                self.sock_send.sendto(data_send_bytes, (args.host_send, args.port_send))  
                if args.verbose:         
                    print(f'Sent data to MLC: {data_send_bytes}') 
                
                
                if len(self.y_pred_SI) == 0:
                    self.y_pred_SI.append(self.y_pred_log_SI[-1])
                    self.t_pred.append(time.time())                    
                elif self.y_pred_log_SI[-1] != self.y_pred_SI[-1]:
                    self.y_pred_SI.append(self.y_pred_log_SI[-1])
                    self.t_pred.append(time.time())
                else:
                    # as for the input data, we want to add a point to the list with the data to be plotted only if it is a new point
                    pass           
                
                await asyncio.sleep(args.T_interpolation) # interpolation interval


                try:
                    if ((len(self.t_true) == args.save/args.T_imaging)) and (args.save != 0):
                        info=''
                        # info='MeanMotion'
                        fn_saving_SI = f'{self.start_time_string}_{info}_latency{args.latency}_T_imaging{args.T_imaging}_' + \
                                        f'T_prediction{args.T_prediction}_T_interpolation{args.T_interpolation}_noisy_training{args.noisy_training}_' + \
                                        f'smooth{args.smooth}_online{args.online}_model{args.model}_SI'
                        
                        # traces and timestamps to txt
                        np.savetxt('/home/results/simulations/' + fn_saving_SI + '_y_true_log.txt', np.stack([self.t_true_log, self.y_true_log_SI], axis=1),
                                    delimiter="\t", fmt="%s")
                        np.savetxt('/home/results/simulations/' + fn_saving_SI + '_y_pred_log.txt', np.stack([self.t_pred_log, self.y_pred_log_SI], axis=1),
                                    delimiter="\t", fmt="%s")
                        np.savetxt('/home/results/simulations/' + fn_saving_SI + '_y_true.txt', np.stack([self.t_true, self.y_true_SI], axis=1),
                                    delimiter="\t")
                        np.savetxt('/home/results/simulations/' + fn_saving_SI + '_y_pred.txt', np.stack([self.t_pred, self.y_pred_SI], axis=1),
                                    delimiter="\t")                            
                        
                        # plot traces at T_imaging
                        # true_vs_pred_timestamped_plot(args.latency, self.t_true, self.y_true_SI, 
                        #                                 self.t_pred, self.y_pred_SI, 
                        #                                 fn_save=fn_saving_SI + '_full.jpg')
                        true_vs_pred_timestamped_plot(args.latency, self.t_true, self.y_true_SI, 
                                                        self.t_pred, self.y_pred_SI, 
                                                        fn_save=fn_saving_SI + '_last.jpg',
                                                        s_start=30, s_stop=90)
                        
                        if args.online:
                            np.savetxt('/home/results/simulations/' + fn_saving_SI + '_online_durations.txt', self.online_duration,
                                       fmt='%i')
                        
                        print('--------------------Positions and plots saved--------------------')   
                        # sys.exit()                 
                except:
                    print('---------------Attention! Saving did not work!-----------------------------')
                    # sys.exit()

    async def optimize(self):
        """Coroutine to accumulate 20 s of data and perform online optimization of specified model."""
        while True:
            # re-train LSTM based on last min_train_data_length motion   
            if args.online:
                if (self.predicting is False) and (self.model == 'LSTM'):
                    # print('Online training not started yet...')
                    pass
                else:
                    if len(self.input_seq_SI) >= (self.wdw_size_i + self.wdw_size_o):
                        t_i_online = time.time()
                        
                        perform_online_training(args.T_imaging, self.output_t, self.input_seq_SI, 
                                                    self.wdw_size_i, self.wdw_size_o, 
                                                    self.train_inputs_SI,  
                                                    self.train_labels_SI, args.smooth,
                                                    self.output_t_4hz, self.wdw_size_i_4hz, self.wdw_size_o_4hz, 
                                                    self.train_data_length_4hz, args.online,
                                                    self.model, self.optimizer, self.device, args.verbose)
                        
                        t_f_online = time.time()
                        if args.verbose:
                            print(f'Time needed for online optimization: {round((t_f_online - t_i_online)*1000,1)} ms') 
                        self.online_duration.append(round((t_f_online - t_i_online)*1000,1))

            await asyncio.sleep(1e-3)
            
#%%
# instantiate class (can access command line args)
prediction_instance = ConcurrentMotionPrediction()

# combine coroutines into single future
async def main():
    if args.model == 'LSTM':
        # first predict, then optimize iteratively
        await asyncio.gather(prediction_instance.get_data(),
                                prediction_instance.predict(), 
                                prediction_instance.interpolate(),
                                prediction_instance.optimize())
    else:
        # first solve analitically (very fast), then predict
        await asyncio.gather(prediction_instance.get_data(),
                                prediction_instance.optimize(),
                                prediction_instance.predict(), 
                                prediction_instance.interpolate())
#%%
# run concurrent script
asyncio.run(main()) 
