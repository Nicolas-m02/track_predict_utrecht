#%%

import socket
import struct
import time


import os
import socket
import numpy as np
import time
import pandas as pd
import struct
import torch

#data =  pd.read_csv("/utrecht_exp/data/sub-0001_task-restingstate_acq-mb3_recording-respcardiac_physio.tsv",sep='\t')

data = np.loadtxt('/utrecht_exp/data/eval_199CORfixed_angles_trace_3_outer.npy')


# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Define the host and port to connect to
host = 'localhost'  # or use '
port = 9001

# Connect to the server
s.connect((host, port))

print(f'Connected to server at {host}:{port}')
time.sleep(1)  # Wait for a moment before sending data
# Send the lung data to the server

for i, value in enumerate(data):
    print(f'Sending data point {i}: {value}')
    value = struct.pack('2f', value[0], value[1])  # Convert the float to bytes

    s.send(value)
    time.sleep(0.09)


s.close()
s.shutdown(socket.SHUT_RDWR)
