#%%
import os
import socket
import numpy as np
import time
import pandas as pd
import struct
import torch
import SimpleITK as sitk
import datetime

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Define the host and port to connect to
host = 'localhost'  # or use 'utrecht_gui_02' 
port = 1220

logfile = f"/utrecht_exp/gui/real_time_image_send_log.txt"

# Connect to the server
s.connect((host, port))

print(f'Connected to server at {host}:{port}')

virt_framerate = 8

print(f'Sending images at a framerate of {virt_framerate} fps...')
time.sleep(1)  # Wait for a moment before sending data

with open(logfile, 'w') as f:
    f.write(f"Log file created at {datetime.datetime.now()}\n\n")


im_bytes_list = []
end_point = 700

for i,file in enumerate(sorted(os.listdir('/utrecht_exp/data/test_images/')[:end_point])):
    file = sitk.ReadImage('/utrecht_exp/data/test_images/' + file)
    file_bytes = sitk.GetArrayFromImage(file).tobytes()
    im_bytes_list.append(file_bytes)


for i,file in enumerate(sorted(os.listdir('/utrecht_exp/data/test_images/')[:end_point])):

    start_time = time.time()
    file_bytes = im_bytes_list[i]
    file_size = len(file_bytes)
    
    # Send the size of the file first
    s.send(struct.pack('!I', file_size))
    s.send(file_bytes)

    with open(logfile, 'a') as f:
        f.write(f"Sent image of size {file_size} bytes at {datetime.datetime.now()}\n")
    end_time = time.time()
    print(f'Sent image [{i}] of size {file_size} bytes in {end_time - start_time:.4f} seconds')
    time.sleep(1/virt_framerate)  

print('Finished sending images, closing connection.')

s.close()
s.shutdown(socket.SHUT_RDWR)


