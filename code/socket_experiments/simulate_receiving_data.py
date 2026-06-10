#%%

import asyncio

import zmq
import time
import numpy as np

log_file = "received_data_log.txt"
with open(log_file, "w") as f:
    f.write("Timestamp,Value1,Value2\n")

class DataReceiver:
    def __init__(self, host='0.0.0.0', port=9003):
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        #self.socket.setsockopt_string(zmq.SUBSCRIBE)
        print(f"DataReceiver connected to at {self.host}:{self.port}...")

    async def receive_data(self):
        while True:
            
            self.socket.send_string("Requesting data")
            data = self.socket.recv()
            value1, value2 = np.frombuffer(data, dtype=np.float32)
            print(f"Received data: {value1}, {value2}")

            # Log the received data
            with open(log_file, "a") as f:
                f.write(f"{time.time()},{value1},{value2}\n")

            await asyncio.sleep(0.05)  # Simulate processing time

async def main():
    receiver = DataReceiver()
    await receiver.receive_data()

asyncio.run(main())

import sys
sys.exit()

#%%
import numpy as np
log_file = "received_data_log.txt"

with open(log_file, "r") as f:
    lines = f.readlines()

timestamps = []
values1 = []
values2 = []

for line in lines[1:]:  # Skip header
    timestamp, value1, value2 = line.strip().split(",")
    timestamps.append(float(timestamp))
    values1.append(float(value1))
    values2.append(float(value2))

print(1/np.mean(np.diff(timestamps)))





