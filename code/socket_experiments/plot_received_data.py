#%%

import matplotlib.pyplot as plt
import numpy as np
import socket
import struct



host = 'localhost'
port = 1210



class plot_received_data:
    def __init__(self):
        self.data = []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-')  # Initialize an empty line
        self.ax.set_xlim(0, 100)  # Set x-axis limits
        self.ax.set_ylim(-1, 1)   # Set y-axis limits
        plt.show()

    def connect(self,host=host, port=port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((host, port))
        self.s.listen(1)
        print("Server listening...")
        self.conn, self.addr = self.s.accept()
        print("Connected by", self.addr)


    def update_plot(self, new_data):
        self.data.append(new_data)
        if len(self.data) > 100:  # Keep only the last 100 data points
            self.data.pop(0)
        
        self.line.set_xdata(np.arange(len(self.data)))
        self.line.set_ydata(self.data)
        self.ax.relim()  # Recalculate limits
        self.ax.autoscale_view()  # Autoscale the view
        plt.draw()

    def receive_data(self):
        # Simulate receiving data from the socket
        try:
            data = struct.unpack('2f', self.conn.recv(1024))
            # print(data.shape) 
        except:
            data = None
        return data 
    
    def close_connection(self):
        self.conn.close()
        self.s.close()

    def save_data(self):
        return self.data

import asyncio

plotter = plot_received_data()
plotter.connect()

count = 0
while count < 110:
    new_data = plotter.receive_data()
    if new_data is not None:
        print("Received data:", new_data)
        #plotter.update_plot(new_data[0])  # Update the plot with the first value of the received data
        count += 1

extracted_data = plotter.save_data()



plotter.close_connection()



