#%%

import pymri
import numpy as np
import matplotlib.pyplot as plt
import os
import asyncio


handler = pymri.QueuedImageHandler()
recv = pymri.EmuImageReceiver.create('/utrecht_exp/data/all_dat_files/small_dat_files',handler)

import time 

class Receiver:

    def __init__(self, data_handler, recv):
        self.data = []
        self.handler = data_handler
        self.recv = recv
        self.time = time.time()

    async def receive_data(self):
        while True:
            image = self.handler.get_image()

            if image is not None:
                print(image)
                self.data.append(image['data'])
                self.time = time.time()

            if time.time() - self.time > 5:
                return self.close()

            await asyncio.sleep(0.001)

    def close(self):
        return self.data


async def main():
    im_receiver = Receiver(handler, recv)

    data = await im_receiver.receive_data()

    print("Collected data:")
    print(data)

    return data


result = await main()


#%%

plt.imshow(result[0])