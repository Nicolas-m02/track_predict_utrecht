#%%

import datetime
import os
import numpy as np
import socket
import struct
import time
import cv2
import asyncio
import matplotlib.pyplot as plt
os.chdir("/utrecht_exp/segmentation/")
import torch

host_rec = 'localhost' 
port_rec = 1220
host_send = 'utrecht_prediction_01'
port_send = 9001

# SAM2 Configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_mask_threshold = 0.0

sam_type = "small"  

if sam_type == "large":
    overwrite_checkpoint = "./sam2.1_hiera_large.pt"
    overwrite_model_cfg= "configs/sam2.1/sam2.1_hiera_l.yaml"
elif sam_type == "small":
    overwrite_checkpoint= "./sam2_hiera_small.pt"
    overwrite_model_cfg= "sam2_hiera_s.yaml"

from sam2.build_sam import build_sam2_camera_predictor

testing = True

from scipy.ndimage import center_of_mass

def torch_center_of_mass(mask):
    mask = mask.float()
    h, w = mask.shape[-2:]
    y = torch.arange(h, device=mask.device).view(-1, 1)
    x = torch.arange(w, device=mask.device).view(1, -1)

    total = mask.sum()
    cy = (mask * y).sum() / total
    cx = (mask * x).sum() / total

    return torch.stack([cy, cx])



class ReceiveImages:
    # Init functions to set up queues, SAM, connections

    def __init__(self, image_dimensions=(112,112),send_data=False,protocol='tcp',max_queue_size=0):
        #self.seen_images = []

        

        # Asyncio queue
        self.seen_images_queue = asyncio.Queue(maxsize=max_queue_size) # can add maxsize parameter
        self.preprocessed_images_queue = asyncio.Queue(maxsize=max_queue_size)
        self.masks_queue = asyncio.Queue(maxsize=max_queue_size)
        self.coms_queue = asyncio.Queue(maxsize=max_queue_size)
        

        self.prompt = None
        self.time_taken_per_frame = []
        self.image_dimensions = image_dimensions
        self.send_data = send_data

        self.protocol = protocol

        self.frame_no = 0

        # Testing params
        self.break_point = 10000  # Set a break point after which to stop receiving images for testing purposes
        

        # SAM 2 initialization
        self.checkpoint = overwrite_checkpoint
        self.model_cfg = overwrite_model_cfg
        self.predictor = build_sam2_camera_predictor(self.model_cfg, self.checkpoint,device=device)
        self.predictor.fill_hole_area = 0
        self.predictor.multimask_output_in_sam = True
        self.downcast_dtype = torch.float16
        print(f"SAM 2 {sam_type} initialized")

        import datetime
        # logging 
        self.logging = True
        if self.logging:
            with open("/utrecht_exp/logs/receive_images_enter.txt", 'w') as f:
                f.write(f"Log file created at {datetime.datetime.now()}\n\n")
            with open("/utrecht_exp/logs/receive_images_exit.txt", 'w') as f:
                f.write(f"Log file created at {datetime.datetime.now()}\n\n")

    def connect(self, host=host_rec, port=port_rec):
        if self.protocol.lower() == 'tcp':
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.s.bind((host, port))
            self.s.listen(1)
            print("Tracking module waiting for TCP connection...")
            self.conn, self.addr = self.s.accept()
            print("Connected by", self.addr)
        elif self.protocol.lower() == 'udp':
            self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.s.bind((host, port))
            print("Tracking module waiting for UDP connection...")
            self.conn = self.s
            self.addr = (host, port)
        else:
            raise ValueError("Protocol must be 'tcp' or 'udp'")

    def connect_send(self, host=host_send, port=port_send):
        if self.send_data:
            if self.protocol.lower() == 'tcp':
                self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.send_socket.connect((host, port))
            elif self.protocol.lower() == 'udp':
                self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.send_socket.connect((host, port))
            else:
                raise ValueError("Protocol must be 'tcp' or 'udp'")
            print(f"Connected to server at {host}:{port} for sending data, using protocol {self.protocol.lower()}")
        else:
            print("send_data is False, not connecting to send socket")

    # Splitting into easier and faster small blocks for better threading and less blockage


    async def receive_images(self):
        while True:
            # Receive the size of the incoming image
            loop = asyncio.get_running_loop()
            data = await loop.sock_recv(self.conn, 4)
            if data is not None:
                start_time = time.time()
                img_size = struct.unpack('!I', data)[0]
                #print(img_size)
                # Receive the image data based on the size
                img_data = b''
                while len(img_data) < img_size:
                    packet = await loop.sock_recv(self.conn, img_size - len(img_data))
                    img_data += packet

                # Convert the byte data to a numpy array and reshape it to the original image dimensions
                img_array = np.frombuffer(img_data, dtype=np.uint16)
                # print('Received image of size:', img_array.size)
                img_array = img_array.reshape(self.image_dimensions)  # Adjust dimensions as needed
                end_time = time.time()
                #print(f"Received image of size {img_array.size} in {end_time - start_time:.4f} seconds")
                #print(f"Received image of size {img_array.size} in {end_time - start_time:.4f} seconds")
                await self.seen_images_queue.put(img_array)

                if self.logging:
                    with open("/utrecht_exp/logs/receive_images_enter.txt", 'a') as f:
                        f.write(f"Received image of size {img_array.size} for frame {self.frame_no} at {datetime.datetime.now()}\n")

                #print(f"Number of seen images: {len(self.seen_images)}")
            await asyncio.sleep(0.005)  # Sleep briefly to avoid busy waiting
                

    async def preprocess_image(self):
        while True:
            image = await self.seen_images_queue.get()

            prep_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            self.preprocessed_images_queue.put_nowait(prep_image)
            #return prep_image

    async def track_frame(self):
        while True: 
            image = await self.preprocessed_images_queue.get()
            self.frame_no += 1
            with torch.inference_mode():
                    with torch.autocast('cuda', dtype=self.downcast_dtype):
                        
                        if self.frame_no == 1: 
                            print(f"Initializing SAM {sam_type} with the first frame and prompt")

                            start_time_sam = time.time()

                            self.predictor.load_first_frame(image)

                            _, _, out_mask_logits = self.predictor.add_new_mask(frame_idx=0, obj_id=0, mask=self.prompt)

                            out_mask = out_mask_logits>sam_mask_threshold

                            self.masks_queue.put_nowait(out_mask)
                        
                            print("First frame processed, starting tracking...")
                            end_time_sam = time.time()
                            print(f"Time taken to process first frame with SAM: {end_time_sam - start_time_sam:.4f} seconds")

                        else:
                            #print("Tracking new frame...")
                            _, out_mask_logits = self.predictor.track(image)
                            out_mask = out_mask_logits>sam_mask_threshold
                            self.masks_queue.put_nowait(out_mask)


    async def postprocess_mask(self):
        while True: 
            new_mask = await self.masks_queue.get()            #print("Calculating COM from mask...")

            new_com = torch_center_of_mass(new_mask)
            self.coms_queue.put_nowait(new_com)
            #print(f"New COM: {new_com}")
            await asyncio.sleep(0.002)  # Sleep briefly to avoid busy waiting

    async def send_com(self):
        if self.send_data:
            while True:
                new_com = await self.coms_queue.get()
                if self.send_data:
                    if self.frame_no == 1:
                        print(f"Sent center of mass for frame {self.frame_no}: {new_com} at {datetime.datetime.now()}")
                        value = struct.pack('2f', new_com[0], new_com[1])  # Convert the float to bytes
                    else:
                        value = struct.pack('2f', new_com[0], new_com[1])  # Convert the float to bytes
                    self.send_socket.send(value)

                    if self.logging:
                        with open("/utrecht_exp/logs/receive_images_exit.txt", 'a') as f:
                            f.write(f"Sent center of mass for frame {self.frame_no}: {new_com} at {datetime.datetime.now()}\n")

    # One time use functions

    def initialize_prompt(self):
        if testing:
            import SimpleITK as sitk
            self.prompt = sitk.GetArrayFromImage(sitk.ReadImage("/utrecht_exp/data/prompt.mha"))[0]
            print('Initialized prompt')

    def close_connection(self):
        self.conn.close()
        self.s.close()
    
    def save_masks(self, save_dir="/utrecht_exp/segmentation/output_masks/"):
        os.makedirs(save_dir, exist_ok=True)
        for idx, mask in enumerate(self.out_masks):
            plt.figure(figsize=(5,5))
            plt.imshow(mask.squeeze().cpu().numpy().astype(np.uint8)*255, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"mask_{idx:03d}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

    
print("Initializing improved tracking module...")
async def main():
    image_receiver = ReceiveImages(send_data=True)
    image_receiver.initialize_prompt()
    image_receiver.connect(host=host_rec, port=port_rec)
    image_receiver.connect_send(host=host_send, port=port_send)
    print("Starting tracking...")
    await asyncio.gather(
        image_receiver.receive_images(),
        image_receiver.preprocess_image(),
        image_receiver.track_frame(),
        image_receiver.postprocess_mask(),
        image_receiver.send_com()
    )

await main()







