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

host = 'localhost' 
port = 1220
host_send = 'utrecht_prediction_01'
port_send = 9001

# SAM2 Configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_mask_threshold = 0.0

sam_type = "large"  

if sam_type == "large":
    overwrite_checkpoint = "./sam2.1_hiera_large.pt"
    overwrite_model_cfg= "configs/sam2.1/sam2.1_hiera_l.yaml"
elif sam_type == "small":
    overwrite_checkpoint= "./sam2_hiera_small.pt"
    overwrite_model_cfg= "sam2_hiera_s.yaml"

from sam2.build_sam import build_sam2_camera_predictor

testing = True

from scipy.ndimage import center_of_mass

class ReceiveImages:
    def __init__(self, image_dimensions=(112,112),send_data=False,protocol='tcp'):
        self.seen_images = []
        self.out_masks = []
        
        self.prompt = None
        self.time_taken_per_frame = []
        self.image_dimensions = image_dimensions
        self.center_of_mass = np.zeros(2)
        self.send_data = send_data

        self.protocol = protocol


        # Testing params
        self.break_point = 10000  # Set a break point after which to stop receiving images for testing purposes
        

        # SAM 2 initialization
        self.checkpoint = overwrite_checkpoint
        self.model_cfg = overwrite_model_cfg
        self.predictor = build_sam2_camera_predictor(self.model_cfg, self.checkpoint,device=device)
        self.predictor.fill_hole_area = 0
        self.predictor.multimask_output_in_sam = True
        print(f"SAM 2 {sam_type} initialized")

        import datetime
        # logging 
        self.logging = True
        with open("/utrecht_exp/logs/receive_images_enter.txt", 'w') as f:
            f.write(f"Log file created at {datetime.datetime.now()}\n\n")
        with open("/utrecht_exp/logs/receive_images_exit.txt", 'w') as f:
            f.write(f"Log file created at {datetime.datetime.now()}\n\n")

    def connect(self, host=host, port=port):
        if self.protocol.lower() == 'tcp':
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.s.bind((host, port))
            self.s.listen(1)
            print("Segmentation server waiting for TCP connection...")
            self.conn, self.addr = self.s.accept()
            print("Connected by", self.addr)
        elif self.protocol.lower() == 'udp':
            self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.s.bind((host, port))
            print("Segmentation server waiting for UDP connection...")
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


    async def receive_images(self):
        while True:
            # Receive the size of the incoming image
            data = self.conn.recv(4)
            if data is not None:
                start_time = time.time()
                img_size = struct.unpack('!I', data)[0]
                #print(img_size)
                # Receive the image data based on the size
                img_data = b''
                while len(img_data) < img_size:
                    packet = self.conn.recv(img_size - len(img_data))
                    img_data += packet

                # Convert the byte data to a numpy array and reshape it to the original image dimensions
                img_array = np.frombuffer(img_data, dtype=np.uint16)
                # print('Received image of size:', img_array.size)
                img_array = img_array.reshape(self.image_dimensions)  # Adjust dimensions as needed
                end_time = time.time()
                #print(f"Received image of size {img_array.size} in {end_time - start_time:.4f} seconds")
                #print(f"Received image of size {img_array.size} in {end_time - start_time:.4f} seconds")
                self.seen_images.append(img_array)
                #print(f"Number of seen images: {len(self.seen_images)}")
            await asyncio.sleep(0.01)  # Sleep briefly to avoid busy waiting
                

    async def track_frames(self):
        print("Starting to track frames...")
        while True:
            #print(len(self.seen_images))
            if len(self.seen_images) == 0:
                await asyncio.sleep(0.02)  # Sleep briefly to avoid busy waiting
                
                continue
                
            if len(self.seen_images) != len(self.out_masks):
                new_image = self.seen_images[-1]

                if self.logging:
                    with open("/utrecht_exp/logs/receive_images_enter.txt", 'a') as f:
                        f.write(f"Received image at {datetime.datetime.now()}\n")
                start_time = time.time()
                with torch.inference_mode():
                    with torch.autocast('cuda', dtype=torch.float16):
                        #print(f"Received image {len(self.seen_images)}")
                        if len(self.seen_images) == 1: 
                            print(f"Initializing SAM {sam_type} with the first frame and prompt")
                            print(f'Seen images: {len(self.seen_images)}') 
                            print(f'Out masks: {len(self.out_masks)}')

                            start_time_sam = time.time()
                            image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB)
                            self.predictor.load_first_frame(image)

                            _, _, out_mask_logits = self.predictor.add_new_mask(frame_idx=0, obj_id=0, mask=self.prompt)

                            out_mask = out_mask_logits>sam_mask_threshold
                            self.out_masks.append(out_mask)
                            self.center_of_mass = np.array(center_of_mass(out_mask.squeeze().cpu().numpy()))[None,:]
                            print(f"Initial center of mass: {self.center_of_mass}"
                                  f"\nInitial mask shape: {self.center_of_mass.shape}"
                                  f"\nInitial mask len: {len(self.out_masks)}")
                            print("First frame processed, starting tracking...")
                            end_time_sam = time.time()
                            print(f"Time taken to process first frame with SAM: {end_time_sam - start_time_sam:.4f} seconds")
                        elif len(self.seen_images) > 1:
                            #print(f"Tracking new frame {len(self.seen_images)}")
                            image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB)
                            _, out_mask_logits = self.predictor.track(image)
                            out_mask = out_mask_logits>sam_mask_threshold
                            self.out_masks.append(out_mask)
                            #print(np.sum(out_mask.cpu().numpy()))
                            if center_of_mass(out_mask.squeeze().cpu().numpy()) == (0.0, 0.0):
                                print(f"Warning: Center of mass for frame {len(self.seen_images)} is at origin")
                            self.center_of_mass = np.concatenate((self.center_of_mass, np.array(center_of_mass(out_mask.squeeze().cpu().numpy()))[None,:]), axis=0)

                            #print(f"Processed frame {len(self.seen_images)}")
                            #print(self.
                            # center_of_mass.shape)

                        if self.send_data:
                            if len(self.seen_images) == 1:
                                print(f"Sent center of mass for frame {len(self.seen_images)}: {self.center_of_mass[-1]}")
                                print(f"Center of mass shape: {self.center_of_mass.shape}")
                                value = struct.pack('2f', self.center_of_mass[0,0], self.center_of_mass[0,1])  # Convert the float to bytes
                            else:
                                value = struct.pack('2f', self.center_of_mass[-1, 0], self.center_of_mass[-1, 1])  # Convert the float to bytes
                            
                            if self.logging:
                                with open("/utrecht_exp/logs/receive_images_exit.txt", 'a') as f:
                                    f.write(f"Sent center of mass for frame {len(self.seen_images)}: {self.center_of_mass[-1]} at {datetime.datetime.now()}\n")
                            
                            self.send_socket.send(value)

                end_time = time.time()
                #print(f"Time taken to process frame {len(self.seen_images)}: {end_time - start_time:.4f} seconds")
                self.time_taken_per_frame.append(end_time - start_time)

            else:
                await asyncio.sleep(0.02)  # Sleep briefly to avoid busy waiting

            if testing and len(self.seen_images) >= self.break_point:
                print("Testing break point reached, stopping tracking.")
                #self.save_masks()
                print(self.center_of_mass)
                print(self.center_of_mass.shape)
                break

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

    
print("Initializing Tracking module...")
async def main():
    image_receiver = ReceiveImages(send_data=True)
    image_receiver.initialize_prompt()
    image_receiver.connect(host=host, port=port)
    image_receiver.connect_send(host=host_send, port=port_send)
    print("Starting tracking...")
    await asyncio.gather(
        image_receiver.receive_images(),
        image_receiver.track_frames())

await main()







