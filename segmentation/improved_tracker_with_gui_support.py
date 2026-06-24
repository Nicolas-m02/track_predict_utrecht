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
import math
os.chdir("/utrecht_exp/segmentation/")
import torch

host_rec = '0.0.0.0' 
port_rec = 6056
host_send = 'prediction_container'
port_send = 9002

# Gui configs
host_gui = 'gui_container_local' # online testing
# host_gui ='gui_container' # offline testing
port_gui = 7000

#MRTC Receiver configs
mrtc_port = 4005 # receiving images from MR
stack_update_host = '0.0.0.0'
stack_update_port = 54323   # controlling the MR


# SAM2 Configs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sam_mask_threshold = 0.0

sam_type = "small"  

if sam_type == "large":
    overwrite_checkpoint = "./sam2.1_hiera_large.pt"
    overwrite_model_cfg= "configs/sam2.1/sam2.1_hiera_l.yaml"
elif sam_type == "small":
    overwrite_checkpoint= "./sam2_hiera_small.pt"
    overwrite_model_cfg= "sam2_hiera_s.yaml"
elif sam_type == "tiny":
    overwrite_checkpoint= "./sam2_hiera_tiny.pt"
    overwrite_model_cfg= "sam2_hiera_t.yaml"

from sam2.build_sam import build_sam2_camera_predictor

testing = False

from scipy.ndimage import center_of_mass

def torch_center_of_mass(mask, img_size_px, voxel_size_mm):
    mask = mask.float()
    h, w = mask.shape[-2:]
    y = torch.arange(h, device=mask.device).view(-1, 1)
    x = torch.arange(w, device=mask.device).view(1, -1)

    total = mask.sum()
    cy = (mask * y).sum() / total
    cx = (mask * x).sum() / total

    dy_px = cy - img_size_px // 2
    dx_px = cx - img_size_px // 2

    dx_mm = dx_px * voxel_size_mm
    dy_mm = - dy_px * voxel_size_mm      # changed direction to comply with motion of phantom


    return torch.stack([dx_mm, dy_mm])



class ReceiveImages:
    # Init functions to set up queues, SAM, connections

    def __init__(self, image_dimensions=(128,128),send_data=False,protocol='tcp',max_queue_size=0,send_timestamps=False):
        #self.seen_images = []

         # Receiving data params
        self.zmq_prot = True
        self.emulation = False
        #self.emu_path = "/utrecht_exp/data/all_dat_files/small_dat_files"
        self.emu_path = "/utrecht_data/20260616/dat_imgs/"

        # Asyncio queue
        self.seen_images_queue = asyncio.Queue(maxsize=max_queue_size) # can add maxsize parameter
        self.preprocessed_images_queue = asyncio.Queue(maxsize=max_queue_size)
        self.masks_queue = asyncio.Queue(maxsize=max_queue_size)
        self.coms_queue = asyncio.Queue(maxsize=max_queue_size)
        self.gui_queue = asyncio.Queue(maxsize=max_queue_size)

        self.prompt_library = {}
        self.current_angle = None
        self.last_angle = 0

        self.prompt = None
        self.time_taken_per_frame = []
        self.image_dimensions = image_dimensions
        self.send_data = send_data

        self.send_timestamps = send_timestamps
        self.protocol = protocol

        self.MRTC_prot = True
        self.mrtc_port = mrtc_port 
        self.stack_update_host = stack_update_host
        self.stack_update_port = stack_update_port   

        self.voxel_size_mm = 1.95
        self.img_size_px = image_dimensions[0]

        self.frame_no = 0
        

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
            self.time_logging = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            self.receive_images_enter_log = (f"/utrecht_exp/logs/receive_images_enter_{self.time_logging}.txt")
            self.receive_images_exit_log = (f"/utrecht_exp/logs/receive_images_exit_{self.time_logging}.txt")

            with open(self.receive_images_enter_log, "w") as f:
                f.write(f"Log file created at {datetime.datetime.now()}\n\n")

            with open(self.receive_images_exit_log, "w") as f:
                f.write(f"Log file created at {datetime.datetime.now()}\n\n")

    def connect(self, host=host_rec, port=port_rec):
        if self.zmq_prot and not self.MRTC_prot and not self.emulation:
            import zmq
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.bind(f"tcp://{host}:{port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            print(f"Tracking module waiting for ZMQ connection on {host}:{port}...")
            self.conn = self.socket

        elif self.zmq_prot and self.MRTC_prot and not self.emulation:
            import pymri
            self.handler = pymri.QueuedImageHandler()
            print("ZMQ protocol enabled, setting up ZMQ image receiver")
            print("port is ", self.mrtc_port)
            self.recv = pymri.MRTCImageReceiver.create(self.mrtc_port, self.stack_update_host, self.stack_update_port, self.handler, False) 
            
        elif self.emulation:
            print("Emulation mode enabled, not setting up actual socket connection")
            import pymri
            self.handler = pymri.QueuedImageHandler()
            self.recv = pymri.EmuImageReceiver.create(self.emu_path, self.handler)

        else:
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
                self.send_socket.bind((host, port))
            else:
                raise ValueError("Protocol must be 'tcp' or 'udp'")
            print(f"Connected to server at {host}:{port} for sending data, using protocol {self.protocol.lower()}")
        else:
            print("send_data is False, not connecting to send socket")

    def connect_to_gui(self, host=host_gui, port=port_gui):
        self.gui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gui_socket.connect((host, port))
        print(f"Connected to GUI at {host}:{port}")


    # Splitting into easier and faster small blocks for better threading and less blockage


    async def receive_images(self):
        if self.zmq_prot and not self.MRTC_prot and not self.emulation:
            while True:

                # receive message
                msg = self.conn.recv()
                # find beginning of binary image data
                sep = b"DATA\n"

                idx = msg.find(sep)

                if idx == -1:
                    print("no DATA separator found")
                    continue

                # split header and binary payload
                
                
                header = msg[:idx].decode("latin-1")  # use latin-1 to preserve byte values
                #print(len(header), "bytes of header")
                #print("header:", header)
                if header.count("\n") < 3:
                    print("header does not contain enough lines, skipping message")
                    continue
                

                meta = {}

                for line in header.splitlines():

                    parts = line.strip().split()

                    if len(parts) == 0:
                        continue

                    key = parts[0]
                    #print("key:", key)
                    if "timestamp" in key:
                        meta["timestamp"] = int(parts[1])
                    elif key == "dim":
                        meta["dim"] = [int(x) for x in parts[1:]]
                    elif key == "fov":
                        meta["fov"] = [float(x) for x in parts[1:]]
                    elif key == "resolution":
                        meta["resolution"] = [float(x) for x in parts[1:]]
                    elif key == "row_direction_cosines":
                        meta["row_direction_cosines"] = [float(x) for x in parts[1:]]
                        self.current_angle = int(np.round(math.degrees(math.atan2(meta["row_direction_cosines"][1], meta["row_direction_cosines"][0]))))
                        print(f"Current angle: {self.current_angle}")
                raw = msg[idx + len(sep):]

                arr = np.frombuffer(raw, dtype=np.float32)

                img_array = arr.reshape(meta["dim"])

                if self.send_timestamps:
                    await self.seen_images_queue.put((img_array, meta["timestamp"]))
                else:
                    await self.seen_images_queue.put(img_array)

                if self.logging:
                    with open(f"/utrecht_exp/logs/receive_images_enter{self.time_logging}.txt", 'a') as f:
                        f.write(f"Received image of size {img_array.size} for frame {self.frame_no} at {datetime.datetime.now()}\n")

                await asyncio.sleep(0.005)  # Sleep briefly to avoid busy waiting

        elif self.emulation or (self.zmq_prot and self.MRTC_prot):
            while True:

                image = self.handler.get_image()

                if image is not None:

                    self.current_angle = int(np.round(math.degrees(math.atan2(image['row_direction_cosines'][1], image['row_direction_cosines'][0]))))
                    print(f"Current angle: {self.current_angle}")
                    if self.send_timestamps:
                        await self.seen_images_queue.put((image['data'], image['timestamp']))
                    else:
                        await self.seen_images_queue.put(image['data'])
                    #print(f"Received image of size {image['data'].size} for frame {self.frame_no} at {datetime.datetime.now()}")
                    if self.logging:
                        with open(f"/utrecht_exp/logs/receive_images_enter{self.time_logging}.txt", 'a') as f:
                            f.write(f"Received image of size {image['data'].size} for frame {self.frame_no} at {datetime.datetime.now()}\n")
                await asyncio.sleep(0.002)  # Sleep briefly to avoid busy waiting

        else:
            while True:
                # Receive the size of the incoming image
                loop = asyncio.get_running_loop()
                data = await loop.sock_recv(self.conn, 4)
                if data is not None:
                    #print(f"Received data for frame {self.frame_no + 1} at {datetime.datetime.now()}")
                    start_time = time.time()
                    img_size = struct.unpack('!I', data)[0]
                    #print(img_size)
                    # Receive the image data based on the size
                    img_data = b''
                    while len(img_data) < img_size:
                        packet = await loop.sock_recv(self.conn, img_size - len(img_data))
                        img_data += packet

                    # Convert the byte data to a numpy array and reshape it to the original image dimensions
                    img_array = np.frombuffer(img_data, dtype=np.float32)
                    # print('Received image of size:', img_array.size)
                    img_array = img_array.reshape(self.image_dimensions)  # Adjust dimensions as needed
                    
                    img_array = img_array.astype(np.uint16)  # Convert to uint8 for OpenCV processing
                    end_time = time.time()
                    #print(f"Received image of size {img_array.size} in {end_time - start_time:.4f} seconds")
                    #print(f"Received image of size {img_array.size} in {end_time - start_time:.4f} seconds")
                    await self.seen_images_queue.put(img_array)

                    if self.logging:
                        with open(f"/utrecht_exp/logs/receive_images_enter{self.time_logging}.txt", 'a') as f:
                            f.write(f"Received image of size {img_array.size} for frame {self.frame_no} at {datetime.datetime.now()}\n")
                #print(f"Number of seen images: {len(self.seen_images)}")
                await asyncio.sleep(0.005)  # Sleep briefly to avoid busy waiting
                

    async def preprocess_image(self):
        while True:
            if self.send_timestamps:
                image, timestamp = await self.seen_images_queue.get()
            else:
                image = await self.seen_images_queue.get()
            prep_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            if self.send_timestamps:
                self.preprocessed_images_queue.put_nowait((prep_image, timestamp))
            else:
                self.preprocessed_images_queue.put_nowait(prep_image)
            #return prep_image

    async def track_frame(self):
        while True: 
            if self.send_timestamps:
                image, timestamp = await self.preprocessed_images_queue.get()
            else:
                image = await self.preprocessed_images_queue.get()
            self.frame_no += 1

            with torch.inference_mode():
                    with torch.autocast('cuda', dtype=self.downcast_dtype):
                        
                        if self.frame_no == 1 or self.current_angle != self.last_angle: 
                            if self.frame_no == 1:
                                print(f"Initializing SAM {sam_type} with the first frame and prompt")

                            else: 
                                print(f"Angle change detected (current: {self.current_angle}, last: {self.last_angle}), reinitializing SAM {sam_type} with new prompt")
                            start_time_sam = time.time()

                            self.predictor.load_first_frame(image)

                            if testing:
                                _, _, out_mask_logits = self.predictor.add_new_mask(frame_idx=0, obj_id=0, mask=self.prompt)

                            else: 
                                # Find mask with specified angle 
                                #print(self.prompt_library.keys())
                                #print(self.prompt_library[str(self.current_angle)].shape)
                                if str(self.current_angle) not in self.prompt_library:
                                    print(self.prompt_library.keys())
                                    print(f"Current angle: {self.current_angle}")
                                    print(f"Last angle: {self.last_angle}")
                                    print(f"No prompt found for angle {self.current_angle}, using default prompt")
                                    _, _, out_mask_logits = self.predictor.add_new_mask(frame_idx=0, obj_id=0,mask= self.prompt_library[str(self.last_angle)])
                                else:
                                    print(f"Using prompt for angle {self.current_angle}")
                                    _, _, out_mask_logits = self.predictor.add_new_mask(frame_idx=0, obj_id=0,mask= self.prompt_library[str(self.current_angle)])
                                self.last_angle = self.current_angle



                            out_mask = out_mask_logits>sam_mask_threshold

                            if self.send_timestamps:
                                self.masks_queue.put_nowait((out_mask, timestamp))
                            else:
                                self.masks_queue.put_nowait(out_mask)

                            print("First frame processed, starting tracking...")
                            end_time_sam = time.time()
                            print(f"Time taken to process angle change with SAM: {end_time_sam - start_time_sam:.4f} seconds")
                            self.gui_queue.put_nowait((image, out_mask))
                        else:
                            #print("Tracking new frame...")
                            _, out_mask_logits = self.predictor.track(image)
                            out_mask = out_mask_logits>sam_mask_threshold
                            if self.send_timestamps:
                                self.masks_queue.put_nowait((out_mask, timestamp))
                            else:
                                self.masks_queue.put_nowait(out_mask)
                            self.gui_queue.put_nowait((image, out_mask))


    async def postprocess_mask(self):
        while True: 
            if self.send_timestamps:
                new_mask, timestamp = await self.masks_queue.get()
            else:
                new_mask = await self.masks_queue.get()

            new_com = torch_center_of_mass(new_mask, self.img_size_px, self.voxel_size_mm)
            
            if self.send_timestamps:
                self.coms_queue.put_nowait((new_com, timestamp))
            else:
                self.coms_queue.put_nowait(new_com)
            #print(f"New COM: {new_com}")
            await asyncio.sleep(0.002)  # Sleep briefly to avoid busy waiting

    async def send_com(self):
        if self.send_data:
            while True:
                if self.send_timestamps:
                    new_com, tstamp_send = await self.coms_queue.get()
                else:
                    new_com = await self.coms_queue.get()


                if self.send_data:
                    if self.frame_no == 1:
                        print(f"Sent center of mass for frame {self.frame_no}: {new_com} at {datetime.datetime.now()}")
                        if self.send_timestamps:
                            #print(f"Timestamp sent: {tstamp_send/1e9} seconds")
                            value = struct.pack('2fQ', new_com[0], new_com[1], tstamp_send)  # Convert the float and timestamp to bytes
                        else:
                            value = struct.pack('2f', new_com[0], new_com[1])  # Convert the float to bytes
                    else:
                        if self.send_timestamps:
                            #print(f"Timestamp sent: {tstamp_send/1e9} seconds")
                            value = struct.pack('2fQ', new_com[0], new_com[1], tstamp_send)  # Convert the float and timestamp to bytes
                        else:
                            value = struct.pack('2f', new_com[0], new_com[1])  # Convert the float to bytes
                    self.send_socket.send(value)

                    if self.logging:
                        with open(f"/utrecht_exp/logs/receive_images_exit{self.time_logging}.txt", 'a') as f:
                            f.write(f"Sent center of mass for frame {self.frame_no}: {new_com} at {datetime.datetime.now()}\n")
                await asyncio.sleep(0.002)  # Sleep briefly to avoid busy waiting
    # GUI main function 

    async def send_im_and_mask_to_gui(self):
        def compute_largest_contour(binary_mask):
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            largest_contour = max(contours, key=lambda x: len(x))
            return np.array(largest_contour)[:,0,:]

        while True:
            
            image, mask = await self.gui_queue.get()

            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            #print(image.shape, mask.shape)
            #print(image.dtype, mask.dtype)
            #print(np.max(mask.cpu().numpy()), np.min(mask.cpu().numpy()), mask.cpu().numpy().shape)

            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            #print(mask.cpu().numpy().squeeze().astype(np.uint8).shape, np.unique(mask.cpu().numpy().astype(np.uint8)))
            receiving_time_end = time.time()
            #print(f"Received image and mask from queues in {receiving_time_end - receiving_time_start:.4f} seconds")
            try:
                contour_time_start = time.time()
                image = cv2.drawContours(image.copy(), [compute_largest_contour(mask.cpu().numpy().squeeze().astype(np.uint8)*255)], -1, (255,255,255), 1)
                contour_time_end = time.time()
                #print(f"Time taken to compute largest contour: {contour_time_end - contour_time_start:.4f} seconds")
            except Exception as e:
                print(f"Error occurred while drawing contours: {e}")
                pass
            # Convert the image and mask to bytes

            #print(image.dtype, image.shape)

            #_, img_encoded = cv2.imencode('.jpg', image)
            #sending_time_start = time.time()
            img_bytes = image.tobytes()

            #print(img_encoded.dtype, img_encoded.shape)

            # send image 
            img_size = len(img_bytes)
            self.gui_socket.send(struct.pack('!I', img_size))  # Send the size of the image first
            self.gui_socket.send(img_bytes)  # Then send the image data

            #sending_time_end = time.time()
            #print(f"Sent image to GUI in {sending_time_end - sending_time_start:.4f} seconds")
            end_time = time.time()
            #print(f"Sent image and mask to GUI in {end_time - start_time:.4f} seconds")
            
            with open(f"/utrecht_exp/logs/gui_sent{self.time_logging}.txt", 'a') as f:
                f.write(f"Sent image and mask for frame {self.frame_no} to GUI at {datetime.datetime.now()}\n")

    # One time use functions

    def initialize_prompt(self, prompt_library_path=None):
        if testing:
            import SimpleITK as sitk
            self.prompt = sitk.GetArrayFromImage(sitk.ReadImage("/utrecht_exp/data/prompt.mha"))[0]
            print('Initialized prompt')

        elif prompt_library_path is not None:
            import SimpleITK as sitk
            for file in os.listdir(prompt_library_path):
                if file.endswith(".mha"):
                    #print(file.split(".")[0].split("_")[-1])
                    prompt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(prompt_library_path, file)))[0]
                    #print(prompt.shape)

                    self.prompt_library[file.split(".")[0].split("_")[-1]] = prompt

                    #print(f"Initialized prompt from {file}")
            print(self.prompt_library.keys())
            print(f"Initialized {len(self.prompt_library)} prompts from library")



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

    
print("Initializing improved tracking module with gui support...")
async def main():
    image_receiver = ReceiveImages(send_data=True,image_dimensions=(128,128),send_timestamps=True)
    image_receiver.initialize_prompt(prompt_library_path="/utrecht_exp/segmentation/prompt_library/circle_2206/")
    image_receiver.connect_send(host=host_send, port=port_send)
    image_receiver.connect_to_gui(host=host_gui, port=port_gui)
    image_receiver.connect(host=host_rec, port=port_rec)

    print("Starting tracking...")
    await asyncio.gather(
        image_receiver.receive_images(),
        image_receiver.preprocess_image(),
        image_receiver.track_frame(),
        image_receiver.postprocess_mask(),
        image_receiver.send_com(),
        image_receiver.send_im_and_mask_to_gui()
    )

asyncio.run(main())

