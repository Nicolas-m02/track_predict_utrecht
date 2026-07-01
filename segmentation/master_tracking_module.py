#%%

import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
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
import math
import sys
import argparse

parser = argparse.ArgumentParser(description='Master Tracking Module')
parser.add_argument('--t_logging', type=str, default='TESTING', help='Timestamp for logging')
args = parser.parse_args()

sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))
from socket_experiments import PositionServer_pb2 as ps


with open("/utrecht_exp/config.yaml", 'r') as f:
    import yaml
    config = yaml.safe_load(f)

#host_rec = 'localhost' 
#port_rec = 1220
host_rec = '0.0.0.0' 
port_rec = 6056






if config['tracker']['connect_to_external_predictor']:
    host_send = config['ports']['host_send_com_to_extern']
    port_send = config['ports']['port_send_com_to_extern']
else:
    host_send = config['ports']['host_send_com']
    port_send = config['ports']['port_send_com']






# Gui configs
host_gui = config['ports']['host_gui']
port_gui = config['ports']['port_gui_images']

host_clicks = config['ports']['host_gui']
port_clicks = config['ports']['port_gui_clicks']

#MRTC Receiver configs
mrtc_port = config['ports']['mrtc_port'] # receiving images from MR
stack_update_host = config['ports']['stack_update_host']
stack_update_port = config['ports']['stack_update_port']   # controlling the MR


# SAM2 Configs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sam_mask_threshold = config['tracker']['sam_mask_threshold']

sam_type = config['tracker']['sam_type'] 

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

def torch_center_of_mass(mask, img_size_px, interpol, voxel_size_mm):
    mask = mask.float()
    h, w = mask.shape[-2:]
    y = torch.arange(h, device=mask.device).view(-1, 1)
    x = torch.arange(w, device=mask.device).view(1, -1)

    total = mask.sum()
    cy = (mask * y).sum() / total
    cx = (mask * x).sum() / total

    dy_px = cy/interpol - img_size_px/interpol // 2
    dx_px = cx/interpol - img_size_px/interpol // 2

    dx_mm = dx_px * voxel_size_mm
    dy_mm = - dy_px * voxel_size_mm      # changed direction to comply with motion of phantom


    return torch.stack([dx_mm, dy_mm])



class ReceiveImages:
    # Init functions to set up queues, SAM, connections

    def __init__(self, image_dimensions=(128,128),send_data=False,protocol='tcp',max_queue_size=1000,send_timestamps=False):
        #self.seen_images = []

         # Receiving data params
        self.zmq_prot = config['settings']['ZMQ']
        self.emulation = config['settings']['emulation']
        self.emu_path = config['settings']['emu_path']

        self.image_dimensions = image_dimensions

        # Click params
        self.click_received = False
        self.click_coordinates = None
        self.mask_received = False
        self.mask_data = None
        
        # interpolation to upscale images
        self.interpolation_scale = 1

        # Adding shift 
        self.shift = (0,0)


        # Asyncio queue
        self.seen_images_queue = asyncio.Queue(maxsize=max_queue_size) # can add maxsize parameter
        self.preprocessed_images_queue = asyncio.Queue(maxsize=max_queue_size)
        self.masks_queue = asyncio.Queue(maxsize=max_queue_size)
        self.coms_queue = asyncio.Queue(maxsize=max_queue_size)
        self.gui_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Non interactive mode
        self.prompt_library = {}
        self.current_angle = None
        self.last_angle = 0

        # Testing params
        self.prompt = None
        self.time_taken_per_frame = []
        self.send_data = send_data

        self.send_timestamps = send_timestamps
        self.protocol = protocol

        self.MRTC_prot = config['settings']['MRTC']
        self.mrtc_port = mrtc_port 
        self.stack_update_host = stack_update_host
        self.stack_update_port = stack_update_port   

        self.voxel_size_mm = config['tracker']['px_size']
        self.img_size_px = image_dimensions[0]
        self.image_dimensions = image_dimensions

        self.frame_no = 0
        self.out_masks = []  # List to store output masks for later saving

        # SAM 2 initialization
        self.checkpoint = overwrite_checkpoint
        self.model_cfg = overwrite_model_cfg
        self.predictor = build_sam2_camera_predictor(self.model_cfg, self.checkpoint,device=device)
        self.predictor.fill_hole_area = 0
        self.predictor.multimask_output_in_sam = True
        self.downcast_dtype = torch.float16
        print(f"SAM 2 {sam_type} initialized")

        # logging 
        LOG_DIR_POINT_TRACKING = os.path.join(config["logging"]["folder"],args.t_logging, "tracking_module")
        
        os.makedirs(LOG_DIR_POINT_TRACKING, exist_ok=True)

        now = datetime.datetime.now(ZoneInfo("Europe/Amsterdam"))
        ts = now.strftime("%Y%m%dT%H%M%S.%f")

        if config["logging"]["sam_log"] or config["logging"]["debug"]:

            LOG_FILE_PATH_EXIT = os.path.join(LOG_DIR_POINT_TRACKING, f"mri{ts}.txt")
            self.receive_images_exit_log = (LOG_FILE_PATH_EXIT)
            with open(self.receive_images_exit_log, "w") as f:
                f.write(f"Log file created at {ts}\n\n")

        if config["logging"]["debug"]:

            LOG_FILE_PATH_ENTER = os.path.join(LOG_DIR_POINT_TRACKING, f"receive_images_enter_{ts}.txt")
            LOG_FILE_PATH_GUISENT = os.path.join(LOG_DIR_POINT_TRACKING, f"gui_sent_{ts}.txt")

            self.receive_images_enter_log = (LOG_FILE_PATH_ENTER)
            self.gui_sent_log = (LOG_FILE_PATH_GUISENT)

            with open(self.receive_images_enter_log, "w") as f:
                f.write(f"Log file created at {ts}\n\n")

            with open(self.gui_sent_log, "w") as f:
                f.write(f"Log file created at {ts}\n\n")


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
        if config['tracker']['connect_to_external_predictor']:
            import zmq
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://{host}:{port}")
            print(f"Connecting to atlantictracking on {host}:{port}...")
            print(self.socket.getsockopt(zmq.LAST_ENDPOINT))
            self.conn_send = self.socket
        else:
            print(f"Connecting to prediction container on {host}:{port}...")
            if self.protocol.lower() == 'tcp':
                self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.send_socket.connect((host, port))
            elif self.protocol.lower() == 'udp':
                self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.send_socket.bind((host, port))
            else:
                raise ValueError("Protocol must be 'tcp' or 'udp'")
            print(f"Connected to server at {host}:{port} for sending data, using protocol {self.protocol.lower()}")

    def connect_to_gui(self, host=host_gui, port=port_gui):
        if config["settings"]["enable_gui"]:
            self.gui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.gui_socket.connect((host, port))
            print(f"Connected to GUI at {host}:{port}")


    def connect_clicks(self, host=host_clicks, port=port_clicks):
        if config["settings"]["enable_gui"] and config["settings"]["interactive"]:
            self.clicks_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.clicks_socket.connect((host, port))
            self.clicks_socket.setblocking(False)
            print(f"Connected to GUI for clicks at {host}:{port}")

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

                if config["logging"]["debug"]:
                    now = datetime.datetime.now(ZoneInfo("Europe/Amsterdam"))
                    ts = now.strftime("%Y%m%dT%H%M%S.%f")
                    with open(self.receive_images_enter_log, 'a') as f:
                        f.write(f"Received image of size {img_array.size} for frame {self.frame_no} at {ts}\n")

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
                    if config["logging"]["debug"]:
                        now = datetime.datetime.now(ZoneInfo("Europe/Amsterdam"))
                        ts = now.strftime("%Y%m%dT%H%M%S.%f")
                        with open(self.receive_images_enter_log, 'a') as f:
                            f.write(f"Received image of size {image['data'].size} for frame {self.frame_no} at {ts}\n")
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
                    await self.seen_images_queue.put(img_array)

                    if config["logging"]["debug"]:
                        now = datetime.datetime.now(ZoneInfo("Europe/Amsterdam"))
                        ts = now.strftime("%Y%m%dT%H%M%S.%f")
                        with open(self.receive_images_enter_log, 'a') as f:
                            f.write(f"Received image of size {img_array.size} for frame {self.frame_no} at {ts}\n")
                #print(f"Number of seen images: {len(self.seen_images)}")
                await asyncio.sleep(0.005)  # Sleep briefly to avoid busy waiting
                

    async def preprocess_image(self):
        while True:
            if self.send_timestamps:
                image, timestamp = await self.seen_images_queue.get()
            else:
                image = await self.seen_images_queue.get()

            
            prep_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            prep_image = cv2.resize(
                prep_image,
                None,
                fx=self.interpolation_scale,
                fy=self.interpolation_scale,
                interpolation=cv2.INTER_LINEAR
            )
            if self.send_timestamps:
                self.preprocessed_images_queue.put_nowait((prep_image, timestamp))
            else:
                self.preprocessed_images_queue.put_nowait(prep_image)
            #return prep_image

    async def track_frame(self):
        def compute_largest_contour(binary_mask):
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            largest_contour = max(contours, key=lambda x: len(x))
            return np.array(largest_contour)[:,0,:]
        while True: 
            if self.send_timestamps:
                image, timestamp = await self.preprocessed_images_queue.get()
            else:
                image = await self.preprocessed_images_queue.get()  

            if (self.click_received or self.mask_received) or self.frame_no != 0 or not config['settings']['interactive']:
                self.frame_no += 1

                with torch.inference_mode():
                    with torch.autocast('cuda', dtype=self.downcast_dtype):
                        if self.frame_no == 1 or (self.click_received or self.mask_received): 

                            start_time_sam = time.time()

                            self.predictor.load_first_frame(image)

                            # Interactive
                            if config['settings']['interactive']:
                                print(f"Initializing SAM {sam_type} in interactive mode")

                                if self.click_received:
                                    _,_, out_mask_logits = self.predictor.add_new_points(frame_idx=0, obj_id=0, points=np.array([self.click_coordinates]), labels=np.array([1]))
                                    self.click_received = False  # Reset click received flag after processing
                                    self.shift = torch_center_of_mass(out_mask_logits>sam_mask_threshold, self.img_size_px*self.interpolation_scale, self.interpolation_scale, self.voxel_size_mm)
                                    print(self.shift)
                                    print('Point prompt initialized')
                                elif self.mask_received:
                                    _,_, out_mask_logits = self.predictor.add_new_mask(frame_idx=0, obj_id=0, mask=self.mask_data)
                                    self.mask_received = False  # Reset mask received flag after processing
                                    self.shift = torch_center_of_mass(out_mask_logits>sam_mask_threshold, self.img_size_px*self.interpolation_scale, self.interpolation_scale, self.voxel_size_mm)
                                    
                                    print(self.shift)
                                    print('Mask prompt initialized')
                            
                            # Prompt library mode
                            if (self.frame_no == 1 or (self.current_angle != self.last_angle)) and not config['settings']['interactive']:
                                if self.frame_no == 1:
                                    print(f"Initializing SAM {sam_type} with the first frame and prompt")
                                else: 
                                    print(f"Angle change detected (current: {self.current_angle}, last: {self.last_angle}), reinitializing SAM {sam_type} with new prompt")

                                if testing:
                                    print('Testing mode, default prompt used')
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
                                    self.shift = torch_center_of_mass(out_mask_logits>sam_mask_threshold, self.img_size_px*self.interpolation_scale, self.interpolation_scale, self.voxel_size_mm)

                            out_mask = out_mask_logits>sam_mask_threshold
                            self.out_masks.append(out_mask)  # Store the first mask for later saving
                            
                            if self.send_timestamps:
                                self.masks_queue.put_nowait((out_mask,timestamp))
                            else:
                                self.masks_queue.put_nowait(out_mask)                            
                            
                            prompt_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                            LOG_DIR_POINT_TRACKING = os.path.join(config["logging"]["folder"],args.t_logging, "tracking_module")

                            prompt_image = cv2.normalize(prompt_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            prompt_image = cv2.drawContours(prompt_image.copy(), [compute_largest_contour(out_mask.cpu().numpy().squeeze().astype(np.uint8)*255)], -1, (255,255,255), 1)                                    
                            im_save_path = os.path.join(LOG_DIR_POINT_TRACKING, f"overlay_{self.frame_no:03d}.png")

                            cv2.imwrite(im_save_path, prompt_image)

                            print("First frame processed, starting tracking...")
                            end_time_sam = time.time()
                            print(f"Time taken to process first frame with SAM: {end_time_sam - start_time_sam:.4f} seconds")
                            if config["settings"]["enable_gui"]:
                                self.gui_queue.put_nowait((image, out_mask))
                        else:
                            print('Tracking new mask')
                            _, out_mask_logits = self.predictor.track(image)
                            
                            out_mask = out_mask_logits>sam_mask_threshold
                            
                            self.out_masks.append(out_mask)  # Store the first mask for later saving
                            
                            if self.send_timestamps:
                                self.masks_queue.put_nowait((out_mask,timestamp))
                            else:
                                self.masks_queue.put_nowait(out_mask)
                            if config["settings"]["enable_gui"]:
                                self.gui_queue.put_nowait((image, out_mask))

            else:
                # Send empty mask to GUI if no click received
                if config["settings"]["enable_gui"]:
                    self.gui_queue.put_nowait((image, torch.zeros(image.shape[:2], dtype=bool)))  

            await asyncio.sleep(0.002)  # Sleep briefly to avoid busy waiting

    async def postprocess_mask(self):
        while True: 
            if self.send_timestamps:
                new_mask, timestamp = await self.masks_queue.get()
            else:
                new_mask = await self.masks_queue.get()

            new_com = torch_center_of_mass(new_mask, self.img_size_px*self.interpolation_scale, self.interpolation_scale, self.voxel_size_mm)
            new_com = new_com - self.shift
            if self.send_timestamps:
                self.coms_queue.put_nowait((new_com, timestamp))
            else:
                self.coms_queue.put_nowait(new_com)
            #print(f"New COM: {new_com}")
            await asyncio.sleep(0.002)  # Sleep briefly to avoid busy waiting

    async def send_com(self):

        while True:
            if self.send_timestamps:
                new_com, tstamp_send = await self.coms_queue.get()
            else:
                new_com = await self.coms_queue.get()


            if self.send_data:

                if config['tracker']['connect_to_external_predictor']:

                    # create and send message
                    vec = ps.Vector()

                    vec.x = float(new_com[0])
                    vec.y = float(new_com[1])
                    vec.z = float(0.0)


                    print(datetime.datetime.now(),' current COM x y z ',vec.x,' ',vec.y,' ',vec.z)
                    sub = ps.LetterPub()
                    sub.payload = vec.SerializeToString()
                    sub.message_type = ps.Letter.POSITION_VECTOR
                    env = ps.Envelope()
                    env.payload = sub.SerializeToString()
                    env.message_type = ps.Envelope.LETTER_PUB
                    self.conn_send.send(env.SerializeToString())
                    print(f"Sent center of mass for frame {self.frame_no}: {new_com} at {datetime.datetime.now()}")

                else:

                    if self.frame_no == 1:
                        print(f"Sent center of mass for frame {self.frame_no}: {new_com} at {datetime.datetime.now()}")
                        if self.send_timestamps:
                            value = struct.pack('2fQ', new_com[0], new_com[1], tstamp_send)  # Convert the float and timestamp to bytes
                        else:
                            value = struct.pack('2f', new_com[0], new_com[1])  # Convert the float to bytes
                    else:
                        if self.send_timestamps:
                            value = struct.pack('2fQ', new_com[0], new_com[1], tstamp_send)  # Convert the float and timestamp to bytes
                        else:
                            value = struct.pack('2f', new_com[0], new_com[1])  # Convert the float to bytes
                    self.send_socket.send(value)
                    now = datetime.datetime.now()
                    ts = now.strftime("%Y%m%dT%H%M%S.%f")
                    if config["logging"]["sam_log"]:
                        with open(self.receive_images_exit_log, 'a') as f:
                            f.write(f"{ts} INFO:    gantry angle is currently: templ_at_angle_90\n")
                            f.write(f"{ts} INFO:    mean_x:{new_com[0]}\n")
                            f.write(f"{ts} INFO:    mean_y:{new_com[1]}\n")
                    if config["logging"]["debug"]:
                        with open(self.receive_images_exit_log, 'a') as f:
                            f.write(f"Sent center of mass for frame {self.frame_no}: {new_com} at {ts}\n")
            await asyncio.sleep(0.002)  # Sleep briefly to avoid busy waiting
    # GUI main function 

    async def send_im_and_mask_to_gui(self):
        def compute_largest_contour(binary_mask):
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            largest_contour = max(contours, key=lambda x: len(x))
            return np.array(largest_contour)[:,0,:]
        if config["settings"]["enable_gui"]:
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
                if config["logging"]["debug"]:
                    with open(self.gui_sent_log, 'a') as f:
                        f.write(f"Sent image and mask for frame {self.frame_no} to GUI at {datetime.datetime.now()}\n")

                await asyncio.sleep(0.005)  # Sleep briefly to avoid busy waiting

    # Click functions
    async def receive_clicks_masks(self):
        

        if config["settings"]["enable_gui"] and config["settings"]["interactive"]:
            loop = asyncio.get_running_loop()

            while True:

                try:
                    data = await asyncio.wait_for(
                        loop.sock_recv(self.clicks_socket, 1),
                        timeout=0.001
                    )

                    if data:
                        msg_type = struct.unpack('B', data)[0]
                        
                        if msg_type == 0:  # Click
                            click_data = await asyncio.wait_for(
                                loop.sock_recv(self.clicks_socket, 8),
                                timeout=0.001
                            )
                            if click_data:
                                self.click_received = True
                                x, y = struct.unpack('2f', click_data)
                                self.click_coordinates = (x, y)
                                print(f"Received click at coordinates: ({x}, {y})")
                        
                        elif msg_type == 1:  # Mask
                            header_data = await asyncio.wait_for(
                                loop.sock_recv(self.clicks_socket, 8),
                                timeout=0.001
                            )
                            if header_data:
                                width, height = struct.unpack('!II', header_data)
                                mask_size = width * height
                                
                                mask_data = await asyncio.wait_for(
                                    loop.sock_recv(self.clicks_socket, mask_size),
                                    timeout=0.1
                                )
                                if mask_data:
                                    self.mask_received = True
                                    mask_array = np.frombuffer(mask_data, dtype=np.uint8)
                                    mask_array = mask_array.reshape((height, width))
                                    self.mask_data = mask_array
                                    print(f"Received mask: {width}x{height}")
                                    print(f"Unique values in mask: {np.unique(mask_array)}")
                                    print(f"Mask data type: {mask_array.dtype}")
                                    print(f"Mask array shape: {mask_array.shape}")


                except asyncio.TimeoutError:
                    pass

                await asyncio.sleep(0.005)  # Sleep briefly to avoid busy waiting





    # One time use functions

    def initialize_prompt(self, prompt_library_path=None):
        if testing:
            import SimpleITK as sitk
            self.prompt = sitk.GetArrayFromImage(sitk.ReadImage("/utrecht_exp/data/prompt.mha"))[0]
            print('Initialized prompt')

        else:
            if config['settings']['interactive']:
                print("Interactive mode enabled, waiting for user click or mask to initialize prompt")
            if not config['settings']['interactive']:
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
    image_receiver = ReceiveImages(send_data=True,image_dimensions=(config['tracker']['image_height'], config['tracker']['image_width']),send_timestamps=config['settings']['timestamps'])
    image_receiver.initialize_prompt(prompt_library_path=config['tracker']['prompt_library_path'])
    image_receiver.connect(host=host_rec, port=port_rec)
    image_receiver.connect_send(host=host_send, port=port_send)
    image_receiver.connect_to_gui(host=host_gui, port=port_gui)
    image_receiver.connect_clicks(host=host_clicks, port=port_clicks)
    print("Starting tracking...")
    await asyncio.gather(
        image_receiver.receive_images(),
        image_receiver.preprocess_image(),
        image_receiver.track_frame(),
        image_receiver.postprocess_mask(),
        image_receiver.send_com(),
        image_receiver.send_im_and_mask_to_gui(),
        image_receiver.receive_clicks_masks()
    )

asyncio.run(main())

