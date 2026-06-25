#%%

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import time
import os

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import base64
import socket
import struct

import numpy as np
import cv2
import SimpleITK as sitk



with open("/utrecht_exp/config.yaml", 'r') as f:
    import yaml
    config = yaml.safe_load(f)


logging = config['logging']['gui_log']


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

LOG_DIR_GUI = Path(config["logging"]["folder"]) / "gui"

if logging:
    LOG_DIR_GUI.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo("Europe/Amsterdam"))
    ts = now.strftime("%Y%m%dT%H%M%S.%f")

    LOG_FILE_PATH = LOG_DIR_GUI / f"stream_log_{ts}.txt"


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
clients = []

click_queue = asyncio.Queue()
mask_queue = asyncio.Queue()

print("Server started, generating stream...")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_json(), timeout=0.01)

                if data is not None:
                    print("Received:", data["type"])

                    if data["type"] == "click":
                        x = data["x"]
                        y = data["y"]

                        await click_queue.put((x, y))
                        print(f"Click at ({x}, {y})")
                    
                    elif data["type"] == "drawing_mask":
                        mask_b64 = data["mask"]
                        width = data["width"]
                        height = data["height"]
                        
                        # Decode base64 mask
                        mask_bytes = base64.b64decode(mask_b64)
                        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
                        mask_array = mask_array.reshape((height, width))
                        
                        # Create SimpleITK image
                        image = sitk.GetImageFromArray(mask_array)
                        
                        # Save as MHA
                        mha_path = BASE_DIR / "received_mask.mha"
                        sitk.WriteImage(image, str(mha_path))
                        
                        print(f"Mask saved to {mha_path}")
                        
                        # Also queue for sending over socket
                        await mask_queue.put((mask_array, width, height))
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.001)

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        if ws in clients:
            clients.remove(ws)

if logging:
    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write("frame,byte_time,total_time\n")

height, width = 128, 128
current_frame = 0
latest_frame = None
latest_value = None


async def receive_images():
    host = "0.0.0.0"
    port = 7000

    print(f"Starting image receiver on {host}:{port}...")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, socket_loop, host, port)


def socket_loop(host, port):
    global current_frame
    global latest_frame, latest_value

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)

    print("Waiting for image sender...")
    conn, addr = s.accept()
    print(f"Connected by {addr}")

    while True:
        start_time = time.time()
        broadcast_start = time.time()

        raw_size = recvall_into(conn, 4)
        if not raw_size:
            break
        file_size = struct.unpack("!I", raw_size)[0]

        img_bytes = recvall_into(conn, file_size)
        broadcast_end = time.time()

        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img_np = img_np.reshape((height, width))

        _, buffer = cv2.imencode(".jpg", img_np)
        frame_b64 = base64.b64encode(buffer).decode()

        end_time = time.time()

        latest_frame = frame_b64

        if logging:
            with open(LOG_FILE_PATH, "a") as log_file:
                log_file.write(f"Frame {current_frame} broadcast at {datetime.now()}\n")

        current_frame += 1


def recvall(sock, n):
    """Helper to receive n bytes or return None if EOF."""
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def recvall_into(conn, size):
    buf = bytearray(size)
    view = memoryview(buf)

    total = 0
    while total < size:
        n = conn.recv_into(view[total:])
        if n == 0:
            raise ConnectionError("Socket closed")
        total += n

    return buf


async def receive_predictions():
    host_pred = "0.0.0.0"
    port_pred = 7005
    print(f"Starting prediction receiver on {host_pred}:{port_pred}...")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, prediction_socket_loop, host_pred, port_pred)


def prediction_socket_loop(host, port):
    global latest_frame, latest_value

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print("Waiting for prediction sender...")
    conn_pred, addr_pred = s.accept()
    print(f"Prediction connected by {addr_pred}")

    while True:
        try:
            data = np.array(struct.unpack("2f", conn_pred.recv(1024)))
            latest_value = data[1]
        except Exception:
            data = None


async def send_clicks(host_click="0.0.0.0", port_click=8000):
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    c.bind((host_click, port_click))
    c.listen(1)

    print("Connection to click/mask server...")
    conn_click, addr_click = c.accept()
    print(f"Connected to click/mask server by {addr_click}")

    while True:
        try:
            # Check for mask with 0 timeout (non-blocking)
            mask_data = mask_queue.get_nowait()
            mask_array, width, height = mask_data
            
            # Send mask: type(1 byte) + width(4) + height(4) + data
            msg_type = struct.pack('B', 1)  # 1 = mask
            header = struct.pack('!II', width, height)
            mask_bytes = mask_array.tobytes()
            
            conn_click.sendall(msg_type + header + mask_bytes)
            print(f"Sent mask: {width}x{height}")
        except asyncio.QueueEmpty:
            pass
        
        try:
            # Check for click with 0 timeout (non-blocking)
            x, y = click_queue.get_nowait()
            
            # Send click: type(1 byte) + x(4) + y(4)
            msg_type = struct.pack('B', 0)  # 0 = click
            coords = struct.pack('2f', x, y)
            
            conn_click.sendall(msg_type + coords)
            print(f"Sent click: ({x}, {y})")
        except asyncio.QueueEmpty:
            pass

        await asyncio.sleep(0.005)


async def broadcast(data):
    for client in clients:
        try:
            await client.send_json(data)
        except Exception:
            pass


async def periodic_broadcast():
    global latest_frame, latest_value

    target_fps = 30
    interval = 1.0 / target_fps

    t = 0
    while True:
        if latest_frame is not None or latest_value is not None:
            await broadcast(
                {
                    "frame": latest_frame,
                    "t": t,
                    "value": latest_value,
                }
            )
            t += 1

        await asyncio.sleep(interval)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(receive_images())
    asyncio.create_task(receive_predictions())
    asyncio.create_task(periodic_broadcast())
    asyncio.create_task(send_clicks())
