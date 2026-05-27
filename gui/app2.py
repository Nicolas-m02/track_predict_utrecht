#%%

from datetime import datetime
from time import time

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import base64

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
clients = []

print("Server started, generating stream...")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await asyncio.sleep(1)
    except:
        clients.remove(ws)


# TESTING STREAMING HERE

import asyncio
import math
import base64
import numpy as np
import cv2

async def test_stream_data():
    t = 0
    while True:
        # Simulate a frame
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, f"t={t}", (50,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        _, buffer = cv2.imencode('.jpg', img)
        frame_b64 = base64.b64encode(buffer).decode()

        # Simulate time-series
        value = math.sin(t * 0.1) + math.cos(t * 0.8)

        # Send to all connected clients
        await broadcast({"frame": frame_b64, "value": value, "t": t})
        print(f"Sent t={t}, value={value}")  # log

        t += 1
        await asyncio.sleep(0.09) 

# Start streaming when FastAPI starts
# @app.on_event("startup")
# async def startup_event():
#     asyncio.create_task(test_stream_data())
import time

import socket
import struct

log_file_path = "/utrecht_exp/gui/stream_log.txt"

with open(log_file_path, "w") as log_file:
    log_file.write("frame,byte_time,total_time\n")

height, width = 112,112
current_frame = 0
latest_frame = None
latest_value = None



async def receive_images():
    host = '0.0.0.0'
    port = 7000
    print(f"Starting image receiver on {host}:{port}...")
    
    # Run socket in thread so we don't block asyncio
    loop = asyncio.get_event_loop()

    await loop.run_in_executor(None, socket_loop, host, port)

import datetime
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

        # Read 4-byte length
        raw_size = recvall_into(conn, 4)
        if not raw_size:
            break
        file_size = struct.unpack('!I', raw_size)[0]

        # Read the image bytes
        #img_bytes = recvall(conn, file_size)
        img_bytes = recvall_into(conn, file_size)
        
        broadcast_end = time.time()
        #print(f"Byte time {current_frame} in {broadcast_end - broadcast_start:.4f} seconds")

        # Convert to OpenCV image for broadcasting
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        # If your images are single-channel, you may need to reshape
        img_np = img_np.reshape((height, width))  
        
        # Encode as JPEG for browser

        _, buffer = cv2.imencode('.jpg', img_np)
        frame_b64 = base64.b64encode(buffer).decode()

        # Broadcast
        end_time = time.time()
        
        #asyncio.run(broadcast({"frame": frame_b64, "t": current_frame, "value": end_time - start_time}))
        
        latest_frame = frame_b64
        
        # Temporary time log, replace with predictions
        #latest_value = end_time - start_time

        with open(log_file_path, "a") as log_file:
            log_file.write(f"Frame {current_frame} broadcast at {datetime.datetime.now()}\n")
        #print(f"Total e2e {end_time - start_time:.4f} seconds")
        current_frame += 1
    
def recvall(sock, n):
    """Helper to receive n bytes or return None if EOF"""
    data = b''
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
    host_pred = '0.0.0.0'
    port_pred = 7005
    print(f"Starting prediction receiver on {host_pred}:{port_pred}...")
    
    # Run socket in thread so we don't block asyncio
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
            data = np.array(struct.unpack('2f', conn_pred.recv(1024)))
            print(data.shape) 
            latest_value = data[1]  # Assume the second value is the one we want
            print(f"Received prediction: {latest_value}")
        except Exception as e:
            #print(f"Error receiving prediction: {e}")
            data = None
        
        


# Broadcasting functions
async def broadcast(data):
    for client in clients:
        try:
            await client.send_json(data)
        except:
            pass

async def periodic_broadcast():
    global latest_frame, latest_value

    target_fps = 30
    interval = 1.0 / target_fps

    t = 0
    while True:
        if latest_frame is not None or latest_value is not None:
            await broadcast({
                "frame": latest_frame,
                "t": t,
                "value": latest_value
            })
            t += 1


        await asyncio.sleep(interval)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(receive_images())
    asyncio.create_task(receive_predictions())
    asyncio.create_task(periodic_broadcast())

