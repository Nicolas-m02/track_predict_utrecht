#%%

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

async def broadcast(data):
    for client in clients:
        try:
            await client.send_json(data)
        except:
            pass

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
        value = math.sin(t * 0.1) + math.cos(t * 0.05)

        # Send to all connected clients
        await broadcast({"frame": frame_b64, "value": value, "t": t})
        print(f"Sent t={t}, value={value}")  # log

        t += 1
        await asyncio.sleep(0.09) 

# Start streaming when FastAPI starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(test_stream_data())

import socket
import struct

async def receive_images():
    host = '0.0.0.0'
    port = 1220
    print(f"Starting image receiver on {host}:{port}...")
    
    # Run socket in thread so we don't block asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, socket_loop, host, port)

def socket_loop(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print("Waiting for sender...")
    conn, addr = s.accept()
    print(f"Connected by {addr}")

    while True:
        # Read 4-byte length
        raw_size = recvall(conn, 4)
        if not raw_size:
            break
        file_size = struct.unpack('!I', raw_size)[0]

        # Read the image bytes
        img_bytes = recvall(conn, file_size)

        # Convert to OpenCV image for broadcasting
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        # If your images are single-channel, you may need to reshape
        # img_np = img_np.reshape((height, width))  

        # Encode as JPEG for browser
        _, buffer = cv2.imencode('.jpg', img_np)
        frame_b64 = base64.b64encode(buffer).decode()

        # Broadcast
        asyncio.run(broadcast({"frame": frame_b64, "t": 0, "value": 0}))

def recvall(sock, n):
    """Helper to receive n bytes or return None if EOF"""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# @app.on_event("startup")
# async def startup_event():
#     asyncio.create_task(receive_images())
