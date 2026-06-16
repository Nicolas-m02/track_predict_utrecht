#%%
 
import socket
import struct
import numpy as np
import cv2
from flask import Flask, Response
import threading
import datetime
import os

HOST = '0.0.0.0'
PORT = 1220


log_out_dir = "/utrecht_exp/logs/"
os.makedirs(log_out_dir, exist_ok=True)
log_file = os.path.join(log_out_dir, f"gui_test_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

HOST = '0.0.0.0'
PORT = 1220

latest_frame = None

with open(log_file, 'w') as f:
    f.write(f"Log file created at {datetime.datetime.now()}\n\n")

def receive_images():
    global latest_frame

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)

    print(f"Listening on {HOST}:{PORT}")
    conn, addr = s.accept()
    print(f"Connected by {addr}")



    while True:
        # receive image size
        size_data = conn.recv(4)
        if not size_data:
            break

        img_size = struct.unpack('!I', size_data)[0]

        # receive full image
        data = b''
        while len(data) < img_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        with open(log_file, 'a') as f:
            f.write(f"Received image of size {img_size} bytes at {datetime.datetime.now()}\n")

        frame = np.frombuffer(data, dtype=np.uint16).reshape((112, 112))

        #normalize to 8-bit for visualization
        frame_8bit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_8bit = frame_8bit.astype(np.uint8)

        # optional: make it easier to see
        frame_8bit = cv2.resize(frame_8bit, (400, 400))

        # optional: apply colormap (VERY useful for uint16 data)
        # frame_8bit = cv2.applyColorMap(frame_8bit, cv2.COLORMAP_JET)
        frame_8bit = cv2.applyColorMap(frame_8bit, cv2.COLORMAP_JET)


        latest_frame = frame_8bit


# --- Flask GUI ---
app = Flask(__name__)

def generate():
    global latest_frame

    while True:
        if latest_frame is None:
            continue

        _, buffer = cv2.imencode('.jpg', latest_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    threading.Thread(target=receive_images, daemon=True).start()
    app.run(host='0.0.0.0', port=32)