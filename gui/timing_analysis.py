#%%
import numpy 
import matplotlib.pyplot as plt
import datetime

gui_sent_file = "/utrecht_exp/gui/gui_sent.txt"
gui_broadcast_file = "/utrecht_exp/gui/stream_log.txt"

with open(gui_sent_file, 'r') as f:
    gui_sent_lines = f.readlines()

with open(gui_broadcast_file, 'r') as f:
    gui_broadcast_lines = f.readlines()

gui_sent_lines = gui_sent_lines[2:]  # skip header
gui_broadcast_lines = gui_broadcast_lines[1:]  # skip header

print(len(gui_sent_lines), len(gui_broadcast_lines))


all_times = []

for i in range(len(gui_sent_lines)):
    send_com_line = gui_sent_lines[i]
    receive_com_line = gui_broadcast_lines[i]

    if "Sent" in send_com_line and "broadcast" in receive_com_line:
        t_diff = datetime.datetime.strptime(receive_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(send_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')
        #print(f"Frame {i+1}: Time difference between send and receive of center of mass: {t_diff.total_seconds()*1000:.2f} ms")
        all_times.append(t_diff.total_seconds()*1000)
plt.figure(figsize=(10,5))
plt.title("Time difference between Sent and Broadcast for each frame")
plt.plot(all_times)
plt.xlabel("Frame")
plt.ylabel("Time difference (ms)")
plt.figure(figsize=(10,5))

t_diff_sent = []
t_diff_broadcast = []

for i in range(len(gui_sent_lines)-1):
    send_com_line = gui_sent_lines[i+1]
    prev_send_com_line = gui_sent_lines[i]
    receive_com_line = gui_broadcast_lines[i+1]
    prev_receive_com_line = gui_broadcast_lines[i]

    if "Sent" in send_com_line and "broadcast" in receive_com_line:
        t_diff_sent.append((datetime.datetime.strptime(send_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(prev_send_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')).total_seconds()*1000)
        t_diff_broadcast.append((datetime.datetime.strptime(receive_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')- datetime.datetime.strptime(prev_receive_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')).total_seconds()*1000)

plt.plot(t_diff_sent[:50], label='Sent')
#plt.plot(t_diff_broadcast, label='Broadcast')

#%%

import numpy as np
import matplotlib.pyplot as plt
import datetime

im_send_file = '/utrecht_exp/gui/real_time_image_send_log.txt'
im_receive_file = '/utrecht_exp/gui/receive_images_enter.txt'

com_send_file = '/utrecht_exp/gui/receive_images_exit.txt'
com_receive_file = '/utrecht_exp/logs/online_received.txt'

predicted_com_file = '/utrecht_exp/logs/online_log.txt'

# Logs for stream
gui_sent_file = "/utrecht_exp/gui/gui_sent.txt"
gui_broadcast_file = "/utrecht_exp/gui/stream_log.txt"



with open(im_send_file, 'r') as f:
    im_send_lines = f.readlines()

with open(im_receive_file, 'r') as f:
    im_receive_lines = f.readlines()

with open(com_send_file, 'r') as f:
    com_send_lines = f.readlines()

with open(com_receive_file, 'r') as f:
    com_receive_lines = f.readlines()

with open(predicted_com_file, 'r') as f:
    predicted_com_lines = f.readlines()

with open(gui_sent_file, 'r') as f:
    gui_sent_lines = f.readlines()

with open(gui_broadcast_file, 'r') as f:    
    gui_broadcast_lines = f.readlines()


print(len(im_send_lines), len(im_receive_lines), len(com_send_lines), len(com_receive_lines), len(predicted_com_lines))
print(len(gui_sent_lines), len(gui_broadcast_lines))

print(f"Sent images log: {len(im_send_lines)} lines")
print(f"Received images log: {len(im_receive_lines)} lines")
print(f"Sent center of mass log: {len(com_send_lines)} lines")
print(f"Received center of mass log: {len(com_receive_lines)} lines")
print(f"Predicted center of mass log: {len(predicted_com_lines)} lines")

print(f"GUI sent log: {len(gui_sent_lines)} lines")
print(f"GUI broadcast log: {len(gui_broadcast_lines)} lines")

t_tracking = []
for i in range(len(com_send_lines)):
    im_receive_line = im_receive_lines[i]
    com_send_line = com_send_lines[i]

    if "Received image" in im_receive_line and "Sent center of mass" in com_send_line:
        t_diff = datetime.datetime.strptime(com_send_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(im_receive_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')
        t_tracking.append(t_diff.total_seconds()*1000)

plt.figure(figsize=(10,5))
plt.title("Time difference between receiving image and sending center of mass")
plt.plot(t_tracking)

t_track_gui = []
for i in range(len(com_send_lines)):
    im_receive_line = im_receive_lines[i]
    gui_sent_line = gui_sent_lines[i]

    if "Received image" in im_receive_line and "Sent image and mask for frame" in gui_sent_line:
        t_diff = datetime.datetime.strptime(gui_sent_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(im_receive_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')
        t_track_gui.append(t_diff.total_seconds()*1000)

plt.figure(figsize=(10,5))
plt.title("Time difference between sending center of mass and sending image to GUI")
plt.plot(t_track_gui)


receive_predict_coms = []
for i in range(len(predicted_com_lines)):
    receive_com_line = com_receive_lines[99+i]
    predicted_com_line = predicted_com_lines[i]

    if "Received data" in receive_com_line and "Prediction" in predicted_com_line:
        t_diff = datetime.datetime.strptime(predicted_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(receive_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')
        #print(f"Frame {i+1}: Time difference between receiving center of mass and prediction: {t_diff.total_seconds()*1000:.2f} ms")
        receive_predict_coms.append(t_diff.total_seconds()*1000)

corrected_frames = np.arange(len(receive_predict_coms)) + 99

plt.figure()
plt.plot(corrected_frames, receive_predict_coms, label=f'Receive CoM to Prediction Latency\n Mean: {np.mean(receive_predict_coms):.2f} ms \n Max: {np.max(receive_predict_coms):.2f} ms \n Min: {np.min(receive_predict_coms):.2f} ms')
plt.xlabel('Frame Index')
plt.ylabel('Latency (ms)')
plt.title('Latency from receiving center of mass to prediction')
plt.legend(loc='upper right')


