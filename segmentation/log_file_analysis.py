#%% finding latencies between sending and receiving images in the logs

import os
import numpy as np
import datetime

log_gui = '/utrecht_exp/logs/gui_test_log_20260330_141221.txt'
log_send = '/utrecht_exp/logs/real_time_image_send_log_20260330_141229.txt'

with open(log_gui, 'r') as f:
    gui_lines = f.readlines()

with open(log_send, 'r') as f:
    send_lines = f.readlines()


print(len(gui_lines))
print(len(send_lines))

all_latencies = []
for i in range(len(gui_lines)):
    gui_line = gui_lines[i]
    send_line = send_lines[i]

    

    if "Received image" in gui_line and "Sent image" in send_line:
        t_diff = datetime.datetime.strptime(gui_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(send_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')
        print(f"Frame {i+1}: Time difference between send and receive: {t_diff.total_seconds()*1000:.2f} ms")

    all_latencies.append(t_diff.total_seconds()*1000)


print(f"Average latency: {np.mean(all_latencies):.2f} ms")


#%% End to end test of sending and predicting

prediction_file = '/utrecht_exp/logs/online_log.txt'

send_file = '/utrecht_exp/logs/real_time_image_send_log_20260330_163407.txt'

with open(prediction_file, 'r') as f:
    pred_lines = f.readlines()

with open(send_file, 'r') as f:
    send_lines = f.readlines()


print(len(pred_lines))
print(len(send_lines))

# Crop out metadata

pred_lines = pred_lines[2:]
send_lines = send_lines[102:]

print(len(pred_lines))
print(len(send_lines))

# prediction_starts at line 100

all_latencies = []

print(pred_lines[0])
print(send_lines[0])
import datetime
for i in range(len(pred_lines)):
    pred_line = pred_lines[i]
    send_line = send_lines[i]

    if "Prediction" in pred_line and "Sent image" in send_line:
        t_diff = datetime.datetime.strptime(pred_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(send_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')
        print(f"Frame {i+1}: Time difference between send and prediction: {t_diff.total_seconds()*1000:.2f} ms")

        all_latencies.append(t_diff.total_seconds()*1000)

print(f"Average latency: {np.mean(all_latencies):.2f} ms")


#%%
import matplotlib.pyplot as plt
import numpy as np
import datetime

im_send_file = '/utrecht_exp/logs/real_time_image_send_log.txt'
im_receive_file = '/utrecht_exp/logs/receive_images_enter.txt'

com_send_file = '/utrecht_exp/logs/receive_images_exit.txt'
com_receive_file = '/utrecht_exp/logs/online_received.txt'

predicted_com_file = '/utrecht_exp/logs/online_log.txt'


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

im_send_lines = im_send_lines[2:]
im_receive_lines = im_receive_lines[2:]
com_send_lines = com_send_lines[2:]
com_receive_lines = com_receive_lines[2:]
predicted_com_lines = predicted_com_lines[2:]

print(f"Images sent: {len(im_send_lines)}")
print(f"Images received: {len(im_receive_lines)}")
print(f"Center of mass sent: {len(com_send_lines)}")
print(f"Center of mass received: {len(com_receive_lines)}")
print(f"Predicted center of mass: {len(predicted_com_lines)}")


send_receive_ims = []
receive_track_ims = []
track_receive_coms = []
receive_predict_coms = []

for i in range(len(im_send_lines)):
    send_line = im_send_lines[i]
    receive_line = im_receive_lines[i]

    if "Sent image" in send_line and "Received image" in receive_line:
        t_diff = datetime.datetime.strptime(receive_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(send_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')
        #print(f"Frame {i+1}: Time difference between send and receive of center of mass: {t_diff.total_seconds()*1000:.2f} ms")
        send_receive_ims.append(t_diff.total_seconds()*1000)

print(f"Average latency between sending images and receiving them: {np.mean(send_receive_ims):.2f} ms")
plt.figure()
plt.title('Latency from sending image to receiving it')
plt.plot(send_receive_ims, label=f'Send to Receive Image Latency\n Mean: {np.mean(send_receive_ims):.2f} ms \n Max: {np.max(send_receive_ims):.2f} ms \n Min: {np.min(send_receive_ims):.2f} ms\n Mean without outliers: {np.mean([x for x in send_receive_ims if x < 200]):.2f} ms')
plt.xlabel('Frame Index')
plt.ylabel('Latency (ms)')
plt.legend()

for i in range(len(com_send_lines)):
    receive_line = im_receive_lines[i]
    send_com_line = com_send_lines[i]

    if "Received image" in receive_line and "Sent center of mass" in send_com_line:
        t_diff =  datetime.datetime.strptime(send_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')- datetime.datetime.strptime(receive_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')
        #print(f"Frame {i+1}: Time difference between send and receive of center of mass: {t_diff.total_seconds()*1000:.2f} ms")
        receive_track_ims.append(t_diff.total_seconds()*1000)


plt.figure()
plt.title('Latency from receiving image to sending center of mass')
plt.plot(receive_track_ims, label=f'Receive Image to Send CoM Latency\n Mean: {np.mean(receive_track_ims):.2f} ms \n Max: {np.max(receive_track_ims):.2f} ms \n Min: {np.min(receive_track_ims):.2f} ms\nMean without outliers: {np.mean([x for x in receive_track_ims if x < 500]):.2f} ms')
plt.xlabel('Frame Index')
plt.ylabel('Latency (ms)')
plt.legend()

for i in range(len(com_receive_lines)):
    send_com_line = com_send_lines[i]
    receive_com_line = com_receive_lines[i]

    if "Sent center of mass" in send_com_line and "Received data" in receive_com_line:
        t_diff = datetime.datetime.strptime(receive_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(send_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')
        #print(f"Frame {i+1}: Time difference between send and receive of center of mass: {t_diff.total_seconds()*1000:.2f} ms")
        track_receive_coms.append(t_diff.total_seconds()*1000)

plt.figure()
plt.title('Latency from sending center of mass to receiving it')
plt.plot(track_receive_coms, label=f'Send CoM to Receive CoM Latency\n Mean: {np.mean(track_receive_coms):.2f} ms \n Max: {np.max(track_receive_coms):.2f} ms \n Min: {np.min(track_receive_coms):.2f} ms')
plt.xlabel('Frame Index')
plt.ylabel('Latency (ms)')
plt.title('Latency from sending center of mass to receiving it')
plt.legend()

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

#%% end to end

e2e_latencies = []
for i in range(len(predicted_com_lines)):
    sent_image_line = im_send_lines[99+i]
    predicted_com_line = predicted_com_lines[i]

    if "Sent image" in sent_image_line and "Prediction" in predicted_com_line:
        t_diff = datetime.datetime.strptime(predicted_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(sent_image_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')
        #print(f"Frame {i+1}: Time difference between receiving center of mass and prediction: {t_diff.total_seconds()*1000:.2f} ms")
        e2e_latencies.append(t_diff.total_seconds()*1000)

corrected_frames = np.arange(len(e2e_latencies)) + 99

plt.figure()
plt.title('End-to-End Latency from sending image to prediction')
plt.plot(corrected_frames, e2e_latencies, label=f'End-to-End Latency\n Mean: {np.mean(e2e_latencies):.2f} ms \n Max: {np.max(e2e_latencies):.2f} ms \n Min: {np.min(e2e_latencies):.2f} ms')
plt.plot(corrected_frames, [np.mean([x for x in e2e_latencies if x < 250])] * len(corrected_frames), label=f'Mean without outliers Latency: {np.mean([x for x in e2e_latencies if x < 250]):.2f} ms', linestyle='--')
plt.xlabel('Frame Index')
plt.ylabel('Latency (ms)')
plt.legend()



#%%

def parse_log_file(file_path, keyword):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = lines[2:]  # Skip metadata
    timestamps = []
    for line in lines:
        if keyword in line:
            timestamp_str = line.split(' at ')[-1].strip()
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(timestamp)
    return timestamps

orig_im_send_timestamps = parse_log_file(im_send_file, "Sent image")
orig_im_receive_timestamps = parse_log_file(im_receive_file, "Received image")
orig_com_send_timestamps = parse_log_file(com_send_file, "Sent center of mass")
orig_com_receive_timestamps = parse_log_file(com_receive_file, "Received data")
orig_predicted_com_timestamps = parse_log_file(predicted_com_file, "Prediction")

#%%
import matplotlib.pyplot as plt

im_send_timestamps = np.array([t - orig_im_send_timestamps[0] for t in orig_im_send_timestamps]).astype('timedelta64[ms]').astype(float)
im_receive_timestamps = np.array([t - orig_im_receive_timestamps[0] for t in orig_im_receive_timestamps]).astype('timedelta64[ms]').astype(float)
com_send_timestamps = np.array([t - orig_com_send_timestamps[0] for t in orig_com_send_timestamps]).astype('timedelta64[ms]').astype(float)
com_receive_timestamps = np.array([t - orig_com_receive_timestamps[0] for t in orig_com_receive_timestamps]).astype('timedelta64[ms]').astype(float)
predicted_com_timestamps = np.array([t - orig_predicted_com_timestamps[0] for t in orig_predicted_com_timestamps]).astype('timedelta64[ms]').astype(float)

plt.figure(figsize=(10, 6))
plt.plot(im_send_timestamps, label='Sent Image', marker='o')
plt.plot(im_receive_timestamps, label='Received Image', marker='o')
plt.plot(com_send_timestamps, label='Sent Center of Mass', marker='o')
plt.plot(com_receive_timestamps, label='Received Data', marker='o')
plt.plot(predicted_com_timestamps, label='Predicted CoM', marker='o')

plt.xlabel('Frame Index')
plt.ylabel('Time (ms)')
plt.title('Timestamp Analysis')
plt.legend()
plt.ylim(0, 10000)  # Adjust y-axis limit for better visualization
plt.xlim(0, 100)  
plt.show()


#%%

import matplotlib.pyplot as plt
print(f"Number of predicted center of mass timestamps: {len(orig_predicted_com_timestamps)}")
print(f"Number of received data timestamps: {len(orig_com_receive_timestamps)}")
print(f"Number of sent center of mass timestamps: {len(orig_com_send_timestamps)}")
print(f"Number of sent image timestamps: {len(orig_im_send_timestamps)}")


plt.figure(figsize=(10, 6))
plt.plot(orig_predicted_com_timestamps[180:220], label='Predicted CoM')
plt.plot(orig_com_receive_timestamps[280:320], label='Received Data')
plt.plot(orig_com_send_timestamps[280:320], label='Sent Center of Mass')



#%% find differences in time for each step

im_send_diff = np.array([t - orig_im_send_timestamps[i-1] for i,t in enumerate(orig_im_send_timestamps)]).astype('timedelta64[ms]').astype(float)
im_receive_diff = np.array([t - orig_im_receive_timestamps[i-1] for i,t in enumerate(orig_im_receive_timestamps)]).astype('timedelta64[ms]').astype(float)
com_send_diff = np.array([t - orig_com_send_timestamps[i-1] for i,t in enumerate(orig_com_send_timestamps)]).astype('timedelta64[ms]').astype(float)
com_receive_diff = np.array([t - orig_com_receive_timestamps[i-1] for i,t in enumerate(orig_com_receive_timestamps)]).astype('timedelta64[ms]').astype(float)
predicted_diff = np.array([t - orig_predicted_com_timestamps[i-1] for i,t in enumerate(orig_predicted_com_timestamps)]).astype('timedelta64[ms]').astype(float)


print(len(im_send_diff))
print(len(im_receive_diff))
print(len(com_send_diff))
print(len(com_receive_diff))
print(len(predicted_diff))


plt.figure(figsize=(10, 6))
plt.title('Time Differences Between Image sends')
plt.plot(im_send_diff[1:], label='Sent Image Diff')

plt.figure(figsize=(10, 6))
plt.title('Time Differences Between Image receives')
plt.plot(im_receive_diff[1:], label='Received Image Diff')


plt.figure(figsize=(10, 6))
plt.title('Time Differences Between Center of Mass sends')
plt.plot(com_send_diff[1:], label='Sent Center of Mass Diff')

plt.figure(figsize=(10, 6))
plt.title('Time Differences Between Data receives')
plt.plot(com_receive_diff[1:], label='Received Data Diff')

plt.figure(figsize=(10, 6))
plt.title('Time Differences Between Predictions')
plt.plot(predicted_diff[1:], label='Predicted CoM Diff')



#%%

print(f"Average time between sending images: {np.mean(im_send_diff[1:]):.2f} ms")
print(f"Average time between receiving images: {np.mean(im_receive_diff[1:]):.2f} ms")
print(f"Average time between sending center of mass: {np.mean(com_send_diff[1:]):.2f} ms")
print(f"Average time between receiving data: {np.mean(com_receive_diff[1:]):.2f} ms")
print(f"Average time between predictions: {np.mean(predicted_diff[1:]):.2f} ms")




