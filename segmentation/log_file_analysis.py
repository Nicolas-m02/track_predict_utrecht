#%%
import matplotlib.pyplot as plt
import numpy as np
import datetime

im_send_file = '/utrecht_exp/logs/real_time_image_send_log.txt'
im_receive_file = '/utrecht_exp/logs/receive_images_enter.txt'

com_send_file = '/utrecht_exp/logs/receive_images_exit.txt'
com_receive_file = '/utrecht_exp/logs/online_received.txt'

predicted_com_file = '/utrecht_exp/logs/online_log.txt'

with open(im_receive_file, 'r') as f:
    im_receive_lines = f.readlines()

with open(com_send_file, 'r') as f:
    com_send_lines = f.readlines()

with open(com_receive_file, 'r') as f:
    com_receive_lines = f.readlines()

with open(predicted_com_file, 'r') as f:
    predicted_com_lines = f.readlines()

im_receive_lines = im_receive_lines[2:]
com_send_lines = com_send_lines[2:]
com_receive_lines = com_receive_lines[2:]
predicted_com_lines = predicted_com_lines[2:]

print(f"Images received: {len(im_receive_lines)}")
print(f"Center of mass sent: {len(com_send_lines)}")
print(f"Center of mass received: {len(com_receive_lines)}")
print(f"Predicted center of mass: {len(predicted_com_lines)}")


send_receive_ims = []
receive_track_ims = []
track_receive_coms = []
receive_predict_coms = []

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
plt.savefig('/utrecht_exp/results/receive_image_to_send_com_latency.png')

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
plt.savefig('/utrecht_exp/results/send_com_to_receive_com_latency.png')

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
plt.savefig('/utrecht_exp/results/receive_com_to_prediction_latency.png')
#%% end to end

e2e_latencies = []
for i in range(len(predicted_com_lines)):
    sent_image_line = im_receive_lines[99+i]
    predicted_com_line = predicted_com_lines[i]

    if "Received image" in sent_image_line and "Prediction" in predicted_com_line:
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
plt.savefig('/utrecht_exp/results/e2e.png')


