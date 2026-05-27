#%%
import numpy 
import matplotlib.pyplot as plt
import datetime

gui_sent_file = "/utrecht_exp/logs/gui_sent.txt"
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

plt.plot(all_times)

t_diff_sent = []
t_diff_broadcast = []

for i in range(len(gui_sent_lines)-1):
    send_com_line = gui_sent_lines[i+1]
    prev_send_com_line = gui_sent_lines[i]
    receive_com_line = gui_broadcast_lines[i+1]
    prev_receive_com_line = gui_broadcast_lines[i]

    if "Sent" in send_com_line and "broadcast" in receive_com_line:
        t_diff_sent.append(datetime.datetime.strptime(send_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(prev_send_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f'))
        t_diff_broadcast.append(datetime.datetime.strptime(receive_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')- datetime.datetime.strptime(prev_receive_com_line.split(' at ')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f'))



#%%
plt.hist(all_times, bins=20)
