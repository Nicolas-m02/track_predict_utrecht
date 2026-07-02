#%%
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from backports.zoneinfo import ZoneInfo


def parse_timestamp(ts: str) -> np.datetime64:
    """Parse log timestamp of the form 'YYYYMMDDTHHMMSS.xxxxxx'."""
    return np.datetime64(f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:]}")

def parse_prediction(line: str) -> np.array:
    """Parse prediction line of the form 'Pred X: [value]'."""
    return np.array([float(line.split('[')[1].split(']')[0].split()[0]), float(line.split('[')[1].split(']')[0].split()[1])])

log_dir = "/utrecht_exp/results/to_analyze/volunteer_lmu_circle_20260701T164753.834233"


# READ IN LSTM PREDICTIONS
# DATA FORMAT: 
# PRED 1:
# PRED 2:
# PRED 3:
# PRED 4: 

lstm_pred_log_file = [f for f in os.listdir(os.path.join(log_dir,'lstm')) if f.startswith("pred_log_")][0]

with open(os.path.join(log_dir,'lstm',lstm_pred_log_file), "r") as f:
    lstm_lines = f.readlines()

lstm_lines = lstm_lines[3:]
print(lstm_lines[:5])

now = datetime.datetime.now(ZoneInfo("Europe/Amsterdam"))
ts = now.strftime("%Y%m%dT%H%M%S.%f")

all_times = []
pred_1_list = []
pred_2_list = []
pred_3_list = []
pred_4_list = []

for i, line in enumerate(lstm_lines):
    t_stamp = parse_timestamp(line.split()[0])
    #print(t_stamp)

    if 'Pred 1:' in line:
        pred_1 = parse_prediction(line)
        #print(f"Pred 1: {pred_1}")
        pred_1_list.append(pred_1)
    if 'Pred 2:' in line:
        pred_2 = parse_prediction(line)
        #print(f"Pred 2: {pred_2}")
        pred_2_list.append(pred_2)
    if 'Pred 3:' in line:
        pred_3 = parse_prediction(line)
        #print(f"Pred 3: {pred_3}")
        pred_3_list.append(pred_3)
    if 'Pred 4:' in line:
        pred_4 = parse_prediction(line)
        #print(f"Pred 4: {pred_4}")
        pred_4_list.append(pred_4)
        all_times.append(t_stamp)

#%%

pred_1_array = np.array(pred_1_list)
pred_2_array = np.array(pred_2_list)
pred_3_array = np.array(pred_3_list)
pred_4_array = np.array(pred_4_list)

input_latency = 263

interpolation_point = (input_latency / 90)

print(f"Interpolation point: {interpolation_point}")


interpolated_predictions = np.zeros((len(all_times), 2))

if interpolation_point > 1 and interpolation_point < 2:
    rel_interpolation_point = interpolation_point - 1
    interpolated_predictions[:, 0] = (1 - interpolation_point) * pred_1_array[:, 0] + interpolation_point * pred_2_array[:, 0]
    interpolated_predictions[:, 1] = (1 - interpolation_point) * pred_1_array[:, 1] + interpolation_point * pred_2_array[:, 1]

elif interpolation_point > 2 and interpolation_point < 3:
    rel_interpolation_point = interpolation_point - 2
    interpolated_predictions[:, 0] = (1 - interpolation_point) * pred_2_array[:, 0] + interpolation_point * pred_3_array[:, 0]
    interpolated_predictions[:, 1] = (1 - interpolation_point) * pred_2_array[:, 1] + interpolation_point * pred_3_array[:, 1]

elif interpolation_point > 3 and interpolation_point < 4:
    rel_interpolation_point = interpolation_point - 3
    interpolated_predictions[:, 0] = (1 - interpolation_point) * pred_3_array[:, 0] + interpolation_point * pred_4_array[:, 0]
    interpolated_predictions[:, 1] = (1 - interpolation_point) * pred_3_array[:, 1] + interpolation_point * pred_4_array[:, 1]
else: 
    print("Interpolation point is not in the expected range (1, 4). No interpolation performed.")
    interpolated_predictions = pred_4_array

print(f"Interpolated predictions: {interpolated_predictions[:5]}")
#%%

plt.figure(figsize=(12, 6))
plt.plot(all_times, interpolated_predictions[:,1], label='Pred 1 Y')
plt.xlabel('Time')
plt.ylabel('Prediction')
plt.title(f'LSTM Predictions to {input_latency} ms')
plt.legend()
plt.show()


#%% find latency between images_enter and images exit


sam_received_log_file = [f for f in os.listdir(os.path.join(log_dir,'tracking_module')) if f.startswith("receive")][0]
sam_sent_log_file = [f for f in os.listdir(os.path.join(log_dir,'tracking_module')) if f.startswith("mri")][0]

print(f"Reading SAM received log file: {sam_received_log_file}")
print(f"Reading SAM sent log file: {sam_sent_log_file}")    


with open(os.path.join(log_dir,'tracking_module',sam_received_log_file), "r") as f:
    rec_lines = f.readlines()

with open(os.path.join(log_dir,'tracking_module',sam_sent_log_file), "r") as f:
    sent_lines = f.readlines()

rec_lines = rec_lines[2:]
sent_lines = sent_lines[2:]


for i, line in enumerate(rec_lines):
    t_stamp = parse_timestamp(line.split()[0])
    if 'images_enter' in line:
        images_enter_time = t_stamp
        print(f"images_enter time: {images_enter_time}")
    if 'images_exit' in line:
        images_exit_time = t_stamp
        print(f"images_exit time: {images_exit_time}")