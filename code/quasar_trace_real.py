#%%
import numpy as np


time_spacing = 10 # ms

trace_spacing = 1000/11 # ms
px_to_mm = 3.6

sample_trace = np.loadtxt('/utrecht_exp/data/eval_199CORfixed_angles_trace_3_outer.npy')*px_to_mm

print(f"Sample trace shape: {sample_trace.shape}")
print(f"Maximum value in sample trace: {np.max(sample_trace[:,1])-np.mean(sample_trace[:,1])} mm")
print(f"Minimum value in sample trace: {np.min(sample_trace[:,1])-np.mean(sample_trace[:,1])} mm")
total_time = trace_spacing * sample_trace.shape[0] # ms

print(f"Total time: {total_time} ms")


#%%
interpolation_type = 'linear' # 'linear' or 'cubic'

interpolated_trace = np.zeros(int(total_time / time_spacing))
print(f"Interpolated trace shape: {interpolated_trace.shape}")

if interpolation_type == 'linear':

    trace_time_points = np.arange(0, total_time, trace_spacing)
    interp_time_points = np.arange(0, total_time, time_spacing)
    

    print(f"Trace time points shape: {trace_time_points.shape}")
    print(f"Interp time points shape: {interp_time_points.shape}")

    interpolated_trace = np.interp(interp_time_points, trace_time_points, sample_trace[:,1])
    print(f"Interpolated trace shape after linear interpolation: {interpolated_trace.shape}")

interpolated_trace -= np.mean(interpolated_trace)

print(np.max(interpolated_trace), np.min(interpolated_trace))
print(f"Range of motion: {np.max(interpolated_trace) - np.min(interpolated_trace):.2f} mm")

normalized_trace = 2 * ((interpolated_trace - np.min(interpolated_trace)) / (np.max(interpolated_trace) - np.min(interpolated_trace))) - 1

print(np.max(normalized_trace), np.min(normalized_trace))
#%% save interpolated trace

qrm_file = '/utrecht_exp/results/quasar_traces/real_trace.qrm'

with open(qrm_file, 'w') as f:
    f.write('% QUASAR Respiratory + Cardiac motion file\r\n')
    f.write(f'{normalized_trace.shape[0]}\r\n')
    f.write('1.5\r\n') 
    for val in normalized_trace:
        f.write(f'{val}\r\n')

