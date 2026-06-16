#%%
import numpy as np


time_spacing = 10 # ms
hz = 11 #Hz
trace_spacing = 1000/11 # ms
px_to_mm = 3.6

# sample_trace = np.loadtxt('/utrecht_exp/data/eval_199CORfixed_angles_trace_3_outer.npy')*px_to_mm
sample_trace = np.loadtxt('/utrecht_exp/data/quasar_traces/original_traces/2_2_COR_0_fixed_angles_trace_3_outer.npy')*px_to_mm

print(f"Sample trace shape: {sample_trace.shape}")
print(f"Maximum value in sample trace: {np.max(sample_trace[:,1])-np.mean(sample_trace[:,1])} mm")
print(f"Minimum value in sample trace: {np.min(sample_trace[:,1])-np.mean(sample_trace[:,1])} mm")
total_time = trace_spacing * sample_trace.shape[0] # ms

print(f"Total time: {total_time} ms")
print(f"Time {total_time/1000:.2f} seconds")

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

qrm_file = '/utrecht_exp/data/quasar_traces/qrm_converted/volunteer_outer-sector_trace.qrm'

with open(qrm_file, 'w') as f:
    f.write('% QUASAR Respiratory motion file\r\n')
    f.write(f'{normalized_trace.shape[0]}\r\n')
    f.write('1.5\r\n') 
    for val in normalized_trace:
        f.write(f'{val}\r\n')


#%% make lung and cardiac traces

def convert_to_heart(trace,hz,heart_thres=3.0, lung_thres=0.4):
    time_ax = np.arange(0, len(trace)/hz, 1/hz)
    freqs = np.fft.fftfreq(len(time_ax), d=(time_ax[1] - time_ax[0]))

    if trace.ndim > 1:
        trace_ap = np.fft.fft(trace[:,0])
        trace_si = np.fft.fft(trace[:,1])
        trace_ap[np.abs(freqs) > heart_thres] = 0
        trace_ap[np.abs(freqs) < lung_thres] = 0
        trace_si[np.abs(freqs) > heart_thres] = 0
        trace_si[np.abs(freqs) < lung_thres] = 0

        trace_ap = np.fft.ifft(trace_ap).real
        trace_si = np.fft.ifft(trace_si).real
        return np.vstack((trace_ap,trace_si)).T
    else:
        freqs = np.fft.fftfreq(len(time_ax), d=(time_ax[1] - time_ax[0]))
        heart_trace = np.fft.fft(trace) 
        heart_trace[np.abs(freqs) > heart_thres] = 0
        heart_trace[np.abs(freqs) < lung_thres] = 0
        # plt.plot(freqs,np.abs(heart_trace), label='Heart FFT Amplitude')
        heart_trace = np.fft.ifft(heart_trace).real
        return heart_trace

def convert_to_lung(trace,hz,lung_thres=0.4):
    time_ax = np.arange(0, len(trace)/hz, 1/hz)
    freqs = np.fft.fftfreq(len(time_ax), d=(time_ax[1] - time_ax[0]))

    if trace.ndim > 1: 
        trace_ap = np.fft.fft(trace[:,0])
        trace_si = np.fft.fft(trace[:,1])
        trace_ap[np.abs(freqs) > lung_thres] = 0
        trace_si[np.abs(freqs) > lung_thres] = 0

        trace_ap = np.fft.ifft(trace_ap).real
        trace_si = np.fft.ifft(trace_si).real
        return np.vstack((trace_ap,trace_si)).T    

    else:
        lung_trace = np.fft.fft(trace)
        lung_trace[np.abs(freqs) > lung_thres] = 0
        # plt.figure()
        # plt.plot(freqs,np.abs(lung_trace), label='Lung FFT Amplitude')
        lung_trace = np.fft.ifft(lung_trace).real
        return lung_trace

lung_trace = convert_to_lung(sample_trace, 10)
heart_trace = convert_to_heart(sample_trace, 10)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(lung_trace[:400], label='Lung Trace')
plt.title('Lung Trace')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()    
plt.subplot(2, 1, 2)
plt.plot(heart_trace[:400], label='Heart Trace', color='orange')
plt.title('Heart Trace')

#%% heart and lung qrm files


interpolated_heart= np.zeros(int(total_time / time_spacing))
interpolated_lung = np.zeros(int(total_time / time_spacing))   
print(f"Interpolated trace shape: {interpolated_heart.shape}")

if interpolation_type == 'linear':

    trace_time_points = np.arange(0, total_time, trace_spacing)
    interp_time_points = np.arange(0, total_time, time_spacing)
    

    print(f"Trace time points shape: {trace_time_points.shape}")
    print(f"Interp time points shape: {interp_time_points.shape}")

    interpolated_heart = np.interp(interp_time_points, trace_time_points, heart_trace[:,1])
    interpolated_lung = np.interp(interp_time_points, trace_time_points, lung_trace[:,1])

    print(f"Interpolated heart trace shape after linear interpolation: {interpolated_heart.shape}")
    print(f"Interpolated lung trace shape after linear interpolation: {interpolated_lung.shape}")

interpolated_heart -= np.mean(interpolated_heart)
interpolated_lung -= np.mean(interpolated_lung)

print(np.max(interpolated_heart), np.min(interpolated_heart))
print(f"Range of motion: {np.max(interpolated_heart) - np.min(interpolated_heart):.2f} mm")

normalized_heart = 2 * ((interpolated_heart - np.min(interpolated_heart)) / (np.max(interpolated_heart) - np.min(interpolated_heart))) - 1
normalized_lung = 2 * ((interpolated_lung - np.min(interpolated_lung)) / (np.max(interpolated_lung) - np.min(interpolated_lung))) - 1   

#%%

qrm_lung = '/utrecht_exp/results/quasar_traces/real_lung_trace.qrm'
qrm_heart = '/utrecht_exp/results/quasar_traces/real_heart_trace.qrm'

with open(qrm_lung, 'w') as f:
    f.write('% QUASAR Respiratory + Cardiac motion file\r\n')
    f.write(f'{normalized_lung.shape[0]}\r\n')
    f.write('1.5\r\n') 
    for val in normalized_lung:
        f.write(f'{val}\r\n')


with open(qrm_heart, 'w') as f:
    f.write('% QUASAR Respiratory + Cardiac motion file\r\n')
    f.write(f'{normalized_heart.shape[0]}\r\n')
    f.write('1.5\r\n') 
    for val in normalized_heart:
        f.write(f'{val}\r\n')


