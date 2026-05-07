#%%
import numpy as np
import matplotlib.pyplot as plt
import os

# %% init variables

# --- respiratory parameters
resp_amplitude_mm_pp = 20
resp_cpm = 15
resp_func_name = "cosPower4"

# --- cardiac parameters
card_amplitude_mm_pp = 10
card_bpm = 67
card_func_name = "cosPower4"

# -----------------------------------------------------------------------------
# --- determine functions and adjust amplitudes for correct peak-to-peak values
resp_cpm_peaks = 0
card_bpm_peaks = 0

if resp_func_name == "cos" or resp_func_name == "sin":
    resp_func = np.cos if resp_func_name == "cos" else np.sin
    resp_amplitude_mm = resp_amplitude_mm_pp / 2.0
    resp_cpm_peaks = resp_cpm
elif resp_func_name == "cosPower4" or resp_func_name == "sinPower4":
    resp_func = lambda x: np.cos(x)**4 if resp_func_name == "cosPower4" else lambda x: np.sin(x)**4
    resp_amplitude_mm = resp_amplitude_mm_pp
    resp_cpm_peaks = resp_cpm / 2.0  # adjust cpm for power function to maintain same frequency
else:
    raise ValueError(f"Unsupported respiratory function: {resp_func_name}")

if card_func_name == "cos" or card_func_name == "sin":
    card_func = np.cos if card_func_name == "cos" else np.sin
    card_amplitude_mm = card_amplitude_mm_pp / 2.0
    card_bpm_peaks = card_bpm
elif card_func_name == "cosPower2" or card_func_name == "sinPower2":
    card_func = lambda x: np.cos(x)**2 if card_func_name == "cosPower2" else lambda x: np.sin(x)**2
    card_amplitude_mm = card_amplitude_mm_pp
    card_bpm_peaks = card_bpm / 2.0  # adjust bpm for power function to maintain same frequency
elif card_func_name == "cosPower4" or card_func_name == "sinPower4":
    card_func = lambda x: np.cos(x)**4 if card_func_name == "cosPower4" else lambda x: np.sin(x)**4
    card_amplitude_mm = card_amplitude_mm_pp
    card_bpm_peaks = card_bpm / 2.0  # adjust bpm for power function to maintain same frequency
elif card_func_name == "cosPower8" or card_func_name == "sinPower8":
    card_func = lambda x: np.cos(x)**8 if card_func_name == "cosPower8" else lambda x: np.sin(x)**8
    # adjust amplitude for power function to maintain same peak-to-peak
    card_amplitude_mm = card_amplitude_mm_pp
    card_bpm_peaks = card_bpm / 2.0  # adjust bpm for power function to maintain same frequency
else:
    raise ValueError(f"Unsupported cardiac function: {card_func_name}")
# -----------------------------------------------------------------------------

time_increment_ms = 10
total_time_ms = 60 * 60 * 1000  # 60 min


#time_increment_ms = 10
#total_time_ms = 2 * 60 * 1000  # 60 min

resp_period_ms = 60_000.0 / resp_cpm_peaks
card_period_ms = 60_000.0 / card_bpm_peaks

data_pts = round(total_time_ms / time_increment_ms)

t_ms = np.zeros(data_pts)
resp_mm = np.zeros(data_pts)
card_mm = np.zeros(data_pts)
pos_mm = np.zeros(data_pts)

# %% calc positions
for i in range(data_pts):
    t_ms[i] = i * time_increment_ms
    resp_mm[i] = resp_amplitude_mm * resp_func(2.0 * np.pi * t_ms[i] / resp_period_ms)
    card_mm[i] = card_amplitude_mm * card_func(2.0 * np.pi * t_ms[i] / card_period_ms)
    pos_mm[i] = resp_mm[i] + card_mm[i]

# mean = 0
pos_mm -= np.mean(pos_mm)

# normalize for correct phantom input
pos_mm_norm = pos_mm / 15

# %% output paths
out_dir = (
    "/utrecht_exp/results/quasar_traces"
)

os.makedirs(out_dir, exist_ok=True)

qrm_file = os.path.join(
    out_dir, f"{resp_func_name}_{resp_amplitude_mm}mm_pp_{resp_cpm}cpm_{card_func_name}_{card_amplitude_mm}mm_pp_{card_bpm}bpm.qrm"
)
png_file = os.path.join(
    out_dir, f"{resp_func_name}_{resp_amplitude_mm}mm_pp_{resp_cpm}cpm_{card_func_name}_{card_amplitude_mm}mm_pp_{card_bpm}bpm.png"
)

# %% plot and save
plt.figure(figsize=(10, 4))
plt.plot(t_ms / 1000, pos_mm, label='Combined motion')
plt.plot(t_ms / 1000, resp_mm, '-o', label='Respiratory')
plt.plot(t_ms / 1000, card_mm, ':', label='Cardiac')
plt.xlabel('Time [s]')
plt.ylabel('Normalized position')
plt.legend()
plt.xlim(0, 30)
plt.tight_layout()

print(np.max(pos_mm), np.min(pos_mm))
print(np.max(pos_mm_norm), np.min(pos_mm_norm))
#plt.savefig(png_file, dpi=300)

# %% export qrm
with open(qrm_file, 'w') as f:
    f.write('% QUASAR Respiratory + Cardiac motion file\r\n')
    f.write(f'{data_pts}\r\n')
    f.write('1.5\r\n')
    for val in pos_mm_norm:
        f.write(f'{val}\r\n')

