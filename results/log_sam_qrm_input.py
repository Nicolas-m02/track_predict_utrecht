# ------------------ Imports ------------------
import re
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ------------------ Settings ------------------
QRM_TRACE_PATH = "/smb/user/cschorli/DS-Data/Radiotherapie/Research/User/cschorli/data_code/project_munich/track_predict_utrecht/" \
"data/qrm_traces/umcu_volunteer.qrm"


# umcu_heart_trace
# lmu_heart_trace
# modif_lmu_heart_long
# modif_umcu_heart_long

# sin_10.0mm_pp_15cpm
# cosPower4_15mm_pp_15cpm_cosPower4_5mm_pp_67bpm


#QRM_TRACE_PATH = "utrecht_exp/data/qrm_traces/cosPower4_15mm_pp_15cpm_cosPower4_5mm_pp_67bpm.qrm"


SPEC_LOG = "volunteer_umcu_circle_20260701T162735.348864" # specify a log directory name to use, otherwise the latest

SAM_LSTM = "lstm" # sam or lstm

SEARCH_SHIFT = False # search for time shift between reference and estimated motion
MANUAL_SHIFT = 0.0 # if SEARCH_SHIFT is False, use this manual time shift [s]

ANGLE_DEG_PHANTOM = 0               # phantom rotation angle [deg]

time_start = 5.0
time_end = 300.0

TRACKING_MODE = "SAG" # NONE | SAG | BEV

EXCLUDE_AFTER_ANGLE_CHANGE_S = 1   # exclude time for plotting and calculatio of error after angle changed [s]


PLOT_DIM = 1 # 1D or 2D plot of motion (1 or 2)
PLOT_DIRECTION = "y" # x or y direction to plot in 1D mode


SAVE_IMG = True



plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})


# ------------------ Reporting ------------------
class Report:
    def __init__(self):
        self.sections = []

    def add(self, title, rows):
        self.sections.append((title, rows))

    def _fmt(self, v, unit=""):
        if isinstance(v, float):
            return f"{v:8.3f}{unit}"
        return str(v)

    def print(self):
        width = 72
        print("\n" + "="*width)
        print(" MOTION ESTIMATION ANALYSIS REPORT ".center(width, "="))
        print("="*width)

        for title, rows in self.sections:
            print(f"\n{title}")
            print("-"*width)
            for k, v in rows:
                if isinstance(v, tuple):
                    value, unit = v
                    print(f"{k:<40}{self._fmt(value, unit)}")
                else:
                    print(f"{k:<40}{v}")

        print("\n" + "="*width + "\n")


report = Report()


# ------------------ Helpers ------------------
def parse_timestamp(ts: str) -> np.datetime64:
    """Parse log timestamp of the form 'YYYYMMDDTHHMMSS.xxxxxx'."""
    return np.datetime64(f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:]}")


def load_quasar_trace(path: str, phantom_scale: float = 15.0) -> np.ndarray:
    """Load a QUASAR motion trace and return displacement in mm."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"QUASAR motion trace file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines or not lines[0].startswith("%"):
        raise ValueError(f"Unexpected QUASAR trace format in {path}")
    n_samples = int(lines[1])
    values = np.asarray(lines[3:3 + n_samples], dtype=float)

    if len(values) != n_samples:
        raise ValueError(f"Expected {n_samples} samples, found {len(values)}")

    # undo normalization from the generation script
    values *= 7.5# 15 phantom_scale ###############################################################################################################
    values += 3             ###################################################################################################################

    return values

# ------------------ Log File Selection ------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, ".."))

log_dirs = [
    d for d in glob.glob(os.path.join(parent_dir, "logs", "*"))
    if os.path.isdir(d)
]

if not log_dirs:
    raise FileNotFoundError("No timestamp directories found in 'logs/'.")

if SPEC_LOG != "latest":
    latest_log_dir = os.path.join(parent_dir, "logs", SPEC_LOG)
    if not os.path.isdir(latest_log_dir):
        raise FileNotFoundError(f"Specified log directory '{SPEC_LOG}' does not exist.")
elif SPEC_LOG == "latest":
    latest_log_dir = max(log_dirs, key=os.path.getmtime)

if SAM_LSTM == "sam":
    log_subdir = "tracking_module/mri*"
elif SAM_LSTM == "lstm":
    log_subdir = "lstm/pred_log_file*"
txt_files = glob.glob(os.path.join(latest_log_dir, f"{log_subdir}"))

if not txt_files:
    raise FileNotFoundError(
        f"No .txt file found in '{latest_log_dir}/{log_subdir}'."
    )

if len(txt_files) > 1:
    print(f"Warning: Multiple .txt files found in '{latest_log_dir}/{log_subdir}'. Using the first one.")
log_file = txt_files[0]

report.add("Input", [
    ("Log file directory", os.path.basename(latest_log_dir)),
    ("Log file", os.path.basename(log_file)),
])

# ------------------ Data Parsing ------------------
mean_x, mean_y, angles, angle_idx, time_vals = [], [], [], [], []
current_ts, current_img_ms = None, None
pending_angle = None      # saves angle printed prior in same loop as mean_y/x
first_data_seen = False
idx = -1

if SAM_LSTM == "sam":
    patterns = {
        "x": re.compile(r"mean_x:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "y": re.compile(r"mean_y:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "angle": re.compile(r"gantry angle is currently:\s*templ_at_angle_(\d+)"),
        "ts": re.compile(r"^(\d{8}T\d{6}\.\d+)"),
        "img": re.compile(r"image handling time:\s*([\d.]+)\s*ms")
    }
elif SAM_LSTM == "lstm":
    patterns = {
        "x": re.compile(r"mean_x:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "y": re.compile(r"mean_y:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "angle": re.compile(r"gantry angle \(placeholder for now\) is currently:\s*templ_at_angle_(\d+)"),
        "ts": re.compile(r"^(\d{8}T\d{6}\.\d+)"),
        "img": re.compile(r"image handling time:\s*([\d.]+)\s*ms")
    }
with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # Timestamp
        ts_match = patterns["ts"].search(line)
        if ts_match:
            current_ts = ts_match.group(1)


        # Image handling time
        #img_match = patterns["img"].search(line)
        #if img_match:
        #   current_img_ms = float(img_match.group(1))

        # mean_x / mean_y
        match_x = patterns["x"].search(line)
        match_y = patterns["y"].search(line)
        if match_x or match_y:
            first_data_seen = True  # start parsing 
            idx += 1  # new data point
            if match_x:
                mean_x.append(float(match_x.group(1)))
            if match_y:
                mean_y.append(float(match_y.group(1)))
                if current_ts: 
                    t = parse_timestamp(current_ts) 
                    #if current_img_ms is not None:
                    #    t -= np.timedelta64(int(current_img_ms * 1e6), "ns") 
                    time_vals.append(t)

            # Assign pending angle
            if pending_angle is not None:
                angles.append(pending_angle)
                angle_idx.append(min(idx, len(mean_x)-1))
                pending_angle = None


        # Store new angle for next data point
        match_angle = patterns["angle"].search(line)
        if match_angle and first_data_seen:
            pending_angle = int(match_angle.group(1))

            
# ------------------ Prepare Arrays ------------------
if TRACKING_MODE == "SAG" or TRACKING_MODE == "NONE":
    mean_x = np.zeros_like(mean_x)
elif TRACKING_MODE == "BEV":
    mean_x = np.array(mean_x)
else:
    raise RuntimeError("Choose for TRACKING_MODE either NONE, SAG or BEV")

if TRACKING_MODE == "NONE":
    mean_y = np.zeros_like(mean_y)
elif TRACKING_MODE == "SAG" or TRACKING_MODE == "BEV":
    mean_y = np.array(mean_y)
    # invert Y -> FH direction, H is positive in patient coordinates. 
    # but y+ is downwards as in img first px is top left corner
    mean_y = -mean_y  

angles = np.array(angles)
angle_idx = np.array(angle_idx)
time_vals = np.array(time_vals)

t0 = time_vals[0]
time_s = (time_vals - t0) / np.timedelta64(1, "s")
samples = np.arange(len(mean_x))

angle_rad = np.deg2rad(ANGLE_DEG_PHANTOM)

# ------------------ Reference Generation ------------------
qrm_trace = load_quasar_trace(QRM_TRACE_PATH)
dt_ms = 10  # time increment in ms
qrm_time = np.arange(len(qrm_trace)) * (dt_ms / 1000.0)



# -- start fitting only in between time interval ---

fit_mask_y = (time_s >= time_start) & (time_s <= time_end)

if not np.any(fit_mask_y):
    raise RuntimeError("No samples found in the requested fitting interval.")

t_fit = time_s[fit_mask_y]
y_fit = mean_y[fit_mask_y]

# ---- phantom trace ----
def phantom_trace(t):
    t_arr = np.asarray(t, dtype=float)
    return np.interp(t_arr, qrm_time, qrm_trace)


#################################################################################################

if SEARCH_SHIFT:
    # ---- estimate global time shift τ ----
    search_tau = np.linspace(0, 100, 30000)

    best_tau = 0.0
    best_err = np.inf

    for tau in search_tau:
        err = np.mean((y_fit - phantom_trace(t_fit - tau)*np.cos(angle_rad))**2)
        if err < best_err:
            best_err = err
            best_tau = tau

    report.add("Signal Synchronization", [
        ("Estimated time shift τ", (best_tau, " s")),
        ("Reference trace", QRM_TRACE_PATH),
    ])
else:
    best_tau = MANUAL_SHIFT
    report.add("Signal Synchronization", [
        ("Time shift τ", (best_tau, " s")),
        ("Reference trace", QRM_TRACE_PATH),
    ])    
# ----------- build continuous angle segments -----------
angle_segments = []

if len(angles) > 0:
    seg_start = angle_idx[0]
    prev_angle = angles[0]

    for i in range(1, len(angles)):
        if angles[i] != prev_angle:
            angle_segments.append((prev_angle, seg_start, angle_idx[i]))
            prev_angle = angles[i]
            seg_start = angle_idx[i]

    angle_segments.append((prev_angle, seg_start, len(mean_x)))

# ---- synchronized reference ----      
sinus_ref = phantom_trace(time_s - best_tau)

ref_x = np.zeros_like(mean_x)
ref_y = np.zeros_like(mean_y)



for angle, i0, i1 in angle_segments:

    angle_r = np.deg2rad(angle)
    seg_slice = slice(i0, i1)
    # build reference
    ref_x[seg_slice] = sinus_ref[seg_slice] * np.sin(angle_rad) * np.cos(angle_r)
    ref_y[seg_slice] = sinus_ref[seg_slice] * np.cos(angle_rad)


# ------------------ Exclusion Mask ------------------
exclude_mask = np.zeros_like(time_s, dtype=bool)
if len(angles) > 0:
    prev_angle = angles[0]

    for i in range(1, len(angles)):
        if angles[i] != prev_angle:          # angle change only
            idx = angle_idx[i]
            if 0 <= idx < len(time_s):
                t0 = time_s[idx]
                t1 = t0 + EXCLUDE_AFTER_ANGLE_CHANGE_S
                exclude_mask |= (time_s >= t0) & (time_s <= t1)
            prev_angle = angles[i]


fit_mask_y = (time_s >= time_start) & (time_s <= time_end)
if not np.any(fit_mask_y):
    raise RuntimeError("No samples found in the requested fitting interval.")
else:
    valid_eval_mask = fit_mask_y
    valid_eval_mask &= ~exclude_mask

# ------------------ Compute Deviations ------------------
if not np.any(valid_eval_mask):
    print("No valid data after threshold and angle-change exclusion → skipping evaluation")
else:
    dev_x = np.abs(mean_x[valid_eval_mask] - ref_x[valid_eval_mask])
    dev_y = np.abs(mean_y[valid_eval_mask] - ref_y[valid_eval_mask])

    rmse_x = np.sqrt(np.mean(dev_x**2))
    rmse_y = np.sqrt(np.mean(dev_y**2))
    report.add("Tracking Accuracy (Raw)", [
        ("RMSE X", (rmse_x, " mm")),
        ("Mean |X error|", (np.mean(dev_x), " mm")),
        ("Std  |X error|", (np.std(dev_x), " mm")),
        ("RMSE Y", (rmse_y, " mm")),
        ("Mean |Y error|", (np.mean(dev_y), " mm")),
        ("Std  |Y error|", (np.std(dev_y), " mm")),
    ])


# ------------------ Interpolation ------------------
t_fine = np.linspace(time_s[0], time_s[-1], 1000 * len(time_s))
interp_ref_x = interp1d(time_s, ref_x, kind="linear")
interp_ref_y = interp1d(time_s, ref_y, kind="linear")
ref_x_fine, ref_y_fine = interp_ref_x(t_fine), interp_ref_y(t_fine)
exclude_mask_fine = np.zeros_like(t_fine, dtype=bool)

if len(angles) > 0:
    prev_angle = angles[0]
    for i in range(1, len(angles)):
        if angles[i] != prev_angle:
            idx = angle_idx[i]
            if 0 <= idx < len(time_s):
                t0 = time_s[idx]
                t1 = t0 + EXCLUDE_AFTER_ANGLE_CHANGE_S
                exclude_mask_fine |= (t_fine >= t0) & (t_fine <= t1)
            prev_angle = angles[i]


interp_mean_x = interp1d(time_s, mean_x, kind="linear")
interp_mean_y = interp1d(time_s, mean_y, kind="linear")
mean_x_fine, mean_y_fine = interp_mean_x(t_fine), interp_mean_y(t_fine)



fit_mask_y_fine = (t_fine >= time_start) & (t_fine <= time_end)
if not np.any(fit_mask_y_fine):
    raise RuntimeError("No samples found in the requested fitting interval.")
else:
    valid_eval_mask_fine = fit_mask_y_fine
    valid_eval_mask_fine &= ~exclude_mask_fine


if np.any(valid_eval_mask_fine):
    dev_x_fine = np.abs(mean_x_fine[valid_eval_mask_fine] - ref_x_fine[valid_eval_mask_fine])
    dev_y_fine = np.abs(mean_y_fine[valid_eval_mask_fine] - ref_y_fine[valid_eval_mask_fine])
    # Total (Euclidean) error per point
    dev_total_fine = np.sqrt(dev_x_fine**2 + dev_y_fine**2)

    report.add("Tracking Accuracy (Interpolated)", [
        ("RMSE X", (np.sqrt(np.mean(dev_x_fine**2)), " mm")),
        ("Mean |X error|", (np.mean(dev_x_fine), " mm")),
        ("Std  |X error|", (np.std(dev_x_fine), " mm")),
        ("RMSE Y", (np.sqrt(np.mean(dev_y_fine**2)), " mm")),
        ("Mean |Y error|", (np.mean(dev_y_fine), " mm")),
        ("Std  |Y error|", (np.std(dev_y_fine), " mm")),
        ("RMSE Total", (np.sqrt(np.mean(dev_total_fine**2)), " mm")),
        ("Mean |Total error|", (np.mean(dev_total_fine), " mm")),
        ("Std  |Total error|", (np.std(dev_total_fine), " mm")),
    ])




# ------------------ Plotting ------------------
if PLOT_DIM == 1 and PLOT_DIRECTION == "x":
    fig, (ax_x, ax_ex) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 0.3]}
    )
elif PLOT_DIM == 1 and PLOT_DIRECTION == "y":
    fig, (ax_y, ax_ey) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 0.3]}
    )
elif PLOT_DIM == 2:
    fig, (ax_x, ax_ex, ax_y, ax_ey) = plt.subplots(
        4, 1, sharex=True, figsize=(12, 10),
        gridspec_kw={"height_ratios": [1, 0.3, 1, 0.3]}
    )

# ---------- Build plotting masks ----------
plot_mask = fit_mask_y & ~exclude_mask
plot_mask_fine = fit_mask_y_fine & ~exclude_mask_fine

# ---------- Masked versions for plotting ----------
ref_x_fine_plot = np.full_like(ref_x_fine, np.nan)
ref_y_fine_plot = np.full_like(ref_y_fine, np.nan)
ref_x_fine_plot[plot_mask_fine] = ref_x_fine[plot_mask_fine]
ref_y_fine_plot[plot_mask_fine] = ref_y_fine[plot_mask_fine]

ref_x_plot = np.full_like(ref_x, np.nan)
ref_y_plot = np.full_like(ref_y, np.nan)
ref_x_plot[plot_mask] = ref_x[plot_mask]
ref_y_plot[plot_mask] = ref_y[plot_mask]


if PLOT_DIM == 1 and PLOT_DIRECTION == "x" or PLOT_DIM == 2:
    # X position 
    ax_x.plot(time_s, mean_x, label="Estimated X motion", linewidth=1.8)
    #ax_x.plot(time_s, ref_x_plot, linestyle="--", label="Reference X motion")
    ax_x.plot(t_fine, ref_x_fine_plot, color="green", linewidth=1.2, label="Interpolated Ref X motion")
    ax_x.set_ylabel("Position X [mm]")
    ax_x.set_title("Motion Estimation Performance")
    ax_x.grid(True, alpha=0.5)
    ax_x.legend(loc="upper left")

if PLOT_DIM == 1 and PLOT_DIRECTION == "y" or PLOT_DIM == 2:
    # Y position 
    ax_y.plot(time_s, mean_y, label="Estimated Y motion", linewidth=1.8)
    #ax_y.scatter(time_s, mean_y, label="Estimated Y motion")
    #ax_y.plot(time_s, ref_y_plot, linestyle="--", label="Reference Y motion")
    ax_y.plot(t_fine, ref_y_fine_plot, color="green", linewidth=1.2, label="Interpolated Ref Y motion")
    ax_y.set_ylabel("Position Y [mm]")
    ax_y.grid(True, alpha=0.5)
    ax_y.legend(loc="upper left")

# Errors 
if len(angles) > 0:
    prev_angle = angles[0]
    for i in range(1, len(angles)):
        if angles[i] != prev_angle:
            idx = angle_idx[i]
            if 0 <= idx < len(time_s):
                t0 = time_s[idx]
                t1 = t0 + EXCLUDE_AFTER_ANGLE_CHANGE_S
                if PLOT_DIM == 1 and PLOT_DIRECTION == "x" or PLOT_DIM == 2:
                    ax_ex.axvspan(t0, t1, color="gray", alpha=0.25, zorder=0)
                if PLOT_DIM == 1 and PLOT_DIRECTION == "y" or PLOT_DIM == 2:
                    ax_ey.axvspan(t0, t1, color="gray", alpha=0.25, zorder=0)
            prev_angle = angles[i]

# Interpolated errors
err_x_fine = mean_x_fine - ref_x_fine
err_y_fine = mean_y_fine - ref_y_fine

# Interpolated errors
err_x_fine_plot = np.full_like(err_x_fine, np.nan)
err_y_fine_plot = np.full_like(err_y_fine, np.nan)
err_x_fine_plot[valid_eval_mask_fine] = err_x_fine[valid_eval_mask_fine]
err_y_fine_plot[valid_eval_mask_fine] = err_y_fine[valid_eval_mask_fine]



if PLOT_DIM == 1 and PLOT_DIRECTION == "x" or PLOT_DIM == 2:
    if np.any(valid_eval_mask_fine):
        ax_ex.plot(
            t_fine, err_x_fine_plot,
            linestyle="-", color="red",linewidth=0.5,
            label=f"x error (RMSE={np.sqrt(np.mean(dev_x_fine**2)):.2f} mm)"
        )

    ax_ex.set_ylabel("Error X [mm]")
    ax_ex.grid(True, alpha=0.5)
    ax_ex.set_ylim(-2, 2)
    #ax_ex.set_xlim(30, 47)
    ax_ex.axhline(0, linestyle="--", linewidth=1)

    ax_ex.legend(loc="upper left")


    ax_ex.axhline(y=1, color="coral", linestyle='-', linewidth=0.8)
    ax_ex.axhline(y=-1, color="coral", linestyle='-', linewidth=0.8)
    ax_ex.text(1.0, 1, '+1 mm', transform=ax_ex.get_yaxis_transform(), va='bottom', ha='right', fontsize=7)
    ax_ex.text(1.0, -1, '-1 mm', transform=ax_ex.get_yaxis_transform(), va='top', ha='right', fontsize=7)

    ax_x.text(
        0.99, 1.02,
        "Red lines / labels = gantry angle",
        transform=ax_x.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        color="red"
    )

if PLOT_DIM == 1 and PLOT_DIRECTION == "y" or PLOT_DIM == 2:
    if np.any(valid_eval_mask_fine):
        ax_ey.plot(
            t_fine, err_y_fine_plot,
            linestyle="-", color="red", linewidth=0.5,
            label=f"y error (RMSE={np.sqrt(np.mean(dev_y_fine**2)):.2f} mm)"
        )


    ax_ey.set_ylabel("Error Y [mm]")
    ax_ey.set_xlabel("Time [s]")

    ax_ey.grid(True, alpha=0.5)

    ax_ey.set_ylim(-2, 2)
    #ax_ey.set_xlim(30, 47)
    ax_ey.legend(loc="upper left")

    ax_ey.axhline(0, linestyle="--", linewidth=1)

    ax_ey.axhline(y=1, color="coral", linestyle='-', linewidth=0.8)
    ax_ey.axhline(y=-1, color="coral", linestyle='-', linewidth=0.8)
    ax_ey.text(1.0, 1, '+1 mm', transform=ax_ey.get_yaxis_transform(), va='bottom', ha='right', fontsize=7)
    ax_ey.text(1.0, -1, '-1 mm', transform=ax_ey.get_yaxis_transform(), va='top', ha='right', fontsize=7)





# Angle markers 
if len(angles) and len(samples) > 0:
    first_angle_pos = 0
    while first_angle_pos < len(angle_idx) and angle_idx[first_angle_pos] < 0:
        first_angle_pos += 1

    if first_angle_pos < len(angles):
        prev_angle = angles[first_angle_pos]
        start_idx = angle_idx[first_angle_pos]

        for i in range(first_angle_pos + 1, len(angles)):
            angle = angles[i]
            idx = angle_idx[i]
            if angle != prev_angle:
                if PLOT_DIM == 1 and PLOT_DIRECTION == "x" or PLOT_DIM == 2:
                    for ax in [ax_x, ax_ex]:
                        ax.axvline(x=time_s[start_idx], color="red",
                                linestyle=":", alpha=0.7)

                if PLOT_DIM == 1 and PLOT_DIRECTION == "y" or PLOT_DIM == 2:
                    for ax in [ax_y, ax_ey]:
                        ax.axvline(x=time_s[start_idx], color="red",
                                linestyle=":", alpha=0.7)
                mid_x = (time_s[start_idx] + time_s[idx - 1]) / 2
                ax_x.text(mid_x, ax_x.get_ylim()[1]*0.95, f"{prev_angle}°",
                          color="red", ha="center", va="top")
                prev_angle = angle
                start_idx = idx
        if PLOT_DIM == 1 and PLOT_DIRECTION == "x" or PLOT_DIM == 2:
            for ax in [ax_x, ax_ex]:
                ax.axvline(x=time_s[start_idx], color="red",
                        linestyle=":", alpha=0.7)
        if PLOT_DIM == 1 and PLOT_DIRECTION == "y" or PLOT_DIM == 2:
            for ax in [ax_y, ax_ey]:
                ax.axvline(x=time_s[start_idx], color="red",
                        linestyle=":", alpha=0.7)


report.print()


# Layout & save 
fig.suptitle(f"Motion Estimation vs Reference — {os.path.basename(log_file)}",y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.97])
if SAVE_IMG:
    plt.savefig(f"{base_dir}/motion_estimation_analysis_log_file.png",dpi=300)
plt.show()

# ------------------ Imports ------------------
import re
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ------------------ Settings ------------------
QRM_TRACE_PATH = "/smb/user/cschorli/DS-Data/Radiotherapie/Research/User/cschorli/data_code/project_munich/track_predict_utrecht/" \
"data/qrm_traces/umcu_heart_trace.qrm"


# umcu_heart_trace
# lmu_heart_trace
# modif_lmu_heart_long
# modif_umcu_heart_long

# sin_10.0mm_pp_15cpm
# cosPower4_15mm_pp_15cpm_cosPower4_5mm_pp_67bpm


#QRM_TRACE_PATH = "utrecht_exp/data/qrm_traces/cosPower4_15mm_pp_15cpm_cosPower4_5mm_pp_67bpm.qrm"


SPEC_LOG = "volunteer_lmu_circle_20260701T164753.834233" # specify a log directory name to use, otherwise the latest

SAM_LSTM = "sam" # sam or lstm

ANGLE_DEG_PHANTOM = 0               # phantom rotation angle [deg]

time_start = 5.0
time_end = 137.0

TRACKING_MODE = "BEV" # NONE | SAG | BEV

EXCLUDE_AFTER_ANGLE_CHANGE_S = 1   # exclude time for plotting and calculatio of error after angle changed [s]


PLOT_DIM = 1 # 1D or 2D plot of motion (1 or 2)
PLOT_DIRECTION = "y" # x or y direction to plot in 1D mode


SAVE_IMG = True



plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})


# ------------------ Reporting ------------------
class Report:
    def __init__(self):
        self.sections = []

    def add(self, title, rows):
        self.sections.append((title, rows))

    def _fmt(self, v, unit=""):
        if isinstance(v, float):
            return f"{v:8.3f}{unit}"
        return str(v)

    def print(self):
        width = 72
        print("\n" + "="*width)
        print(" MOTION ESTIMATION ANALYSIS REPORT ".center(width, "="))
        print("="*width)

        for title, rows in self.sections:
            print(f"\n{title}")
            print("-"*width)
            for k, v in rows:
                if isinstance(v, tuple):
                    value, unit = v
                    print(f"{k:<40}{self._fmt(value, unit)}")
                else:
                    print(f"{k:<40}{v}")

        print("\n" + "="*width + "\n")


report = Report()


# ------------------ Helpers ------------------
def parse_timestamp(ts: str) -> np.datetime64:
    """Parse log timestamp of the form 'YYYYMMDDTHHMMSS.xxxxxx'."""
    return np.datetime64(f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:]}")


def load_quasar_trace(path: str, phantom_scale: float = 15.0) -> np.ndarray:
    """Load a QUASAR motion trace and return displacement in mm."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"QUASAR motion trace file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines or not lines[0].startswith("%"):
        raise ValueError(f"Unexpected QUASAR trace format in {path}")
    n_samples = int(lines[1])
    values = np.asarray(lines[3:3 + n_samples], dtype=float)

    if len(values) != n_samples:
        raise ValueError(f"Expected {n_samples} samples, found {len(values)}")

    # undo normalization from the generation script
    values *= phantom_scale


    return values

# ------------------ Log File Selection ------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, ".."))

log_dirs = [
    d for d in glob.glob(os.path.join(parent_dir, "logs", "*"))
    if os.path.isdir(d)
]

if not log_dirs:
    raise FileNotFoundError("No timestamp directories found in 'logs/'.")

if SPEC_LOG != "latest":
    latest_log_dir = os.path.join(parent_dir, "logs", SPEC_LOG)
    if not os.path.isdir(latest_log_dir):
        raise FileNotFoundError(f"Specified log directory '{SPEC_LOG}' does not exist.")
elif SPEC_LOG == "latest":
    latest_log_dir = max(log_dirs, key=os.path.getmtime)

if SAM_LSTM == "sam":
    log_subdir = "tracking_module/mri*"
elif SAM_LSTM == "lstm":
    log_subdir = "lstm/pred_log_file*"
txt_files = glob.glob(os.path.join(latest_log_dir, f"{log_subdir}"))

if not txt_files:
    raise FileNotFoundError(
        f"No .txt file found in '{latest_log_dir}/{log_subdir}'."
    )

if len(txt_files) > 1:
    print(f"Warning: Multiple .txt files found in '{latest_log_dir}/{log_subdir}'. Using the first one.")
log_file = txt_files[0]

report.add("Input", [
    ("Log file directory", os.path.basename(latest_log_dir)),
    ("Log file", os.path.basename(log_file)),
])

# ------------------ Data Parsing ------------------
mean_x, mean_y, angles, angle_idx, time_vals = [], [], [], [], []
current_ts, current_img_ms = None, None
pending_angle = None      # saves angle printed prior in same loop as mean_y/x
first_data_seen = False
idx = -1


patterns = {
    "x": re.compile(r"mean_x:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "y": re.compile(r"mean_y:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "angle": re.compile(r"gantry angle is currently:\s*templ_at_angle_(\d+)"),
    "ts": re.compile(r"^(\d{8}T\d{6}\.\d+)"),
    "img": re.compile(r"image handling time:\s*([\d.]+)\s*ms")
}
with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # Timestamp
        ts_match = patterns["ts"].search(line)
        if ts_match:
            current_ts = ts_match.group(1)


        # Image handling time
        #img_match = patterns["img"].search(line)
        #if img_match:
        #   current_img_ms = float(img_match.group(1))

        # mean_x / mean_y
        match_x = patterns["x"].search(line)
        match_y = patterns["y"].search(line)
        if match_x or match_y:
            first_data_seen = True  # start parsing 
            idx += 1  # new data point
            if match_x:
                mean_x.append(float(match_x.group(1)))
            if match_y:
                mean_y.append(float(match_y.group(1)))
                if current_ts: 
                    t = parse_timestamp(current_ts) 
                    #if current_img_ms is not None:
                    #    t -= np.timedelta64(int(current_img_ms * 1e6), "ns") 
                    time_vals.append(t)

            # Assign pending angle
            if pending_angle is not None:
                angles.append(pending_angle)
                angle_idx.append(min(idx, len(mean_x)-1))
                pending_angle = None


        # Store new angle for next data point
        match_angle = patterns["angle"].search(line)
        if match_angle and first_data_seen:
            pending_angle = int(match_angle.group(1))

            
# ------------------ Prepare Arrays ------------------
if TRACKING_MODE == "SAG" or TRACKING_MODE == "NONE":
    mean_x = np.zeros_like(mean_x)
elif TRACKING_MODE == "BEV":
    mean_x = np.array(mean_x)
else:
    raise RuntimeError("Choose for TRACKING_MODE either NONE, SAG or BEV")

if TRACKING_MODE == "NONE":
    mean_y = np.zeros_like(mean_y)
elif TRACKING_MODE == "SAG" or TRACKING_MODE == "BEV":
    mean_y = np.array(mean_y)
    # invert Y -> FH direction, H is positive in patient coordinates. 
    # but y+ is downwards as in img first px is top left corner
    mean_y = -mean_y  

angles = np.array(angles)
angle_idx = np.array(angle_idx)
time_vals = np.array(time_vals)

t0 = time_vals[0]
time_s = (time_vals - t0) / np.timedelta64(1, "s")
samples = np.arange(len(mean_x))

angle_rad = np.deg2rad(ANGLE_DEG_PHANTOM)

# ------------------ Reference Generation ------------------
qrm_trace = load_quasar_trace(QRM_TRACE_PATH)
dt_ms = 10  # time increment in ms
qrm_time = np.arange(len(qrm_trace)) * (dt_ms / 1000.0)



# -- start fitting only in between time interval ---

fit_mask_y = (time_s >= time_start) & (time_s <= time_end)

if not np.any(fit_mask_y):
    raise RuntimeError("No samples found in the requested fitting interval.")

t_fit = time_s[fit_mask_y]
y_fit = mean_y[fit_mask_y]


# ---- phantom trace ----
def phantom_trace(t):
    t_arr = np.asarray(t, dtype=float)
    return np.interp(t_arr, qrm_time, qrm_trace, left=qrm_trace[0], right=qrm_trace[-1])


#################################################################################################

# ---- estimate global time shift τ ----
search_tau = np.linspace(0, 60, 20000)

best_tau = 0.0
best_err = np.inf

for tau in search_tau:
    err = np.mean((y_fit - phantom_trace(t_fit - tau)*np.cos(angle_rad))**2)
    if err < best_err:
        best_err = err
        best_tau = tau

report.add("Signal Synchronization", [
    ("Estimated time shift τ", (best_tau, " s")),
    ("Reference trace", QRM_TRACE_PATH),
])

# ----------- build continuous angle segments -----------
angle_segments = []

if len(angles) > 0:
    seg_start = angle_idx[0]
    prev_angle = angles[0]

    for i in range(1, len(angles)):
        if angles[i] != prev_angle:
            angle_segments.append((prev_angle, seg_start, angle_idx[i]))
            prev_angle = angles[i]
            seg_start = angle_idx[i]

    angle_segments.append((prev_angle, seg_start, len(mean_x)))

# ---- synchronized reference ----      
sinus_ref = phantom_trace(time_s - best_tau)

ref_x = np.zeros_like(mean_x)
ref_y = np.zeros_like(mean_y)




for angle, i0, i1 in angle_segments:

    angle_r = np.deg2rad(angle)
    seg_slice = slice(i0, i1)

    # build reference
    ref_x[seg_slice] = sinus_ref[seg_slice] * np.sin(angle_rad) * np.cos(angle_r)
    ref_y[seg_slice] = sinus_ref[seg_slice] * np.cos(angle_rad)  ####################################################


# ------------------ Exclusion Mask ------------------
exclude_mask = np.zeros_like(time_s, dtype=bool)
if len(angles) > 0:
    prev_angle = angles[0]

    for i in range(1, len(angles)):
        if angles[i] != prev_angle:          # angle change only
            idx = angle_idx[i]
            if 0 <= idx < len(time_s):
                t0 = time_s[idx]
                t1 = t0 + EXCLUDE_AFTER_ANGLE_CHANGE_S
                exclude_mask |= (time_s >= t0) & (time_s <= t1)
            prev_angle = angles[i]


fit_mask_y = (time_s >= time_start) & (time_s <= time_end)
if not np.any(fit_mask_y):
    raise RuntimeError("No samples found in the requested fitting interval.")
else:
    valid_eval_mask = fit_mask_y
    valid_eval_mask &= ~exclude_mask

# ------------------ Compute Deviations ------------------
if not np.any(valid_eval_mask):
    print("No valid data after threshold and angle-change exclusion → skipping evaluation")
else:
    dev_x = np.abs(mean_x[valid_eval_mask] - ref_x[valid_eval_mask])
    dev_y = np.abs(mean_y[valid_eval_mask] - ref_y[valid_eval_mask])

    rmse_x = np.sqrt(np.mean(dev_x**2))
    rmse_y = np.sqrt(np.mean(dev_y**2))
    report.add("Tracking Accuracy (Raw)", [
        ("RMSE X", (rmse_x, " mm")),
        ("Mean |X error|", (np.mean(dev_x), " mm")),
        ("Std  |X error|", (np.std(dev_x), " mm")),
        ("RMSE Y", (rmse_y, " mm")),
        ("Mean |Y error|", (np.mean(dev_y), " mm")),
        ("Std  |Y error|", (np.std(dev_y), " mm")),
    ])


# ------------------ Interpolation ------------------
t_fine = np.linspace(time_s[0], time_s[-1], 100 * len(time_s))
interp_ref_x = interp1d(time_s, ref_x, kind="cubic")
interp_ref_y = interp1d(time_s, ref_y, kind="cubic")
ref_x_fine, ref_y_fine = interp_ref_x(t_fine), interp_ref_y(t_fine)
exclude_mask_fine = np.zeros_like(t_fine, dtype=bool)

if len(angles) > 0:
    prev_angle = angles[0]
    for i in range(1, len(angles)):
        if angles[i] != prev_angle:
            idx = angle_idx[i]
            if 0 <= idx < len(time_s):
                t0 = time_s[idx]
                t1 = t0 + EXCLUDE_AFTER_ANGLE_CHANGE_S
                exclude_mask_fine |= (t_fine >= t0) & (t_fine <= t1)
            prev_angle = angles[i]


interp_mean_x = interp1d(time_s, mean_x, kind="cubic")
interp_mean_y = interp1d(time_s, mean_y, kind="cubic")
mean_x_fine, mean_y_fine = interp_mean_x(t_fine), interp_mean_y(t_fine)



fit_mask_y_fine = (t_fine >= time_start) & (t_fine <= time_end)
if not np.any(fit_mask_y_fine):
    raise RuntimeError("No samples found in the requested fitting interval.")
else:
    valid_eval_mask_fine = fit_mask_y_fine
    valid_eval_mask_fine &= ~exclude_mask_fine


if np.any(valid_eval_mask_fine):
    dev_x_fine = np.abs(mean_x_fine[valid_eval_mask_fine] - ref_x_fine[valid_eval_mask_fine])
    dev_y_fine = np.abs(mean_y_fine[valid_eval_mask_fine] - ref_y_fine[valid_eval_mask_fine])
    # Total (Euclidean) error per point
    dev_total_fine = np.sqrt(dev_x_fine**2 + dev_y_fine**2)

    report.add("Tracking Accuracy (Interpolated)", [
        ("RMSE X", (np.sqrt(np.mean(dev_x_fine**2)), " mm")),
        ("Mean |X error|", (np.mean(dev_x_fine), " mm")),
        ("Std  |X error|", (np.std(dev_x_fine), " mm")),
        ("RMSE Y", (np.sqrt(np.mean(dev_y_fine**2)), " mm")),
        ("Mean |Y error|", (np.mean(dev_y_fine), " mm")),
        ("Std  |Y error|", (np.std(dev_y_fine), " mm")),
        ("RMSE Total", (np.sqrt(np.mean(dev_total_fine**2)), " mm")),
        ("Mean |Total error|", (np.mean(dev_total_fine), " mm")),
        ("Std  |Total error|", (np.std(dev_total_fine), " mm")),
    ])




# ------------------ Plotting ------------------
if PLOT_DIM == 1 and PLOT_DIRECTION == "x":
    fig, (ax_x, ax_ex) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 0.3]}
    )
elif PLOT_DIM == 1 and PLOT_DIRECTION == "y":
    fig, (ax_y, ax_ey) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 0.3]}
    )
elif PLOT_DIM == 2:
    fig, (ax_x, ax_ex, ax_y, ax_ey) = plt.subplots(
        4, 1, sharex=True, figsize=(12, 10),
        gridspec_kw={"height_ratios": [1, 0.3, 1, 0.3]}
    )


# ---------- Build plotting masks ----------
plot_mask = fit_mask_y & ~exclude_mask
plot_mask_fine = fit_mask_y_fine & ~exclude_mask_fine

# ---------- Masked versions for plotting ----------
ref_x_fine_plot = np.full_like(ref_x_fine, np.nan)
ref_y_fine_plot = np.full_like(ref_y_fine, np.nan)
ref_x_fine_plot[plot_mask_fine] = ref_x_fine[plot_mask_fine]
ref_y_fine_plot[plot_mask_fine] = ref_y_fine[plot_mask_fine]

ref_x_plot = np.full_like(ref_x, np.nan)
ref_y_plot = np.full_like(ref_y, np.nan)
ref_x_plot[plot_mask] = ref_x[plot_mask]
ref_y_plot[plot_mask] = ref_y[plot_mask]


if PLOT_DIM == 1 and PLOT_DIRECTION == "x" or PLOT_DIM == 2:
    # X position 
    ax_x.plot(time_s, mean_x, label="Estimated X motion", linewidth=1.8)
    #ax_x.plot(time_s, ref_x_plot, linestyle="--", label="Reference X motion")
    ax_x.plot(t_fine, ref_x_fine_plot, color="green", linewidth=1.2, label="Interpolated Ref X motion")
    ax_x.set_ylabel("Position X [mm]")
    ax_x.set_title("Motion Estimation Performance")
    ax_x.grid(True, alpha=0.5)
    ax_x.legend(loc="upper left")

if PLOT_DIM == 1 and PLOT_DIRECTION == "y" or PLOT_DIM == 2:
    # Y position 
    ax_y.plot(time_s, mean_y, label="Estimated Y motion", linewidth=1.8)
    #ax_y.scatter(time_s, mean_y, label="Estimated Y motion")
    #ax_y.plot(time_s, ref_y_plot, linestyle="--", label="Reference Y motion")
    ax_y.plot(t_fine, ref_y_fine_plot, color="green", linewidth=1.2, label="Interpolated Ref Y motion")
    ax_y.set_ylabel("Position Y [mm]")
    ax_y.grid(True, alpha=0.5)
    ax_y.legend(loc="upper left")

# Errors 
if len(angles) > 0:
    prev_angle = angles[0]
    for i in range(1, len(angles)):
        if angles[i] != prev_angle:
            idx = angle_idx[i]
            if 0 <= idx < len(time_s):
                t0 = time_s[idx]
                t1 = t0 + EXCLUDE_AFTER_ANGLE_CHANGE_S
                if PLOT_DIM == 1 and PLOT_DIRECTION == "x" or PLOT_DIM == 2:
                    ax_ex.axvspan(t0, t1, color="gray", alpha=0.25, zorder=0)
                if PLOT_DIM == 1 and PLOT_DIRECTION == "y" or PLOT_DIM == 2:
                    ax_ey.axvspan(t0, t1, color="gray", alpha=0.25, zorder=0)
            prev_angle = angles[i]

# Interpolated errors
err_x_fine = mean_x_fine - ref_x_fine
err_y_fine = mean_y_fine - ref_y_fine

# Interpolated errors
err_x_fine_plot = np.full_like(err_x_fine, np.nan)
err_y_fine_plot = np.full_like(err_y_fine, np.nan)
err_x_fine_plot[valid_eval_mask_fine] = err_x_fine[valid_eval_mask_fine]
err_y_fine_plot[valid_eval_mask_fine] = err_y_fine[valid_eval_mask_fine]



if PLOT_DIM == 1 and PLOT_DIRECTION == "x" or PLOT_DIM == 2:
    if np.any(valid_eval_mask_fine):
        ax_ex.plot(
            t_fine, err_x_fine_plot,
            linestyle="-", color="red",linewidth=0.5,
            label=f"x error (RMSE={np.sqrt(np.mean(dev_x_fine**2)):.2f} mm)"
        )

    ax_ex.set_ylabel("Error X [mm]")
    ax_ex.grid(True, alpha=0.5)
    ax_ex.set_ylim(-2, 2)
    #ax_ex.set_xlim(30, 47)
    ax_ex.axhline(0, linestyle="--", linewidth=1)

    ax_ex.legend(loc="upper left")


    ax_ex.axhline(y=1, color="coral", linestyle='-', linewidth=0.8)
    ax_ex.axhline(y=-1, color="coral", linestyle='-', linewidth=0.8)
    ax_ex.text(1.0, 1, '+1 mm', transform=ax_ex.get_yaxis_transform(), va='bottom', ha='right', fontsize=7)
    ax_ex.text(1.0, -1, '-1 mm', transform=ax_ex.get_yaxis_transform(), va='top', ha='right', fontsize=7)

    ax_x.text(
        0.99, 1.02,
        "Red lines / labels = gantry angle",
        transform=ax_x.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        color="red"
    )

if PLOT_DIM == 1 and PLOT_DIRECTION == "y" or PLOT_DIM == 2:
    if np.any(valid_eval_mask_fine):
        ax_ey.plot(
            t_fine, err_y_fine_plot,
            linestyle="-", color="red", linewidth=0.5,
            label=f"y error (RMSE={np.sqrt(np.mean(dev_y_fine**2)):.2f} mm)"
        )


    ax_ey.set_ylabel("Error Y [mm]")
    ax_ey.set_xlabel("Time [s]")

    ax_ey.grid(True, alpha=0.5)

    ax_ey.set_ylim(-2, 2)
    #ax_ey.set_xlim(30, 47)
    ax_ey.legend(loc="upper left")

    ax_ey.axhline(0, linestyle="--", linewidth=1)

    ax_ey.axhline(y=1, color="coral", linestyle='-', linewidth=0.8)
    ax_ey.axhline(y=-1, color="coral", linestyle='-', linewidth=0.8)
    ax_ey.text(1.0, 1, '+1 mm', transform=ax_ey.get_yaxis_transform(), va='bottom', ha='right', fontsize=7)
    ax_ey.text(1.0, -1, '-1 mm', transform=ax_ey.get_yaxis_transform(), va='top', ha='right', fontsize=7)





# Angle markers 
if len(angles) and len(samples) > 0:
    first_angle_pos = 0
    while first_angle_pos < len(angle_idx) and angle_idx[first_angle_pos] < 0:
        first_angle_pos += 1

    if first_angle_pos < len(angles):
        prev_angle = angles[first_angle_pos]
        start_idx = angle_idx[first_angle_pos]

        for i in range(first_angle_pos + 1, len(angles)):
            angle = angles[i]
            idx = angle_idx[i]
            if angle != prev_angle:
                if PLOT_DIM == 1 and PLOT_DIRECTION == "x" or PLOT_DIM == 2:
                    for ax in [ax_x, ax_ex]:
                        ax.axvline(x=time_s[start_idx], color="red",
                                linestyle=":", alpha=0.7)

                if PLOT_DIM == 1 and PLOT_DIRECTION == "y" or PLOT_DIM == 2:
                    for ax in [ax_y, ax_ey]:
                        ax.axvline(x=time_s[start_idx], color="red",
                                linestyle=":", alpha=0.7)
                mid_x = (time_s[start_idx] + time_s[idx - 1]) / 2
                ax_x.text(mid_x, ax_x.get_ylim()[1]*0.95, f"{prev_angle}°",
                          color="red", ha="center", va="top")
                prev_angle = angle
                start_idx = idx
        if PLOT_DIM == 1 and PLOT_DIRECTION == "x" or PLOT_DIM == 2:
            for ax in [ax_x, ax_ex]:
                ax.axvline(x=time_s[start_idx], color="red",
                        linestyle=":", alpha=0.7)
        if PLOT_DIM == 1 and PLOT_DIRECTION == "y" or PLOT_DIM == 2:
            for ax in [ax_y, ax_ey]:
                ax.axvline(x=time_s[start_idx], color="red",
                        linestyle=":", alpha=0.7)


report.print()


# Layout & save 
fig.suptitle(f"Motion Estimation vs Reference — {os.path.basename(log_file)}",y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.97])
if SAVE_IMG:
    plt.savefig(f"{base_dir}/motion_estimation_analysis_log_file.png",dpi=300)
plt.show()

