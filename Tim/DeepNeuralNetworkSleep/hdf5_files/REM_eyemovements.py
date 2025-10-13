import mne
import numpy as np
from matplotlib import pyplot as plt

fs = 250
window_length = 10 * fs
epoch_length = int(window_length / fs)
samples_per_epoch = int(fs * window_length)

files = [
    "C:/EEG_Data_stage/2/iEEG/converted_uni_and_bi/2_night1_01.vhdr",
    "C:/EEG_Data_stage/2/iEEG/converted_uni_and_bi/2_night1_02.vhdr",
    "C:/EEG_Data_stage/2/iEEG/converted_uni_and_bi/2_night1_03.vhdr"
]

raw_list = [mne.io.read_raw_brainvision(f, preload=True) for f in files]
raw = mne.concatenate_raws(raw_list)

scores_files = [
    "C:/EEG_Data_stage/2/iEEG/edf/2_night1_01_hypnogram.npy",
    "C:/EEG_Data_stage/2/iEEG/edf/2_night1_02_hypnogram.npy",
    "C:/EEG_Data_stage/2/iEEG/edf/2_night1_03_hypnogram.npy"
]

score_list = [np.load(f) for f in scores_files]
states = np.concatenate(score_list)

# === KEEP EOG CHANNELS ===
picks = ['EOG1', 'EOG2', 'EOG1-EOG2']
raw.pick_channels(picks)

# === FIND REM EPOCHS ===
rem_indices = np.where(states == 4)[0]
print(f"Found {len(rem_indices)} REM epochs")

if len(rem_indices) == 0:
    raise ValueError("No REM epochs found in hypnogram.")

# === EXTRACT AND CONCATENATE REM SEGMENTS ===
rem_segments = []
for idx in rem_indices:
    start = idx * epoch_length
    stop = (idx + 1) * epoch_length

    if start >= raw.times[-1]:  # outside data range
        continue
    stop = min(stop, raw.times[-1])

    # Crop segment and store
    rem_seg = raw.copy().crop(tmin=start, tmax=stop)
    rem_segments.append(rem_seg)

if not rem_segments:
    raise ValueError("No valid REM segments within data duration.")

# Concatenate all REM segments together
rem_raw = mne.concatenate_raws(rem_segments, preload=True)
rem_raw.set_channel_types({
    'EOG1-EOG2':'eog',
    'EOG1':'eog',
    'EOG2':'eog'
})

print(rem_raw)
# rem_raw.plot(scalings=dict(eog=1e-4), duration=30, title='REM-only EOG channels (EOG1, EOG2, EOG1-EOG2)', block=True)

tmin = 60
tmax = tmin + 30
sfreq = rem_raw.info['sfreq']
data, times = rem_raw[:, int(tmin * sfreq):int(tmax * sfreq)]

# Shift time to start at 0
times = times - tmin

# Scale to µV
data = data * 1e6

plt.figure(figsize=(12, 6))

# Vertical spacing between channels
offset = 100
yticks = []
scale = 0.5
data_scaled = data * scale  # data is in µV
for idx, ch_name in enumerate(raw.info['ch_names']):
    ch_offset = idx * offset
    plt.plot(
        times,
        data_scaled[idx] + ch_offset,
        linewidth=0.4,
        color="black"
    )
    yticks.append(ch_offset)

# Label y-axis with channel names
plt.yticks(yticks, rem_raw.info['ch_names'])
plt.xlabel("Time (s)")
plt.ylabel("Channels")
plt.title(f"Rem eye movements 60-90s")

# --- Add scale bars (50 µV) ---
scalebar_height = 50 * scale  # µV
scalebar_x = 1  # 1 s into the fragment

for ch_offset in yticks:  # one bar per channel
    plt.plot([scalebar_x, scalebar_x],
             [ch_offset, ch_offset + scalebar_height],
             color="red", linewidth=2)
    plt.text(scalebar_x + 0.2,
             ch_offset + scalebar_height / 2,
             f"{scalebar_height} µV",
             va="center", ha="left", color="red")

# Top channel
scalebar_y_top = yticks[-1]
plt.plot([scalebar_x, scalebar_x], [scalebar_y_top, scalebar_y_top + scalebar_height],
         color="red", linewidth=2)
plt.text(scalebar_x + 0.2, scalebar_y_top + scalebar_height / 2,
         f"{scalebar_height} µV", va="center", ha="left", color="red")

plt.xlim(0, tmax - tmin)  # set x-axis from 0 to 30
plt.tight_layout()
plt.savefig("Rem_eye_movements_60-90.pdf", dpi=300, format="pdf")
plt.show(block=True)
plt.close()