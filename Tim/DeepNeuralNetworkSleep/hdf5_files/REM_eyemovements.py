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
rem_raw.plot(scalings=dict(eog=1e-4), duration=30, title='REM-only EOG channels (EOG1, EOG2, EOG1-EOG2)', block=True)