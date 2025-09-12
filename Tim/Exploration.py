import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_multitaper

file = mne.io.read_raw_edf("SC4001E0-PSG.edf")
annotations = mne.read_annotations("SC4001EC-Hypnogram.edf")
print(file.ch_names)
print(file.get_channel_types())
file.set_annotations(annotations)
file.set_channel_types({
    'EEG Fpz-Cz': 'eeg',
    'EEG Pz-Oz': 'eeg',
    'EOG horizontal': 'eog',
    'Resp oro-nasal': 'misc',
    'EMG submental': 'emg',
    'Temp rectal': 'misc',
    'Event marker': 'stim'
})
mapping = {
    'EEG Fpz-Cz': 'Fpz-Cz',
    'EEG Pz-Oz': 'Pz-Oz',
    'EOG horizontal': 'EOG',
    'Resp oro-nasal': 'Nasal',
    'EMG submental': 'EMG',
    'Temp rectal': 'Rectal',
    'Event marker': 'Marker'
}


file.rename_channels(mapping)
file.pick_types(eeg=True, eog=True)

scalings = {
    'eeg': 1e-4,
    'eog': 1e-4,
    'emg': 1e-3,
    'misc': 0.5,
    'stim': 100
}



# file.plot(duration=800, scalings=scalings, show_scrollbars=True, block=True)

annotations = file.annotations

stage_mapping = {
    "Sleep stage W": 0,
    "Sleep stage R": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage ?": -1
}

onsets = annotations.onset
durations = annotations.duration
stages = [stage_mapping[d] for d in annotations.description]

sfreq = file.info['sfreq']
end_time = file.times[-1]
times = np.arange(0, end_time, 1/sfreq)

stage_vector = np.zeros_like(times)

for onset, duration, stage in zip(onsets, durations, stages):
    if stage == -1:
        continue
    idx = np.where((times >= onset) & (times < onset + duration))
    stage_vector[idx] = stage


trim_seconds = 20000

start_idx = np.searchsorted(times, trim_seconds)
end_idx = np.searchsorted(times, times[-1] - trim_seconds)

trimmed_stage_vector = stage_vector[start_idx:end_idx]
trimmed_times = times[start_idx:end_idx]

plt.figure(figsize=(15, 3))
plt.plot(trimmed_times, trimmed_stage_vector, drawstyle='steps-post')
plt.yticks([0, 1, 2, 3, 4], ["Wake", "REM", "N1", "N2", "N3"])
plt.gca().invert_yaxis()
plt.xlabel("Time (seconds)")
plt.ylabel("Sleep Stage")
plt.title("Trimmed Hypnogram")
plt.show()
