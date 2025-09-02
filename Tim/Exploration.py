import mne
import numpy as np
import matplotlib.pyplot as plt

file = mne.io.read_raw_edf("SC4001E0-PSG.edf")
annotations = mne.read_annotations("SC4001EC-Hypnogram.edf")
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

scalings = {
    'eeg': 200e-6,
    'eog': 300e-6,
    'emg': 1e-3,
    'misc': 0.5,
    'stim': 100
}
# print(file.get_channel_types())

# file.plot(duration=10, scalings=scalings, show_scrollbars=True)

annotations = file.annotations

stage_mapping = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5
}

onsets = annotations.onset
durations = annotations.duration
stages = [stage_mapping[d] for d in annotations.description]

sfreq = file.info['sfreq']
end_time = file.times[-1]
times = np.arange(0, end_time, 1/sfreq)

stage_vector = np.zeros_like(times)

for onset, duration, stage in zip(onsets, durations, stages):
    idx = np.where((times >= onset) & (times < onset + duration))
    stage_vector[idx] = stage


# plt.figure(figsize=(15, 3))
# plt.plot(times, stage_vector, drawstyle='steps-post')
# plt.yticks([0,1,2,3,4], ["W","N1","N2","N3","R"])
# plt.xlabel("Time (seconds)")
# plt.ylabel("Sleep Stage")
# plt.title("Hypnogram")
# plt.show()

channels = mne.pick_types(file.info, eeg=True, eog=True)

file.plot(
    duration=60,
    scalings=scalings,
    n_channels = len(channels),
    picks=channels,
    show=True,
    block=True
)
