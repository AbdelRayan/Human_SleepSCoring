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

file.pick_types(eeg=True)
# Create epochs (30s windows)
epochs = mne.make_fixed_length_epochs(file, duration=30, preload=True)

# Compute average across epochs
evoked = epochs.average()

power = tfr_multitaper(epochs, freqs=np.arange(1, 40, 2),
                       n_cycles=2, return_itc=False)

# Plot power over time for a single channel
power.plot(picks='EEG Fpz-Cz', baseline=(None, 0), mode='logratio')

# Topographic plot at specific frequency
power.plot_topomap(fmin=12, fmax=15, tmin=0, tmax=2, ch_type='eeg')
# scalings = {
#     'eeg': 1e-4,
#     'eog': 1e-4,
#     'emg': 1e-3,
#     'misc': 0.5,
#     'stim': 100
# }
#

# file.plot(duration=10, scalings=scalings, show_scrollbars=True)

# annotations = file.annotations
#
# stage_mapping = {
#     "Sleep stage W": 0,
#     "Sleep stage 1": 1,
#     "Sleep stage 2": 2,
#     "Sleep stage 3": 3,
#     "Sleep stage 4": 3,
#     "Sleep stage R": 4,
#     "Sleep stage ?": 5
# }
#
# onsets = annotations.onset
# durations = annotations.duration
# stages = [stage_mapping[d] for d in annotations.description]
#
# sfreq = file.info['sfreq']
# end_time = file.times[-1]
# times = np.arange(0, end_time, 1/sfreq)
#
# stage_vector = np.zeros_like(times)
#
# for onset, duration, stage in zip(onsets, durations, stages):
#     idx = np.where((times >= onset) & (times < onset + duration))
#     stage_vector[idx] = stage


# plt.figure(figsize=(15, 3))
# plt.plot(times, stage_vector, drawstyle='steps-post')
# plt.yticks([0,1,2,3,4], ["W","N1","N2","N3","R"])
# plt.xlabel("Time (seconds)")
# plt.ylabel("Sleep Stage")
# plt.title("Hypnogram")
# plt.show()

# channels = mne.pick_types(file.info, eeg=True, eog=True)

file.plot(
    duration=30,
    show=True,
    block=True
)


