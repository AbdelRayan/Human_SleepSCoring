import mne
import numpy as np

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

file.plot(duration=10, scalings=scalings, show_scrollbars=True)
