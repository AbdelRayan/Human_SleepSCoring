import os

import mne

ch_types = {
    'TBAR1': 'eeg',
    'TBAR2': 'eeg',
    'TBAR3': 'eeg',
    'TBAR4': 'eeg',
    'TBPR1': 'eeg',
    'TBPR2': 'eeg',
    'TBPR3': 'eeg',
    'TBPR4': 'eeg',
    'TLR01': 'eeg',
    'TLR02': 'eeg',
    'TLR03': 'eeg',
    'TLR04': 'eeg',
    'TLR05': 'eeg',
    'TLR06': 'eeg',
    'TBAL1': 'eeg',
    'TBAL2': 'eeg',
    'TBAL3': 'eeg',
    'TBAL4': 'eeg',
    'TBPL1': 'eeg',
    'TBPL2': 'eeg',
    'TBPL3': 'eeg',
    'TBPL4': 'eeg',
    'TLL01': 'eeg',
    'TLL02': 'eeg',
    'TLL03': 'eeg',
    'TLL04': 'eeg',
    'TLL05': 'eeg',
    'TLL06': 'eeg',
    'TR01': 'eeg',
    'TR02': 'eeg',
    'TR03': 'eeg',
    'TR04': 'eeg',
    'TR05': 'eeg',
    'TR06': 'eeg',
    'TR07': 'eeg',
    'TR08': 'eeg',
    'TR09': 'eeg',
    'TR10': 'eeg',
    'TL01':'eeg',
    'TL02':'eeg',
    'TL03':'eeg',
    'TL04':'eeg',
    'TL05':'eeg',
    'TL06':'eeg',
    'TL07':'eeg',
    'TL08':'eeg',
    'TL09':'eeg',
    'TL10':'eeg',
    'EKG1':'ecg',
    'EKG2':'ecg',
    'T5':'eeg',
    'T6':'eeg',
    'C3':'eeg',
    'C4':'eeg',
    'Cz':'eeg',
    'Oz':'eeg',
    'EOG1':'eog',
    'EOG2':'eog',
    'EMG1':'emg',
    'EMG2':'emg'
}

def convert_binary_brainvision(file):
    raw = mne.io.read_raw_brainvision(file)
    path_list = file.split('/')
    path = os.path.join(*path_list[:-1])
    file_vhdr = path_list[-1]
    edf = file_vhdr.replace('vhdr', 'edf')
    if not os.path.isdir(os.path.join(path, 'edf_files')):
        os.mkdir(os.path.join(path, 'edf_files'))
    output = os.path.join(path, 'edf_files', edf)
    raw.export(output, fmt='edf')

if __name__ == '__main__':
    # file = "D:/converted_sleep_data/2/2_paradigm.vhdr"
    # convert_binary_brainvision(file)