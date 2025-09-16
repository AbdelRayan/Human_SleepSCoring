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

def get_path(file):
    path_list = file.split('/')
    path = os.path.join(*path_list[:-1])
    file_heh = path_list[-1]
    return path, file_heh

def convert_binary_brainvision(file):
    raw = mne.io.read_raw_brainvision(file)
    path, file_vhdr = get_path(file)
    edf = file_vhdr.replace('vhdr', 'edf')
    if not os.path.isdir(os.path.join(path, 'edf_files')):
        os.mkdir(os.path.join(path, 'edf_files'))
    output = os.path.join(path, 'edf_files', edf)
    raw.export(output, fmt='edf')

def convert_channel_types(raw):
    ch_names = raw.ch_names
    ch_dict = {}
    for entry in ch_names:
        ch_dict[entry] = ch_types[entry]
    raw.set_channel_types(ch_dict)
    return raw

def link_sections(directory, source):
    nights = []
    for data in os.listdir(directory):
        if 'night' in data and 'vhdr' in data:
            nights.append(data)
    night_dict = {}
    for night in nights:
        subject, night, number = night.split("_")
        if night not in night_dict:
            night_dict[night] = [os.path.join(directory, subject + '_' + night + '_' + number)]
        else:
            night_dict[night].append(os.path.join(directory, subject + '_' + night + '_' + number))
    for key in night_dict:
        extension = ""
        files = night_dict[key]
        if source.upper() == 'BRAINVISION':
            raw_list = [mne.io.read_raw_brainvision(f, preload=True) for f in files]
            extension = '.dat'
        elif source.upper() == 'EDF':
            raw_list = [mne.io.read_raw_edf(file, preload=True) for f in files]
            extension = '.edf'
        else:
            print("invalid source selected, choose either BRAINVISION or EDF")
            return
        raw = mne.concatenate_raws(raw_list)
        if not os.path.isdir(os.path.join(directory, 'combined_nights')):
            os.mkdir(os.path.join(directory, 'combined_nights'))
        output = os.path.join(directory, 'combined_nights', key+extension)
        raw.export(output, fmt=source.lower())





if __name__ == '__main__':
    file = "D:/converted_sleep_data/2"
    link_sections(file, 'brainvision')