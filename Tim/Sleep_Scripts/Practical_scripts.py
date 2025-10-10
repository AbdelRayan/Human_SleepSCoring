"""
Several helper functions for converting and altering eeg data
By: Tim Veldema
Date: 16/09/25
"""

import os
import mne

ch_types = {
    'TBAR': 'eeg',
    'TBPR': 'eeg',
    'TLR': 'eeg',
    'TBAL': 'eeg',
    'TBPL': 'eeg',
    'TLL': 'eeg',
    'TR': 'eeg',
    'TL':'eeg',
    'EKG':'ecg',
    'T5':'eeg',
    'T6':'eeg',
    'C3':'eeg',
    'C4':'eeg',
    'Cz':'eeg',
    'Oz':'eeg',
    'EOG1':'eog',
    'EOG2':'eog',
    'EMG1':'emg',
    'EMG2':'emg',
    'EOG1-EOG2':'eog',
    'EMG1-EMG2':'emg'
}


def get_path(file):
    """
    Returns the directory path and file name separately.

    :param file: full path to file - str
    :return: full path to directory, file name - str, str
    """
    path = os.path.dirname(file)
    filename = os.path.basename(file)
    return path, filename

def convert_binary_brainvision(file):
    """
    Converts binary BrainVision .dat files to EDF, and places the resulting file in a folder named 'edf_files'
    in the same directory as the original .vhdr file.
    """
    raw = mne.io.read_raw_brainvision(file)

    # Get directory + filename
    path = os.path.dirname(file)
    file_vhdr = os.path.basename(file)

    # Construct EDF filename
    edf = file_vhdr.replace('.vhdr', '.edf')
    output_dir = os.path.join(path, 'edf_files')

    # Create folder if missing
    os.makedirs(output_dir, exist_ok=True)

    output = os.path.join(output_dir, edf)
    print(f"Saving EDF to: {output}")

    raw.export(output, fmt='edf')

def convert_channel_types(raw):
    """
    Converts channels to their correct types
    :param raw: raw mne file
    :return: raw mne file with corrected channel types
    """
    ch_names = raw.ch_names
    ch_dict = {}
    for entry in ch_names:
        ch_dict[entry] = ch_types[entry]
    raw.set_channel_types(ch_dict)
    return raw

def link_sections(directory, source):
    """
    Links fragmented night segments together to form full nights
    :param directory: path to directory - str
    :param source: file origin (brainvision or edf)
    """
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
        print(f"Linking {key}")
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
        output = os.path.join(directory, 'combined_nights', key + extension)
        raw.export(output, fmt=source.lower())

if __name__ == '__main__':
    file = "D:/converted_sleep_data/2"
    link_sections(file, 'brainvision')