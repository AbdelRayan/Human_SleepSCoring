import os
import Sleep_Scripts.ASCII_to_binary_2ch_downsampled as ASCII

base_dir = "D:/Intercranial_sleep_data/"
binary_path = "D:/converted_sleep_data"
subjects = os.listdir(base_dir)

for subject in subjects:
    files = os.listdir(os.path.join(base_dir, subject, 'iEEG'))
    vhdr_files = [a for a in files if 'vhdr' in a]
    binary_file = os.path.join(binary_path, subject)
    for vhdr_file in vhdr_files:
        print(f'Converting {subject}-{vhdr_file}')
        ASCII.convert_brainvision_ascii(vhdr_file, binary_file, ['EOG1', 'Cz'])



