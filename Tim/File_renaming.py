import os
import shutil


file_dir = "C:/Users/timmi/Documents/rat"
new_folder = "C:/Users/timmi/Documents/Rat1"

for directory in os.listdir(file_dir):
    trial = "".join(directory.split('_')[0:2])
    file_base = "Rat1" + "_" + "CN1" + "_" + trial
    for file in os.listdir(os.path.join(file_dir, directory)):
        file_name = ""
        if "HPC" in file:
            file_name = file_base + "_" + "HPC"
        if "PFC" in file:
            file_name = file_base + "_" + "PFC"
        if "states" in file:
            file_name = file_base + "_" + "states"
        shutil.copy(os.path.join(file_dir, directory, file), os.path.join(new_folder))
        shutil.move(os.path.join(new_folder, file), os.path.join(new_folder, file_name))
