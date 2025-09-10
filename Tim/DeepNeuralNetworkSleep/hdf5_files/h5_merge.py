import h5py
import os

rat_dir = "C:/Users/timmi/Documents/rath5"
out_file = "C:/Users/timmi/Documents/merged/merged.h5"

with h5py.File(out_file, "w") as h5out:
    grp = h5out.create_group("OS_Basic")
    for count, file in enumerate(os.listdir(rat_dir)):
        with h5py.File(os.path.join(rat_dir, file), "r") as h5in:
            h5in.copy("/", grp, name=f"Rat{count+1}")

