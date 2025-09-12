import numpy as np
import mne
import scipy
import os


def extract_raw(file, channel):
    # read raw edf file
    raw = mne.io.read_raw_edf(
        file,
        # Tries to infer channel types from names
        infer_types=True,
        # loads data into memory for data manipulation and indexing
        preload=True,
        # exlude the channel not used for features selection
        include=channel
    )

    return raw


def extract_anno(file, raw):
    """
    """
    anno = mne.read_annotations(file)

    raw.set_annotations(anno, emit_warning=False)

    return anno


def crop_data(data, anno):
    """
    """
    cropped_data = data.crop(
        anno[1]["onset"] - 30 * 60, 
        anno[-2]["onset"] + 30 * 60
        )
    
    return cropped_data


def calc_emg_mean(data, window_len):
    emg_mean_list = []
    # determine start of each epoch
    for start in range(0, len(data) - window_len + 1, window_len):
        # collects data from array in epoch length
        window = data[start:min(start + window_len, len(data))]
        emg_mean = np.mean(window)
        emg_mean_list.append(emg_mean)

    return emg_mean_list


def get_sleep_states(cropped_raw, epoch):
    """
    """
    raw_array = []

    annotation_desc_2_event_id = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }

    sleep_events, _ = mne.events_from_annotations(
        cropped_raw, 
        event_id=annotation_desc_2_event_id, 
        chunk_duration=epoch
    )

    # create array from sleep events with only sleep stages
    for event in sleep_events:
        raw_array.append(int(event[-1]))

    return raw_array



if __name__ == '__main__':
    raw_path = "C:/Users/andri/school/bio-informatics/internship/donders/data/human_test_data/input/raw_input"
    anno_path = "C:/Users/andri/school/bio-informatics/internship/donders/data/human_test_data/input/annotation_raw"
    output_path = "C:/Users/andri/school/bio-informatics/internship/donders/data/human_test_data/mat_files/"
    #raw_file = "C:/Users/andri/school/bio-informatics/internship/donders/data/human_test_data/input/raw_input/SC4092E0-PSG.edf"
    #anno_file = "C:/Users/andri/school/bio-informatics/internship/donders/data/human_test_data/input/annotation_raw/SC4092EC-Hypnogram.edf"
    epoch = 5

    # mat = scipy.io.loadmat('C:/Users/andri/school/bio-informatics/internship/donders/data/rodent_test_data/HPC_100_CH8_0.continuous.mat')
    # print(mat)
    # mat = scipy.io.loadmat('C:/Users/andri/school/bio-informatics/internship/donders/data/human_test_data/mat_files/SC4001E0_Fpz-Cz.mat')
    # print(mat)

    # print(np.ones(5)/5)

    for raw_file, anno_file in zip(os.listdir(raw_path), os.listdir(anno_path)):
        get_annotation = False
        subject = raw_file.split("/")[-1].split("-")[0]
        for channel in ["Fpz-Cz", "Pz-Oz", "horizontal", "submental"]:
            # extract raw data
            raw = extract_raw(f"{raw_path}/{raw_file}", channel)
            anno = extract_anno(f"{anno_path}/{anno_file}", raw)
            # cropped_raw = crop_data(raw, anno)
            # cropped_anno = crop_data(anno, anno)
            while get_annotation == False:
                sleep_states = get_sleep_states(raw, epoch)
                scipy.io.savemat(
                    f"{output_path}{subject}_sleep_states.mat", 
                    {"States": sleep_states}
                )
                get_annotation = True
            # rename obscure channel names
            if channel == "horizontal":
                raw.rename_channels({"horizontal": "EOG"})
                channel = "EOG"
            elif channel == "submental":
                raw.rename_channels({"submental":"EMG"})
                channel = "EMG"
            # save numpy array to .mat file format
            scipy.io.savemat(
                f"{output_path}{subject}_{channel}.mat", 
                {channel: raw._data.reshape((-1,1))}
                )