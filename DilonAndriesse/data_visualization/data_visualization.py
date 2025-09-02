import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import mne
from mne.datasets.sleep_physionet.age import fetch_data

# load in data files
psg_file = "C:/Users/andri/OneDrive - HAN/Desktop/Internship Donders/VSC/data/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf"
anno_file = "C:/Users/andri/OneDrive - HAN/Desktop/Internship Donders/VSC/data/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf"

annotation_desc_2_event_id = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}

event_id = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3/4": 4,
    "Sleep stage R": 5,
}

def read_raw_psg(file):
    file = mne.io.read_raw_edf(
        psg_file,
        # Tries to infer channel types from names
        infer_types=True,
        # loads data into memory for data manipulation and indexing
        preload=True,
    )

    return file


def annotate_psg(anno_file, raw_data):
    # extract annotation from file
    anno = mne.read_annotations(anno_file)
    # annotate raw data
    raw_data.set_annotations(anno, emit_warning=False)

    return anno, raw_data

def plot_data(anno_data):
    anno_data.plot(
        start=60,
        # epoch size
        duration=10,
        # stops the program from closing if plot is still opened
        block=True,
        scalings=dict(
            eeg=1e-4, 
            resp=1e3, 
            eog=1e-4, 
            emg=1e-6, 
            misc=1e-1
        ),
    )


def crop_data(anno, raw_data):
    # crop the annotations so only half an hour before and after
    # being awake is annotated
    anno.crop(anno[1]["onset"] - 30 * 60, anno[-2]["onset"] + 30 * 60)
    # annotate the data with the changed annotations
    raw_data.set_annotations(anno, emit_warning=False)

    sleep_events, _ = mne.events_from_annotations(
        raw_data,event_id=annotation_desc_2_event_id, chunk_duration=10
    )
    print(sleep_events)

    return sleep_events, event_id, raw_data


def plot_events(sleep_events, event_id, raw_data):
    fig = mne.viz.plot_events(
        sleep_events,
        event_id=event_id,
        sfreq=raw_data.info["sfreq"],
        first_samp=sleep_events[0, 0]
    )


if __name__ == '__main__':
    raw_data = read_raw_psg(psg_file)
    anno, anno_data = annotate_psg(anno_file, raw_data)
    plot_data(anno_data)

    sleep_events, event_id, raw_data = crop_data(anno, raw_data)
    plot_events(sleep_events, event_id, raw_data)


    print(raw_data.info)
    print(raw_data.ch_names)
    print(raw_data.get_channel_types())