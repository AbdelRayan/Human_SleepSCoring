import yasa
import mne
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import numpy as np

def plot_data(raw, image_path):
    """
    Plot the raw signal data to an interactive plot with colored 
    annotations sections.

    Parameters:
    cropped_raw (obj): Object containing the raw measurement data.
    """

    tmin, tmax = 5870, 5900  # 30-second fragment
    sfreq = raw.info['sfreq']
    data, times = raw[:, int(tmin * sfreq):int(tmax * sfreq)]

    # Shift time to start at 0
    times = times - tmin

    # Scale to µV
    data[0] = data[0] * 1e6 ## fpz-cz
    data[1] = data[1] * 1e6 ##pz-oz
    data[2] = data[2] * 1e6 ##horizontal
    data[3] = data[3] * 1e8##EMG?????????????????/ 8e7


    plt.figure(figsize=(12, 6))

    # Vertical spacing between channels
    offset = 100
    yticks = []
    scale = 0.5
    data_scaled = data * scale  # data is in µV
    for idx, ch_name in enumerate(raw.info['ch_names']):
        ch_offset = idx * offset
        plt.plot(
            times,
            data_scaled[idx] + ch_offset,
            linewidth=0.4,
            color="black"
        )
        yticks.append(ch_offset)

    # Label y-axis with channel names
    plt.yticks(yticks, raw.info['ch_names'])
    plt.xlabel("Time (s)")
    plt.ylabel("Channels")
    plt.title(f"N2 30s - fragment: 5870-5900 (s)")

    # --- Add scale bars (50 µV) ---
    scalebar_height = 50 * scale # µV
    scalebar_x = 0  # 1 s into the fragment

    ch0_baseline = yticks[0]  # baseline offset for first channel
    scalebar_y_bottom = ch0_baseline

    # bottom channel
    plt.plot([scalebar_x, scalebar_x],
            [scalebar_y_bottom, scalebar_y_bottom + scalebar_height],
            color="red", linewidth=6)
    plt.text(scalebar_x - 0.2,
            scalebar_y_bottom + scalebar_height,
            f"{scalebar_height} µV", va="center", ha="right", color="red", fontsize=12)

    # top channel
    scalebar_y_top = yticks[-1]
    plt.plot([scalebar_x, scalebar_x], [scalebar_y_top, (scalebar_y_top + scalebar_height)],
            color="red", linewidth=6)
    plt.text(scalebar_x - 0.2, scalebar_y_top + scalebar_height,
            f"{0.25} µV", va="center", ha="right", color="red", fontsize=12)

    plt.xlim(0, tmax - tmin)  # set x-axis from 0 to 30
    plt.tight_layout()
    # plt.savefig("eeg_fragment_0_30s.svg", dpi=300, format="svg")
    plt.savefig(os.path.join(image_path, "N2_PSG.svg"), format="svg")
    plt.show(block=True)
    plt.close()

    # cropped_raw = raw.copy().crop(tmin=37400, tmax=37430)
    # #data, times = cropped_raw[:, :]
    # # Get data in µV instead of V
    # data, times = cropped_raw.get_data(return_times=True)
    # data = data * 1e6

    # # Vertical offset for stacking
    # offset = np.ptp(data) * 2
    # plt.figure(figsize=(12, 8))

    # for i in range(data.shape[0]):
    #     plt.plot(times, data[i] + i * offset, label=raw.info['ch_names'][i])

    # plt.xlabel("Time (s)")
    # plt.ylabel("Channels")
    # plt.title("EEG section (100–110s, all channels)")
    # plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    # plt.show()
    # plt.savefig(os.path.join(image_path, "wake_eeg.svg"), format="svg")

    # raw.plot(
    #     start=0,
    #     # window size
    #     duration=300,
    #     # custom scalings
    #     scalings=dict(eeg=1e-4, eog=5e-4, emg=1e-6)
    # )


def plot_events(sleep_events, cropped_raw, image_path):
    """
    Creates a plot to give an overview of the distribution of 
    sleep stages.

    Parameters:
    sleep_events (2d-list): A nested list with epoch index, 
    and sleep stage id.
    cropped_raw (obj): Object containing the raw measurement data.
    """
    # dictionary with the proper sleep stages and ids
    event_id = {
        "Sleep stage W": 0,
        "Sleep stage N1": 1,
        "Sleep stage N2": 2,
        "Sleep stage N3": 3,
        "Sleep stage R": 4,
    }

    mne.viz.plot_events(
        sleep_events,
        event_id=event_id,
        sfreq=cropped_raw.info["sfreq"],
        first_samp=sleep_events[0, 0]
    )

    plt.savefig(os.path.join(image_path, "plot_events.svg"), format="svg")


def hypnogram_vis(cropped_raw, sleep_events, image_path):
    """
    Plot a hypnogram and corresponding spectrogram.

    Parameters:
    cropped_raw (obj): Object containing the raw measurement data.
    sleep_events (2d-list): A nested list with epoch index, 
    and sleep stage id.

    Returns:
    hypno_up (list): List with the upscaled annotation
    """
    raw_array = []
    # remove non-EEG channels
    cropped_raw.drop_channels(["horizontal", "submental"])
    # units to use
    data = cropped_raw.get_data(units="uV")
    # sampling frequency
    sf = cropped_raw.info["sfreq"]
    # channel names
    chan = cropped_raw.ch_names

    # create array from sleep events with only sleep stages
    for event in sleep_events:
        raw_array.append(int(event[-1]))

    # plot hypnogram
    fig, ax = plt.subplots(figsize=(20, 5))
    yasa.plot_hypnogram(raw_array, sf_hypno=float(1/10), ax=ax)
    plt.savefig(os.path.join(image_path, "hypnogram.svg"), format="svg")

    # upsample the annotation to the data
    hypno_up = yasa.hypno_upsample_to_data(
        raw_array, 
        sf_hypno=float(1/10), 
        data=cropped_raw
        )
    # plot spectogram
    yasa.plot_spectrogram(data[chan.index("Fpz-Cz")], sf, hypno_up)
    #plt.savefig(os.path.join(image_path, "hypnogram_spectogram.svg"), format="svg")
    plt.savefig(os.path.join(image_path, "hypnogram_spectogram.png"), format="png")

    return hypno_up


def calc_bandpower(cropped_raw, hypno_up, bp_output, image_path):
    """
    Calculate and plot the band powers to the two EEG channels.

    Parameters:
    cropped_raw (obj): Object containing the raw measurement data.
    hypno_up (list): List with the upscaled annotation
    """
    bandpower = yasa.bandpower(
        cropped_raw, 
        hypno=hypno_up, 
        include=(0, 1, 2, 3, 4),
        bands=[
            (0, 0.5, "Noise"),
            (0.5, 5, "Delta"),
            (6, 10, "Theta"),
            (11, 17, "Sigma"),
            #(22, 30, "Beta"),
            (35, 45, "Gamma"),
            #(0, 30, "Total")
        ]
    )

    power_bands = ["Noise", "Delta", "Theta", "Sigma", "Gamma"]

    state_mask = {
        0: "Wake",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
        }
    
    img_list = [[],[],[],[],[]]

    for band, i in zip(power_bands, range(len(power_bands))):
        for key, value in state_mask.items():
            filename = f"{image_path}/{band}_{value}_topoplot.png"
            fig = yasa.topoplot(bandpower.xs(key)[band], title=value, cbar_title="Spectral powers")
            fig.savefig(filename, dpi=600, bbox_inches="tight")
            plt.close(fig)
            img_list[i].append(filename)

    nrows, ncols = 5, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
    for y_ax, images in enumerate(img_list):
        # nrows, ncols = 1, 5
        # fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
        for x_ax, img_path in enumerate(images):
            ax = axes[y_ax, x_ax]
            image = img.imread(img_path)
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([]) 
            for spine in ax.spines.values():
                spine.set_visible(False)
            power = img_path.split("/")[-1].split("_")[0]
            #ax.set_title(power)
        axes[y_ax, 0].set_ylabel(power, rotation=90, size=12, labelpad=10)
        
    plt.tight_layout(h_pad=0)
    plt.savefig(os.path.join(image_path, "topology.svg"), format="svg")
    plt.show()

    bandpower.to_csv(bp_output)
