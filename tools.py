import mne  # type: ignore

import matplotlib.pyplot as plt
import numpy as np
import pycatch22 as catch22  # type: ignore
import os, gc
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt


def compute_features(x):
    """OBTAINED VIA THE CATCH22 WEBSITE"""
    res = catch22.catch22_all(x, catch24=True)
    return res["values"]


def get_features_w_multithread_from_3D(threeD_ts):
    """OBTAINED VIA THE CATCH22 WEBSITE"""
    all_features = []
    for twoD_ts in threeD_ts:
        threads_to_use = os.cpu_count()
        results_list = Parallel(n_jobs=threads_to_use)(
            delayed(compute_features)(twoD_ts[i]) for i in range(len(twoD_ts))
        )
        all_features.append(results_list)
        del results_list
        gc.collect()
    all_features = np.array(all_features)
    return all_features


def get_features_w_multithread_from_2D(twoD_ts):
    """OBTAINED VIA THE CATCH22 WEBSITE"""
    threads_to_use = os.cpu_count()
    results_list = Parallel(n_jobs=threads_to_use)(
        delayed(compute_features)(twoD_ts[i]) for i in range(len(twoD_ts))
    )
    results_list = np.array(results_list)
    return results_list


def get_epoch_from_event(raw, event_id, verbose=False):
    """
    Extracts epochs from an MNE Raw object based on a specified event type.

    This function creates epochs corresponding to a given event type defined in the
    annotations of the Raw object. It determines the epoch duration from the annotation
    duration of the specified event and returns the segmented data along with the event
    onsets and a list of dropped epochs.

    Note: with mne you can define an epoch centered on the event directly on the Raw,
    at the same time as you define the annotations.
    We're going to use the epochs we've already defined.

    Parameters:
    raw (mne.io.Raw): The input MNE Raw object containing data and annotations.
    event_id (dict): A dictionary mapping event names (as strings) to event IDs (ints).
    verbose (bool): If True, prints reasons for dropped epochs.

    Returns:
    tuple:
        - event_data (numpy.ndarray): Array of extracted epoch data for the specified event.
        - epoch_onsets (numpy.ndarray): Sample indices of event onsets.
        - list_epoch_drop (list): Indices of epochs that were dropped during preprocessing.
    """
    # ALL EVENT ID : {np.str_('blink'): 1, np.str_('button'): 2, np.str_('cardiac'): 3,
    # np.str_('deviant'): 4, np.str_('standard'): 5}
    events, event_id = mne.events_from_annotations(
        raw, event_id=event_id, verbose=False
    )
    sfreq = raw.info["sfreq"]

    annotations = raw.annotations

    events_name = list(event_id.keys())

    for event_name in events_name:
        for onset, duration, description in zip(
            annotations.onset, annotations.duration, annotations.description
        ):
            if (onset - duration) < 0:
                pass
            else:
                if str(description) == str(event_name):

                    tmax = round(float(duration), 2)
                    tmax = (tmax * sfreq - 1) / sfreq
                    break

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
        event_repeated="drop",
    )
    name_event = str(next(iter(event_id.keys())).item())
    event_data = epochs[name_event].get_data()

    list_epoch_drop = []
    for i, log in enumerate(epochs.drop_log):
        if log:
            list_epoch_drop.append(i)
            if verbose:
                print(f"Epoch {i} was dropped because: {log}")

    epoch_onsets = epochs.events[:, 0]

    return event_data, epoch_onsets, list_epoch_drop, epochs


def get_data_on_and_around_artefact(raw, event_id, verbose=False):
    """
    Extracts data on, pre_event, and post_event specified artefact-related events.

    This function identifies epochs around a specific event type (e.g., artefacts like blinks),
    and extracts the data corresponding to:
      - the event itself,
      - a time window before the event (pre-event),
      - and a time window after the event (post-event).

    Moreover, it ensures that pre- and post-event windows do not overlap with other events,
    and filters out invalid or overlapping segments. Dropped or out-of-bounds epochs are
    excluded from the final output.

    Parameters:
    raw (mne.io.Raw): The input MNE Raw object containing data and annotations.
    event_id (dict): A dictionary mapping event names (as strings) to event IDs (ints).
    verbose (bool): If True, prints detailed messages about dropped epochs or issues.

    Returns:
    tuple:
        - data_on (np.ndarray): Epoch data centered on the event.
        - data_pre_event (np.ndarray): Data segments preceding the event.
        - data_post_event (np.ndarray): Data segments following the event.
    """
    raw_copy = raw.copy()

    data_on, all_onset, list_epoch_drop, epochs = get_epoch_from_event(
        raw_copy, event_id, verbose=False
    )

    num_point_to_watch = data_on.shape[-1]

    all_data, _ = raw_copy[:, :]

    all_data_pre_event = []
    epochs_to_remove = []

    for i, onset in enumerate(all_onset):
        far_pre_event = int(onset - num_point_to_watch)
        if i in list_epoch_drop:
            epochs_to_remove.append(i)
        else:
            if far_pre_event < 0:
                epochs_to_remove.append(i)
                if verbose:
                    print("pre_event windows size out of range, not added")
            # MAKE SURE TO NOT TAKE EPOCHE pre_event INSIDE AN EVENT EPOCH
            else:
                for onset_2 in all_onset:
                    if abs(far_pre_event - onset_2) < num_point_to_watch - 1:
                        epochs_to_remove.append(i)
                        if verbose:
                            print(i)
                            print("event epochs too close to pre_event windows")
                        break
        data_pre_event = all_data[:, far_pre_event:onset]
        all_data_pre_event.append(data_pre_event)

    all_data_post_event = []

    for i, onset in enumerate(all_onset):
        far_post_event = int(onset + 2 * num_point_to_watch)
        if i in list_epoch_drop:
            epochs_to_remove.append(i)
        else:
            if all_data.shape[1] > far_post_event:
                # MAKE SURE TO NOT TAKE EPOCHE pre_event INSIDE AN EVENT EPOCH
                for onset_3 in all_onset:
                    if abs(far_post_event - onset_3) < num_point_to_watch - 1:
                        epochs_to_remove.append(i)
                        if verbose:
                            print(i)
                            print("event epochs too close to post_event windows")
                        break
            else:
                epochs_to_remove.append(i)
                if verbose:
                    print("post_event windows size out of range, not added")
        data_post_event = all_data[:, int(onset + num_point_to_watch) : far_post_event]
        all_data_post_event.append(data_post_event)

    epochs_to_remove = np.unique(epochs_to_remove)

    indices_to_keep = [
        i for i in range(len(all_data_pre_event)) if i not in epochs_to_remove
    ]

    data_on = [data_on[i] for i in indices_to_keep]
    all_data_pre_event = [all_data_pre_event[i] for i in indices_to_keep]
    all_data_post_event = [all_data_post_event[i] for i in indices_to_keep]
    onset_keep = [all_onset[i] for i in indices_to_keep]

    data_on = np.array(data_on)
    all_data_pre_event = np.array(all_data_pre_event)
    all_data_post_event = np.array(all_data_post_event)
    onset_keep = np.array(onset_keep)

    return data_on, all_data_pre_event, all_data_post_event, epochs, onset_keep


def compute_robust_std_per_channel(data, low_q=0.1, high_q=0.9):
    """
    Calcule un écart-type robuste par canal, en excluant les valeurs extrêmes.

    Parameters
    ----------
    data : np.ndarray
        Données brutes shape = (n_channels, n_times)
    low_q : float
        Quantile inférieur (e.g., 0.1)
    high_q : float
        Quantile supérieur (e.g., 0.9)

    Returns
    -------
    robust_stds : np.ndarray
        Écart-type robuste pour chaque canal, shape = (n_channels,)
    """
    assert data.ndim == 2, "Les données doivent avoir la forme (n_channels, n_times)"

    n_channels = data.shape[0]
    robust_stds = np.zeros(n_channels)

    for ch in range(n_channels):
        x = data[ch]
        q_low, q_high = np.quantile(x, [low_q, high_q])
        x_filtered = x[(x >= q_low) & (x <= q_high)]
        robust_stds[ch] = np.std(x_filtered)

    return robust_stds


def detect_epochs_to_clean(raw, epochs_data, multiplier=1.0):
    """
    Détecte les epochs dont l'écart-type dépasse l'écart-type global du canal.

    Parameters:
    ----------
    raw : mne.io.Raw
        Données brutes MNE (n_channels, n_times)
    epochs : mne.Epochs
        Epochs extraites (n_epochs, n_channels, n_times)
    multiplier : float
        Facteur multiplicatif pour ajuster le seuil (e.g. 1.5 pour seuil 1.5x plus grand)

    Returns:
    -------
    high_std_mask : np.ndarray
        Tableau booléen de forme (n_epochs, n_channels)
    """

    # 1. Données brutes : (n_channels, n_times)
    raw_data = raw.get_data()

    # 2. Écart-type global par canal
    global_std_per_channel = compute_robust_std_per_channel(
        raw_data, low_q=0.1, high_q=0.9
    )
    # global_mean_std = np.mean(global_std)

    # 4. Écart-type de chaque epoch : (n_epochs, n_channels)
    epoch_std = np.std(epochs_data, axis=2)

    # 5. Comparaison : epoch_std > multiplier * global_std
    high_std_mask = epoch_std > (multiplier * global_std_per_channel[np.newaxis, :])
    # high_std_mask = epoch_std > (multiplier * global_mean_std)

    return high_std_mask


def get_goal_features(features_pre, features_post, features_on=None):
    if features_on is not None:
        goal_features = (
            0.5 * features_pre + 2.0 * features_post + 0.5 * features_on
        ) / 3
    else:
        goal_features = (features_pre + features_post) / 2
    return goal_features


def plot_topomap_from_mask(mask, epochs, title="Détection de canaux anormaux"):

    # 1. Compter combien de fois chaque canal est marqué comme "anormal"
    # count_by_channel = mask.sum(axis=0)  # shape: (n_channels,)
    count_by_channel = mask.sum(axis=0).astype(float)  # shape: (n_channels,)
    # count_by_channel = mask.any(axis=0).astype(
    #     int
    # )  # 0 ou 1 par canal ; shape: (n_channels,)

    # 2. Sélectionner les canaux EEG uniquement (ou MEG selon ton dataset)
    picks = mne.pick_types(
        epochs.info,
        meg=True,
        eeg=False,
        eog=False,
        stim=False,
        ref_meg=False,
        exclude=[],
    )
    count_by_channel = count_by_channel[picks]

    # 3. Obtenir l'info pour les canaux sélectionnés
    info_picked = mne.pick_info(epochs.info, picks)

    # 3. Affichage avec colormap binaire
    fig, ax = plt.subplots(figsize=(6, 5))
    # im, _ = mne.viz.plot_topomap(
    #     count_by_channel,
    #     info_picked,
    #     axes=ax,
    #     show=False,
    #     cmap=plt.cm.get_cmap("RdBu", 2),  # 2 couleurs
    #     contours=0,
    #     # names=[epochs.ch_names[i] for i in picks],
    #     # show_names=False,
    #     # vmin=0,
    #     # vmax=1,
    # )
    im, _ = mne.viz.plot_topomap(
        count_by_channel,
        info_picked,
        axes=ax,
        show=False,
        cmap="Reds",
        contours=0,
        # vmin=0,
        # vmax=mask.shape[0],  # nombre maximal possible d'epochs anormales
        # names=[epochs.ch_names[i] for i in picks],
        # show_names=False
    )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Nombre d'epochs anormales")
    # cbar.ax.set_yticklabels(["Normal", "Anormal"])
    plt.tight_layout()
    plt.show()


def plot_psd(raw):
    data1PSD = raw.compute_psd(
        method="welch",
        fmin=1,
        fmax=70,
        picks="meg",
    )
    data1PSD.plot()
    plt.show()


def plot_time_series(raw):
    raw.plot(duration=10, title="Raw", picks="meg")
    plt.show()


def remplace_event_epoch(raw, clean_event, list_onset):
    raw_copy = raw.copy()

    duration = clean_event.shape[2]

    for i, onset_sample in enumerate(list_onset):
        raw_copy._data[:, onset_sample : onset_sample + duration] = clean_event[i]

    return raw_copy


def highpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Fréquence de Nyquist
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data, axis=1)
