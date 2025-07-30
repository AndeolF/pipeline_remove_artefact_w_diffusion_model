import mne  # type: ignore
import numpy as np
import time
import torch  # type: ignore
import os
import sys
from class_LDM import TsFromFeaturesWithDiffusion
from tools import (
    get_epoch_from_event,
    get_data_on_and_around_artefact,
    plot_psd,
    plot_time_series,
    remplace_event_epoch,
    get_features_w_multithread_from_3D,
    detect_epochs_to_clean,
    plot_topomap_from_mask,
    get_features_w_multithread_from_2D,
    get_goal_features,
    highpass_filter,
)


def pipeline_rm_artefact_with_DiffuFeatures(DiffuModel, raw, event, device):
    raw_copy = raw.copy()

    (
        data_on_event_original,
        data_pre_event_original,
        data_post_event_original,
        epochs,
        onset_keep,
    ) = get_data_on_and_around_artefact(raw_copy, event)

    print(f"data_on_event_original.shape : {data_on_event_original.shape}")

    # TEST QUELLES CHANNEL TOUCHER
    high_std_mask = detect_epochs_to_clean(
        raw_copy, data_on_event_original, multiplier=1.0
    )
    # plot_topomap_from_mask(high_std_mask, epochs, title="Détection de canaux anormaux")

    # A PARTIR D'ICI, ON NE VA TRAVAILLER QU'AVEC LES DONNÉES ANORMAL POUR GAGNER DU TEMPS DE CALCULE
    events_idx, chan_idx = np.where(high_std_mask)

    anomalous_data_on_event_original = data_on_event_original[events_idx, chan_idx, :]
    anomalous_data_pre_event_original = data_pre_event_original[events_idx, chan_idx, :]
    anomalous_data_post_event_original = data_post_event_original[
        events_idx, chan_idx, :
    ]
    print(
        f"anomalous_data_on_event_original.shape : {anomalous_data_on_event_original.shape}"
    )

    # SI ON FILTRE LES DATA ON
    # fs = raw_copy.info["sfreq"]  # en Hz
    # cutoff = 2  # Hz
    # anomalous_data_on_event_original_filtered = highpass_filter(
    #     anomalous_data_on_event_original, cutoff=cutoff, fs=fs
    # )

    # GET THE FEATURES
    features_original_pre_event_anomalous_data = get_features_w_multithread_from_2D(
        anomalous_data_pre_event_original
    )
    features_original_post_event_anomalous_data = get_features_w_multithread_from_2D(
        anomalous_data_post_event_original
    )
    features_original_on_event_anomalous_data = get_features_w_multithread_from_2D(
        anomalous_data_on_event_original
    )

    goal_features_anomalous_data = get_goal_features(
        features_pre=features_original_pre_event_anomalous_data,
        features_post=features_original_post_event_anomalous_data,
        features_on=features_original_on_event_anomalous_data,
    )

    # # STD THE FEATURES
    goal_features_anomalous_data_std = DiffuModel.standardise_features(
        features_no_std=goal_features_anomalous_data
    )

    print("\nstart inferring ...\n")

    _, corrected_anomalous_data_on_event_original_std, _ = DiffuModel.infer(
        raw_ts_input=anomalous_data_on_event_original,
        features_input=goal_features_anomalous_data_std,
        cut_freq=200,
        total_step=100,
        cfg_scale=4,
    )

    print(corrected_anomalous_data_on_event_original_std.shape)

    # # DE-STD THE TS
    corrected_anomalous_data_on_event_original = DiffuModel.destandardise_TS(
        ts_std=corrected_anomalous_data_on_event_original_std
    )

    print(corrected_anomalous_data_on_event_original.shape)

    #  Réinjecter les segments corrigés
    data_on_event_corrected = data_on_event_original.copy()
    for i, (e_idx, c_idx) in enumerate(zip(events_idx, chan_idx)):
        data_on_event_corrected[e_idx, c_idx, :] = (
            corrected_anomalous_data_on_event_original[i]
        )

    print(data_on_event_corrected.shape)

    raw_clean_with_diffu = remplace_event_epoch(
        raw_copy, data_on_event_corrected, onset_keep
    )

    return raw_clean_with_diffu


if __name__ == "__main__":
    name_raw = "sub-0029_ses-02_run-01"

    raw_with_annotations = mne.io.read_raw_fif(
        "raw_data/" + name_raw + "/raw_with_annotations.fif",
        preload=True,
        verbose=False,
    )
    picks_initial = mne.pick_types(
        raw_with_annotations.info,
        meg=True,
        eeg=False,
        eog=False,
        stim=False,
        ref_meg=False,
    )
    raw_with_annotations = raw_with_annotations.pick(picks_initial)
    raw_with_annotations.resample(sfreq=1200)
    raw_with_annotations.filter(l_freq=0.5, h_freq=200, verbose=False)

    # plot_time_series(raw_with_annotations)

    # DEFINE THE EVENT
    bool_blink = True
    bool_cardiac = False

    if bool_blink:
        event = {np.str_("blink"): 1}
    elif bool_cardiac:
        event = {np.str_("cardiac"): 3}
    else:
        event = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"\nDevice : {device}\n")

    start = time.time()

    diffu_model = TsFromFeaturesWithDiffusion(
        device=device,
        VAE_path="Model_save/model_600p_200Hz/final_VAEpretrain_model.pth",
        diffusion_model_path="Model_save/model_600p_200Hz/model.pth",
        path_to_std_ref="Model_save/model_600p_200Hz/std_latent_data_train_encode.npy",
        path_to_ts_scaler="Model_save/model_600p_200Hz/ts_scaler.pkl",
        path_to_features_scaler="Model_save/model_600p_200Hz/features_scaler.pkl",
    )

    raw_clean_with_diffu = pipeline_rm_artefact_with_DiffuFeatures(
        DiffuModel=diffu_model, raw=raw_with_annotations, event=event, device=device
    )

    print(f"Time for the pipeline : {time.time()-start:.2f} secondes")
    # plot_time_series(raw_clean_with_diffu)

    raw_clean_with_diffu.save(
        "raw_data/" + name_raw + "/raw_clean_with_diffu_ponderate_coef_multi1.fif",
        overwrite=True,
    )
