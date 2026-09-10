This is a deposit made by Andéol FOURNIER, undergraduate research trainee in the Brain Imaging Centre Laboratory - Neuroimaging and Neuroinformatics Unit at the Montreal Neurological Institute, Faculty of Medicine, McGill University under the supervision of Sylvain Baillet, PhD.

# MEG Signal Artifact Correction using Latent Diffusion Models

## Project Description
The objective of this project is to correct artifacts (ocular, cardiac, muscular) in magnetoencephalography (MEG) signals by precisely targeting the corrupted segments without altering the artifact-free periods, in contrast to global methods such as ICA or SSP. This approach relies on a latent diffusion model conditioned on temporal features.

> *Note: Pre- and post-correction signal comparisons (inference examples) are available in the [technical_report_LDM.pdf](technical_report_LDM.pdf) file included in this repository.*

## Methodology

### Feature Extraction (Catch22)
The evaluation of signal dynamics is based on the Catch22 feature set ([Lubba et al., 2019](https://doi.org/10.1007/s10618-019-00647-x)). These 22 features are selected for their representativeness among the 4,791 *hctsa* features and benefit from an optimized C implementation for rapid computation. The fundamental assumption is that these extracted features evolve slowly over time. Consequently, to correct a corrupted time window, the model leverages the features of the artifact-containing window as well as those of the immediately adjacent windows (pre-event and post-event).

### Model Architecture
The denoising model is a conditioned latent diffusion model. It is an adaptation of the architecture proposed in *T2S: High-resolution Time Series Generation with Text-to-Series Diffusion Models* (Ge et al., 2025 - [paper link](https://arxiv.org/pdf/2505.02417)).

*   **VQ-VAE (Vector Quantized Variational Autoencoder):** Pre-trained for strict reconstruction on clean signal segments. Frozen during the diffusion training, it structures the latent space and contributes to the denoising process.
*   **Latent Diffusion Model (Flow Matching & DiT):** The generative process relies on a *Flow Matching* framework solved by a *Diffusion Transformer (DiT)*. The latter is conditioned by the feature vector via an Adaptive Layer Normalization (AdaLN) mechanism, which replaces the standard textual conditioning.

**Simplified Architecture Diagram:**
```text
+-------------------+       +-----------------+       +-------------------+
|                   |       |                 |       |                   |
| Noisy Time Series | ----> | VQ-VAE Encoder  | ----> |   Latent Space    |
| (Target Window)   |       |                 |       |   Representation  |
|                   |       +-----------------+       |                   |
+-------------------+                                 +---------+---------+
                                                                |
                                                                v
+-------------------+       +-----------------+       +---------+---------+
|                   |       |                 |       |                   |
| Catch22 Features  | ----> |      AdaLN      | ----> |  DiT (Denoiser)   |
| (Conditioning)    |       |  Conditioning   |       |  (Flow Matching)  |
|                   |       |                 |       |                   |
+-------------------+       +-----------------+       +---------+---------+
                                                                |
                                                                v
+-------------------+       +-----------------+       +---------+---------+
|                   |       |                 |       |                   |
| Corrected Output  | <---- | VQ-VAE Decoder  | <---- |  Denoised Latent  |
| Time Series       |       |                 |       |  Representation   |
|                   |       +-----------------+       |                   |
+-------------------+                                 +-------------------+
```

### Self-Supervised Training Strategy
Training leverages the OMEGA MEG database. The model is trained exclusively on pairs (time series, features) extracted from artifact-free epochs. The objective is to reconstruct a clean series from its own descriptors. This self-supervised approach eliminates the reliance on an artificial ground truth, allowing the model to generalize effectively to real noisy signals.


*(Mathematical details, training setup, and algorithmic inference choices are comprehensively detailed in the [technical_report_LDM.pdf](technical_report_LDM.pdf) document attached to this repository).*


## Inference Pipeline
During application on an MEG recording, the pipeline executes the following steps:
1. Extraction of epochs containing the artifact (*on-event*), as well as adjacent epochs (*pre-event* and *post-event*).
2. Computation of the Catch22 features on these three segments.
3. Combination of the features via a weighted average to generate the target conditioning vector.
4. Generation of a new time series in the latent space by the diffusion model, followed by decoding.
5. Replacement of the noisy *on-event* epoch with the generated series in the initial signal.

## Limitations
* The current architecture applies single-channel correction. It does not leverage the spatial correlations and the multiplicity of sensors inherent to MEG/EEG recordings.

## Other usefull repository
The implementation is base on two other distinct repositories:
* `CREATION_DATASET_CORRECTION_ARTEFACT`: Scripts for dataset creation (MEG segment alignment and feature extraction).
* `Diffusion_model_TS_from_features`: Architecture of the VQ-VAE, the latent diffusion model (Flow Matching), and training scripts.


---

# Requirements

For a simplified and reproducible setup, it is recommended to use **[Poetry](https://python-poetry.org/)** to manage dependencies and the virtual environment. (files .toml and .lock are located at the root of the project)

---

# Repository Structure

- **`datafactory/`**  
  Contains scripts to import and format the extracted data properly.

- **`model/`**  
  Includes the architectures for the models used: VAE (Variational Autoencoder) and Diffusion models, along with their respective backbones.

- **`model_save/`**  
  Stores all necessary components to use the models:
  - Pretrained weights for the VAE and Diffusion models
  - Feature and time-series scalers used during training
  - `std_latent` values used for normalization in the latent space

- **`class_LDM.py`**  
  Defines the `Latent Diffusion Model (LDM)` class and all required methods to load and use the trained Diffusion model.  
  This model is based on the one from the `diffu_from_features` repository.

- **`pipeline_remove_artefact_from_raw.py`**  
  Implements a full processing pipeline that:
  1. Takes a raw MEG file as input
  2. Uses the trained model to correct artifacted segments
  3. Saves the cleaned raw file

  > **Note**: Not all epochs are modified. Only clean pre- and post-event epochs (i.e., free from noise) are used, similar to the strategy applied in the `metric-pre-processing` step.

- **`tools.py`**  
  Contains utility functions required by the pipeline.

---
