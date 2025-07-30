This is a deposit made by Andéol FOURNIER, undergraduate research trainee in the Brain Imaging Centre Laboratory - Neuroimaging and Neuroinformatics Unit at the Montreal Neurological Institute, Faculty of Medicine, McGill University under the supervision of Sylvain Baillet, Phd.

# MEG Raw Artifact Correction using Generative Models

This repository provides scripts and tools for applying generative models to correct artifacts in MEG raw data files.

## 📦 Requirements

For a simplified and reproducible setup, it is recommended to use **[Poetry](https://python-poetry.org/)** to manage dependencies and the virtual environment. (files .toml and .lock are located at the root of the project)

---

## Repository Structure

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
