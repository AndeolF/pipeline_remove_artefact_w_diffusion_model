import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch  # type: ignore
from scipy.stats import norm

from model.denoiser.transformer import Transformer
from model.backbone.DDPM import DDPM
import math
import pycatch22 as catch22  # type: ignore
from scipy.signal import welch
from scipy.signal import firwin
import torch.nn.functional as F  # type: ignore
from tqdm import tqdm
from torch.utils.data import DataLoader  # type: ignore
from model.backbone.rectified_flow import RectifiedFlow
from datafactory.dataset import DatasetFeaturesSeries
import pickle


class TsFromFeaturesWithDiffusion:
    def __init__(
        self,
        device: str,
        VAE_path: str,
        diffusion_model_path: str,
        path_to_std_ref: str,
        path_to_ts_scaler: str,
        path_to_features_scaler: str,
    ):
        """
        Initializes time series generator from characteristics.

        Args:
            device (str): The device used ('cpu' or 'cuda').
                        VAE_path (str): Path to saved VAE model.
            diffusion_model_path (str): Path to saved diffusion model.
        """
        self.device = device
        self.VAE_path = VAE_path
        self.diffusion_model_path = diffusion_model_path
        self.path_to_std_ref = path_to_std_ref

        with open(path_to_ts_scaler, "rb") as file:
            self.ts_scaler = pickle.load(file)
        with open(path_to_features_scaler, "rb") as file:
            self.features_scaler = pickle.load(file)

        self.vae = self.load_vae_model()
        self.diffusion_model = self.load_diffusion_model(self.vae)
        self.rf = RectifiedFlow(
            path_to_std_ref=self.path_to_std_ref, device=self.device
        )

    def load_vae_model(self):
        """
        Loads VAE model from self.VAE_path.

        Returns:
            The loaded VAE model.
        """
        VAE = torch.load(
            self.VAE_path,
            map_location=torch.device(self.device),
        )
        VAE.float().to(self.device).eval()
        return VAE

    def load_diffusion_model(self, VAE):
        """
        Loads the diffusion model from self.diffusion_model_path.

        Returns:
            The loaded diffusion model.
        """
        denoising_model = Transformer
        denoising_model = denoising_model().to(self.device)
        denoising_model.encoder = VAE.encoder

        saved_dict = torch.load(
            self.diffusion_model_path, map_location=torch.device(self.device)
        )
        denoising_model.load_state_dict(saved_dict["model"])
        denoising_model.to(self.device).eval()
        return denoising_model

    @staticmethod
    def get_power_t_schedule(step, gamma=3.0, device="cuda"):
        """
        Generates a non-uniform sequence t of [1, 0], with more steps towards t ~ 0
        gamma > 1 → denser towards 0
        gamma < 1 → denser towards 1
        """
        s = torch.linspace(0, 1, steps=step, device=device)
        t = 1 - s**gamma
        return t

    @staticmethod
    def format_with_batch(vector):
        if isinstance(vector, torch.Tensor):
            if len(vector.shape) > 1:
                return vector
            else:
                return vector.unsqueeze(0)
        elif isinstance(vector, np.ndarray):
            if len(vector.shape) > 1:
                return vector
            else:
                return np.expand_dims(vector, axis=0)
        else:
            return vector

    def infer(
        self,
        raw_ts_input,
        features_input,
        cut_freq,
        total_step=100,
        cfg_scale=4,
        mask=None,
    ):
        """
        Performs inference to generate a conditioned time series.

        Args:
            raw_ts_input: Raw time series (initial input).
            features_input: Feature vectors to guide generation (they must be standardise but compute on no standardise time series !).
            total_step: Total number of steps in broadcast.
            cfg_scale: Scale configuration for guidance.
            cut_freq: Cut frequency (if applicable).

        Returns:
            Time series generated after inference (thoses times series are standardise).
        """

        raw_ts_input_np = raw_ts_input.copy()
        features_input = torch.tensor(features_input, dtype=torch.float32).to(
            self.device
        )
        raw_ts_input = torch.tensor(raw_ts_input, dtype=torch.float32).to(self.device)

        raw_ts_input = self.format_with_batch(raw_ts_input)
        features_input = self.format_with_batch(features_input)

        infer_set = DatasetFeaturesSeries(features=features_input, series=raw_ts_input)
        infer_loader = DataLoader(infer_set, batch_size=64, shuffle=False)

        x_t_list = []
        feat_list = []

        std_ref = np.load(self.path_to_std_ref)
        std_ref = torch.from_numpy(std_ref).to(self.device)

        f_s = 1200
        f_c = cut_freq + 30
        numtaps = 101
        window_choose = ("kaiser", 8.6)  # ("kaiser", 8.6) "hamming"

        kernel_np = firwin(numtaps=numtaps, cutoff=f_c, fs=f_s, window=window_choose)
        kernel = (
            torch.tensor(kernel_np, dtype=torch.float32).view(1, 1, -1).to(self.device)
        )

        with torch.no_grad():
            for data in tqdm(infer_loader):

                raw_ts, feat = data
                raw_ts = raw_ts.float().to(self.device)
                feat = feat.float().to(self.device)
                raw_ts_latent, _ = self.diffusion_model.encoder(raw_ts)
                raw_ts_latent_copy = raw_ts_latent.clone()

                # WITH A WHITE STD NOISE
                x_t_latent = (
                    torch.randn_like(raw_ts_latent_copy).float().to(self.device)
                )
                x_t_latent = (x_t_latent / x_t_latent.std()) * std_ref

                # t SCHEDULER POWER
                t_schedule = self.get_power_t_schedule(
                    step=total_step + 1, gamma=3.0, device=self.device
                )

                for j in range(total_step):

                    # t SCHEDULER UNIFORM
                    # t = (
                    #     torch.round(
                    #         torch.full(
                    #             (x_t_latent.shape[0],), j * 1.0 / step, device=device
                    #         )
                    #         * step
                    #     )
                    #     / step
                    # )

                    # t SCHEDULER POWER
                    t = t_schedule[-j - 1].expand(x_t_latent.shape[0])
                    t_next = t_schedule[-j - 2].expand(x_t_latent.shape[0])

                    # # # EVAL WITH EULER  min(torch.mean(t_next-t),1/args.total_step)
                    pred_uncond = self.diffusion_model(
                        input=x_t_latent, t=t, text_input=None
                    )
                    pred_cond = self.diffusion_model(
                        input=x_t_latent, t=t, text_input=feat
                    )
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t_latent = self.rf.euler(
                        x_t=x_t_latent, v=pred, dt=1 / total_step
                    )

                    # EVAL WITH HEUN
                    # x_t_latent = self.rf.heun(model=self.diffusion_model, x_t=x_t_latent, t=torch.mean(t), t_next=torch.mean(t_next), feat=feat, cfg_scale=cfg_scale, dt=1/args.total_step)

                x_t, _ = self.vae.decoder(x_t_latent, length=raw_ts.shape[-1])

                # WITH A FLITRED ON THE OUTPUT
                if len(x_t.shape) == 1:
                    x_t = x_t.unsqueeze(0)
                x_t = x_t.unsqueeze(1)
                x_t_filtred = F.conv1d(x_t, kernel, padding=numtaps // 2)
                x_t_filtred = x_t_filtred.squeeze(1)
                x_t = x_t_filtred

                x_t = x_t.detach().cpu().numpy().squeeze()
                feat = feat.detach().cpu().numpy().squeeze()

                x_t_list.append(x_t)
                feat_list.append(feat)

        x_t_array = np.concatenate(x_t_list, axis=0)
        feat_array = np.concatenate(feat_list, axis=0)

        x_t = x_t_array
        feat = feat_array

        x_t = self.format_with_batch(x_t)
        feat = self.format_with_batch(feat)

        return raw_ts_input_np, x_t, feat

    def standardise_TS(self, ts_no_std, in_batch=False):
        if in_batch:
            return self.ts_scaler.transform(ts_no_std.reshape(1, -1))
        else:
            return self.ts_scaler.transform(ts_no_std)

    def destandardise_TS(self, ts_std, in_batch=False):
        if in_batch:
            return self.ts_scaler.inverse_transform(ts_std.reshape(1, -1))
        else:
            return self.ts_scaler.inverse_transform(ts_std)

    def standardise_features(self, features_no_std, in_batch=False):
        if in_batch:
            return self.features_scaler.transform(features_no_std.reshape(1, -1))
        else:
            return self.features_scaler.transform(features_no_std)

    def destandardise_features(self, features_std, in_batch=False):
        if in_batch:
            return self.features_scaler.inverse_transform(features_std.reshape(1, -1))
        else:
            return self.features_scaler.inverse_transform(features_std)
