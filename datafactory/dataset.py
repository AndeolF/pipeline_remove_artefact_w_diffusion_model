import os, json
import torch  # type: ignore
import ast
import numpy as np
import pandas as pd
from torch.utils.data import Dataset  # type: ignore
from sklearn.preprocessing import MinMaxScaler


class DatasetFeaturesSeries(Dataset):
    def __init__(self, features, series):
        self.features = features
        self.series = series
        assert (
            self.features.shape[0] == self.series.shape[0]
        ), f"Mismatch in number of samples : features {self.features.shape[0]} ; ts : {self.series.shape[0]}"

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if isinstance(self.features[idx], torch.Tensor):
            feature = (
                self.features[idx]
                .clone()
                .detach()
                .requires_grad_(False)
                .to(dtype=torch.float32)
            )
            series = (
                self.series[idx]
                .clone()
                .detach()
                .requires_grad_(False)
                .to(dtype=torch.float32)
            )
        else:
            feature = torch.tensor(
                self.features[idx], dtype=torch.float32
            )  # shape [24]
            series = torch.tensor(self.series[idx], dtype=torch.float32)  # shape [T]
        return series, feature
