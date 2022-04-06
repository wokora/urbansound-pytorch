import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd


class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_directory,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_directory = audio_directory
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        audio_sample_path = self._get_audio_sample_path(item)
        label = self._get_audio_sample_label(item)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resample = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resample(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, item):
        fold = f'fold{self.annotations.iloc[item, 5]}'
        file_name = self.annotations.iloc[item, 0]
        path = os.path.join(self.audio_directory, fold, file_name)
        return path

    def _get_audio_sample_label(self, item):
        return self.annotations.iloc[item, 6]
