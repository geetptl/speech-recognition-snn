import os
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import TensorDataset


def get_label(file_name):
    return int(file_name.split("_")[0])


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


n_fft = 2048
win_length = None
hop_length = 128
n_mels = 512
n_mfcc = 64


def load(dir_path):
    audio_files = os.listdir(dir_path)[:3000]

    labels = [get_label(file) for file in audio_files]
    inputs = []
    for file in audio_files:
        waveform, sample_rate = torchaudio.load(dir_path + file, normalize=True)

        mfcc_transform = transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "mel_scale": "htk",
            },
        )

        mfcc = mfcc_transform(waveform)
        inputs.append(mfcc[0])

    max_size = max([input_.shape[1] for input_ in inputs])
    inputs = [nn.functional.pad(input_, (0, max_size - input_.shape[1]), "constant", 0) for input_ in inputs]
    inputs = torch.stack(inputs)
    inputs = inputs.unsqueeze(1)

    labels = torch.Tensor(labels).type(torch.LongTensor)

    dataset = TensorDataset(inputs, labels)

    return dataset
