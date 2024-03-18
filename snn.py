import argparse
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import os
import sys
from datetime import datetime
import copy
import math
from scipy.signal import spectrogram
from scipy.io import wavfile
from python_speech_features.sigproc import framesig
import librosa
import random as random
from matplotlib.pyplot import specgram
import wave


def get_label(file_name):
    return file_name.split("_")[0]


def get_features(file_name):
    rate, signal = wavfile.read(file_name)
    print(signal.shape)
    plt.plot(signal)
    plt.show()

    N = 40
    gamma = 0.5
    window_size = signal.shape[0] / (N * (1 - gamma) + gamma)
    frames = framesig(signal, window_size, window_size * 0.5)
    print(frames.shape)

    plt.imshow(frames)
    plt.show()

    weighting = np.hanning(window_size)

    fft = np.fft.fft(frames * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    scale = np.sum(weighting**2) * rate
    fft[1:-1, :] *= 2.0 / scale
    fft[(0, -1), :] /= scale
    fft = np.log(np.clip(fft, a_min=1e-6, a_max=None))
    plt.imshow(fft)
    plt.show()

    freqs = float(rate) / window_size * np.arange(fft.shape[0])

    features = []
    band0 = []
    band1 = []
    band2 = []
    band3 = []
    band4 = []

    frames = []
    for i in range(40):
        bands = []
        band0 = []
        band1 = []
        band2 = []
        band3 = []
        band4 = []
        j = 0
        for freq in freqs:
            if freq <= 333.3:
                band0.append(fft[j][i])
            elif freq > 333.3 and freq <= 666.7:
                band1.append(fft[j][i])
            elif freq > 666.7 and freq <= 1333.3:
                band2.append(fft[j][i])
            elif freq > 1333.3 and freq <= 2333.3:
                band3.append(fft[j][i])
            elif freq > 2333.3 and freq <= 4000:
                band4.append(fft[j][i])
            j += 1
        bands.append(
            np.sum(band0) / (np.shape(band0)[0] if np.shape(band0)[0] > 0 else 1)
        )
        bands.append(
            np.sum(band1) / (np.shape(band1)[0] if np.shape(band1)[0] > 0 else 1)
        )
        bands.append(
            np.sum(band2) / (np.shape(band2)[0] if np.shape(band2)[0] > 0 else 1)
        )
        bands.append(
            np.sum(band3) / (np.shape(band3)[0] if np.shape(band3)[0] > 0 else 1)
        )
        bands.append(
            np.sum(band4) / (np.shape(band4)[0] if np.shape(band4)[0] > 0 else 1)
        )
        frames.append(bands)

    return frames


def get_spikes(features):
    pass


def run(path):
    audio_files = os.listdir(path)[:1]

    labels = []
    features = []
    input_spikes = {}

    for audio_file in audio_files:
        print(audio_file)
        full_path = os.path.join(path, audio_file)
        with open(full_path, "r") as f:
            labels.append(get_label(audio_file))
            features.append(get_features(full_path))
            print(features[-1][0])
            # input_spikes[audio_file] = get_spikes(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nissl Cell Segmentation")
    parser.add_argument(
        "--path",
        default="./.data/recordings",
        type=str,
        help="Path to the downloaded data files",
    )
    args = parser.parse_args()

    run(args.path)
