import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import spectrogram
from scipy.io import wavfile
from python_speech_features.sigproc import framesig
import random as random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split


def get_label(file_name):
    return int(file_name.split("_")[0])


def get_features(file_name):
    rate, signal = wavfile.read(file_name)
    # print(signal.shape)
    # plt.plot(signal)
    # plt.show()

    N = 40
    gamma = 0.5
    window_size = signal.shape[0] / (N * (1 - gamma) + gamma)
    frames = framesig(signal, window_size, window_size * 0.5)
    # print(frames.shape)
    # plt.imshow(frames)
    # plt.show()

    weighting = np.hanning(window_size)

    fft = np.fft.fft(frames * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    scale = np.sum(weighting**2) * rate
    fft[1:-1, :] *= 2.0 / scale
    fft[(0, -1), :] /= scale
    fft = np.log(np.clip(fft, a_min=1e-6, a_max=None))
    # plt.imshow(fft)
    # plt.show()

    freqs = float(rate) / window_size * np.arange(fft.shape[0])

    features = []

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
        features.append(bands)

    return features


class SDDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class Izhi:
    def __init__(self, a, b, c, d, Vth, T, dt):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.Vth = Vth
        self.u = self.b * self.c
        self.T = T
        self.dt = dt
        self.t = np.arange(0, self.T, self.dt)
        self.in_synapes = []
        self.out_synapes = []

    # I is an array of length self.t
    def run(self, I):
        V = np.zeros(len(self.t)) #Initialize a numpy array containing the membrane voltages for all the timesteps
        V[0] = self.c #Initial membrane voltage is the rest potential, defined by the parameter 'c'
        u = np.zeros(len(self.t)) #Initialize a numpy array containing u for all the timesteps
        u[0] = self.u #Initial u
        num_spikes = 0
        for t in range(1, len(self.t)): #the time loop for performing euler's integration
            dv = ((0.04 * V[t-1]**2) + (5 * V[t-1]) + 140 - u[t-1] + I[t-1]) * self.dt
            du = (self.a * ((self.b * V[t-1]) - u[t-1])) * self.dt
            V[t] = V[t-1] + dv
            u[t] = u[t-1] + du
            
            #condition for when membrane potential is greater than the threshold
            if V[t] >= self.Vth:
                V[t] = self.c
                u[t] = self.d + u[t]
                num_spikes += 1

        return V, num_spikes


class Synapse:
    def __init__(self):
        self.weight = 1
    
    def hebbian(self):
        pass

    def anti_hebbian(self):
        pass


class Network:
    def __init__(self):
        self.layer_1 = [Izhi() for i in range(200)]
        self.synapse_layer = [[Synapse() for j in range(200)] for i in range(10)]
        self.layer_2 = [Izhi() for i in range(10)]

        for i, synapse1_ in enumerate(self.synapses):
            for j, synapse_ in enumerate(synapse1_):
                layer_2[i].in_synapes.append(synapse_)
                layer_1[j].out_synapes.append(synapse_)

    def forward(self, feature, label):
        # print("hello")
        pass


def run(path):
    audio_files = os.listdir(path)[:100]

    labels = []
    features = []
    input_spikes = {}

    for audio_file in audio_files:
        full_path = os.path.join(path, audio_file)
        with open(full_path, "r") as f:
            labels.append(get_label(audio_file))
            features.append(get_features(full_path))

    features = np.array(features)
    features = np.reshape(
        features, (features.shape[0], features.shape[1] * features.shape[2])
    )
    labels = np.array(labels)

    print(features.shape)
    print(labels.shape)

    # dataset = SDDataset(features, labels)
    # train, test = random_split(dataset, [0.7, 0.3])

    # print(len(train))
    # print(len(test))


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
