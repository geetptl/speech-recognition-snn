import numpy as np
import os
from scipy.signal import spectrogram
from scipy.io import wavfile
from python_speech_features.sigproc import framesig


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
        bands.append(np.sum(band0) / (np.shape(band0)[0] if np.shape(band0)[0] > 0 else 1))
        bands.append(np.sum(band1) / (np.shape(band1)[0] if np.shape(band1)[0] > 0 else 1))
        bands.append(np.sum(band2) / (np.shape(band2)[0] if np.shape(band2)[0] > 0 else 1))
        bands.append(np.sum(band3) / (np.shape(band3)[0] if np.shape(band3)[0] > 0 else 1))
        bands.append(np.sum(band4) / (np.shape(band4)[0] if np.shape(band4)[0] > 0 else 1))
        features.append(bands)

    return features


def load(audio_files, path):
    labels = []
    features = []

    for audio_file in audio_files:
        labels.append(get_label(audio_file))
        features.append(get_features(os.path.join(path, audio_file)))

    features = np.array(features)
    features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))
    labels = np.array(labels)
    return labels, features
