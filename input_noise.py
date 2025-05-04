import numpy as np
import torch
from scipy.signal import butter, filtfilt


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def generate_white_noise(batch_size, noise_dim, lowcut=0.5, highcut=40.0, fs=256, device='cpu'):
    white_noises = []
    for _ in range(batch_size):
        white_noise = np.random.normal(size=noise_dim)
        filtered_noise = bandpass_filter(white_noise, lowcut, highcut, fs)
        white_noises.append(filtered_noise)

    white_noises = np.array(white_noises)  # (batch_size, noise_dim)
    return torch.tensor(white_noises, dtype=torch.float32, device=device)


def generate_pink_noise_torch(batch_size, noise_dim, alpha=1.0, lowcut=0.5, highcut=40.0, fs=256, device='cpu'):
    def generate_pink_noise(N, alpha):
        white_noise = np.random.normal(size=N)
        fft = np.fft.rfft(white_noise)
        frequencies = np.fft.rfftfreq(N)
        S_f = np.where(frequencies == 0, 1, frequencies ** (-alpha / 2))
        fft = fft * S_f
        pink_noise = np.fft.irfft(fft, n=N)
        return pink_noise

    filtered_noises = []

    for _ in range(batch_size):
        pink_noise = generate_pink_noise(noise_dim, alpha)
        filtered_noise = bandpass_filter(pink_noise, lowcut, highcut, fs)
        filtered_noises.append(filtered_noise)

    filtered_noises = np.array(filtered_noises)
    filtered_noises_torch = torch.tensor(filtered_noises, dtype=torch.float32, device=device)

    return filtered_noises_torch


import numpy as np
import torch


