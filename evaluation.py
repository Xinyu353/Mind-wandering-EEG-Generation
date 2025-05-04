

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch, cwt
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
import pandas as pd




fs = 256
delta_band = (1, 4)
theta_band = (4, 8)
alpha_band = (8, 13)
beta_band = (13, 30)
gamma_band = (30, 40)
time_range = (-0.398, 1.195)




def hjorth_parameters(signal, epsilon=1e-10):
    activity = np.var(signal)
    mobility = np.sqrt(np.var(np.diff(signal)) / (activity + epsilon))
    diff_signal = np.diff(signal)
    complexity = np.sqrt(np.var(np.diff(diff_signal)) / (np.var(diff_signal) + epsilon)) / (mobility + epsilon)
    return complexity


def petrosian_fractal_dimension(signal, epsilon=1e-10):
    diff_signal = np.diff(signal)
    binary_signal = np.sign(diff_signal)
    binary_signal[binary_signal == 0] = -1
    N = len(signal)
    M = np.sum(binary_signal[1:] != binary_signal[:-1])
    PFD = np.log(N+ epsilon) / (np.log(N+ epsilon) + np.log( (N+ epsilon) / (N + 0.4 * M+ epsilon)))
    return PFD




def kraskov_entropy(signal, k=3, d=1):
    N = len(signal)
    nn = NearestNeighbors(n_neighbors=k+1).fit(signal.reshape(-1, 1))
    distances, _ = nn.kneighbors(signal.reshape(-1, 1))
    distances = distances[:, 1:]
    phi_k = digamma(k)
    phi_N = digamma(N)
    V_d = np.pi**(d/2) / np.math.gamma(d/2 + 1)
    mean_log_dist = np.mean(np.log(2 * distances[:, -1]))
    KE = -phi_k + phi_N + np.log(V_d) + (d / N) * mean_log_dist
    return KE

def psd_bandpower(signal, fs=256, bands=None):
    if bands is None:
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'all': (0, 30)}
    f, Pxx = welch(signal, fs=fs, nperseg=fs*2)

    bandpower = {}
    for band, (low, high) in bands.items():
        idx_band = np.logical_and(f >= low, f <= high)
        bandpower[band] = np.sum(Pxx[idx_band])
    return (bandpower.get('delta', 0),
            bandpower.get('theta', 0),
            bandpower.get('alpha', 0),
            bandpower.get('beta', 0),
            bandpower.get('all', 0))


def compute_Wst(data, times, plot_on=True, save_path=None):

    if data.shape != times.shape:
        raise ValueError("Data and times should be in the same shape!")

    srate = compute_srate(times)
    tlag = np.linspace(0, 1000, len(times))
    scale = np.linspace(1, 3500, len(times))

    Wst = np.zeros((len(scale), len(tlag)))

    for ti, t in enumerate(tlag):
        for si, s in enumerate(scale):
            wavelet = mexican_hat(times, t, s)
            Wst[si, ti] = np.sum(data * wavelet) / np.sqrt(s)

    if plot_on:
        plt.figure(figsize=(10, 8))
        plt.imshow(Wst, aspect='auto', extent=[tlag[0], tlag[-1], scale[0], scale[-1]], origin='lower', cmap='coolwarm')
        plt.colorbar()
        plt.contour(tlag, scale, Wst, colors='k', linewidths=0.5)
        plt.ylabel('Scale (ms)')
        plt.xlabel('Time lag (ms)')
        plt.title('Wst Color Map')
        if save_path:
            plt.savefig(save_path, format='png', dpi=600)
        plt.clf()

    return Wst, tlag, scale

def mexican_hat(times, lag, scale):
    t = (times - lag) / scale
    return (1 - 16 * t ** 2) * np.exp(-8 * t ** 2)

def compute_srate(times):
    return 1000 / np.mean(np.diff(times))

def erp_features(Wst, tlag, scale, times):
    p1_window = (120 <= tlag) & (tlag <= 130)
    n1_window = (195 <= tlag) & (tlag <= 205)
    p3_window = (400 <= tlag) & (tlag <= 500)

    if np.any(p1_window):
        p1_amplitude = np.max(Wst[:, p1_window], axis=1)
        p1_latency = tlag[p1_window][np.argmax(p1_amplitude)]
        p1_amplitude = np.max(p1_amplitude)
    else:
        p1_amplitude = np.nan
        p1_latency = np.nan

    if np.any(n1_window):
        n1_amplitude = np.min(Wst[:, n1_window], axis=1)
        n1_latency = tlag[n1_window][np.argmin(n1_amplitude)]
        n1_amplitude = np.min(n1_amplitude)
    else:
        n1_amplitude = np.nan
        n1_latency = np.nan

    if np.any(p3_window):
        p3_amplitude = np.max(Wst[:, p3_window], axis=1)
        p3_latency = tlag[p3_window][np.argmax(p3_amplitude)]
        p3_amplitude = np.max(p3_amplitude)
    else:
        p3_amplitude = np.nan
        p3_latency = np.nan

    return p1_amplitude, n1_amplitude, p3_amplitude, p1_latency, n1_latency, p3_latency


def SpecEn(Sig, Fs=256, N=None, Freqs=(8, 12), Logx=np.exp(1), Norm=True):

    if isinstance(Sig, pd.DataFrame):
        Sig = Sig.values.flatten()
    elif isinstance(Sig, pd.Series):
        Sig = Sig.values

    Sig = np.squeeze(Sig)
    if N is None:
        N = 2 * len(np.squeeze(Sig)) + 1

    assert Sig.shape[0] > 10 and Sig.ndim == 1, "Sig: must be a numpy vector"
    assert N > 1 and isinstance(N, int), "N: must be an integer > 1"
    assert isinstance(Logx, (int, float)) and Logx > 0, "Logx: must be a positive value"
    assert isinstance(Freqs, tuple) and len(Freqs) == 2 and 0 <= Freqs[0] <= Fs / 2 and 0 <= Freqs[1] <= Fs / 2, \
        "Freqs: must be a two element tuple with values in range [0, Fs/2]. The values must be in increasing order."

    Fx = int(np.ceil(N / 2))
    Freqs = np.round(np.array(Freqs) / (Fs / 2) * Fx).astype(int) - 1
    Freqs[Freqs == -1] = 0

    if Freqs[0] > Freqs[1]:
        raise Exception('Lower band frequency must come first.')
    elif Freqs[1] - Freqs[0] < 1:
        raise Exception('Spectrum resolution too low to determine bandwidth.')
    elif min(Freqs) < 0 or max(Freqs) > Fx:
        raise Exception('Freqs must be normalized w.r.t. sampling frequency [0 Fs/2].')

    Pt = abs(np.fft.fft(np.convolve(Sig, Sig), N))
    Pxx = Pt[:Fx] / sum(Pt[:Fx])


    Spec = -sum(Pxx * np.log(Pxx)) / np.log(Logx)
    if Norm:
        Spec = Spec / (np.log(len(Pxx)) / np.log(Logx))


    Pxx_band = Pxx[Freqs[0]:Freqs[1] + 1]
    Pband = Pxx_band / sum(Pxx_band)
    BandEn = -sum(Pband * np.log(Pband)) / np.log(Logx)
    if Norm:
        BandEn = BandEn / (np.log(len(Pxx_band)) / np.log(Logx))

    return Spec, BandEn
