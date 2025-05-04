import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_Wst(data, times, plot_on=True, save_path=None):
    if data.shape != times.shape:
        raise ValueError("Data and times should be same!")

    srate = compute_srate(times)
    tlag = np.linspace(0, 1000, len(times))
    scale = np.linspace(1, 3500, len(times))

    Wst = np.zeros((len(scale), len(tlag)))

    for ti, t in enumerate(tlag):
        for si, s in enumerate(scale):
            wavelet = mexican_hat(times, t, s)
            Wst[si, ti] = np.sum(data * wavelet) / np.sqrt(s)

    if plot_on:
        plt.figure(figsize=(5, 4))
        plt.rcParams['font.family'] = 'Arial'
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

def extract_features(Wst, tlag, scale, times):
    p1_window = (120 <= tlag) & (tlag <= 130)
    n1_window = (175 <= tlag) & (tlag <= 185)
    p3_window = (400 <= tlag) & (tlag <= 500)

    if np.any(p1_window):
        p1_amplitude = np.max(Wst[:, p1_window])
        p1_idx = np.unravel_index(np.argmax(Wst[:, p1_window]), Wst[:, p1_window].shape)
        p1_latency = tlag[p1_window][p1_idx[1]]
    else:
        p1_amplitude = np.nan
        p1_latency = np.nan

    if np.any(n1_window):
        n1_amplitude = np.min(Wst[:, n1_window])
        n1_idx = np.unravel_index(np.argmin(Wst[:, n1_window]), Wst[:, n1_window].shape)
        n1_latency = tlag[n1_window][n1_idx[1]]
    else:
        n1_amplitude = np.nan
        n1_latency = np.nan

    if np.any(p3_window):
        p3_amplitude = np.max(Wst[:, p3_window])
        p3_idx = np.unravel_index(np.argmax(Wst[:, p3_window]), Wst[:, p3_window].shape)
        p3_latency = tlag[p3_window][p3_idx[1]]
    else:
        p3_amplitude = np.nan
        p3_latency = np.nan

    return p1_amplitude, n1_amplitude, p3_amplitude, p1_latency, n1_latency, p3_latency

df = pd.read_csv('channel_2.csv', header=None)
save_dir = ('../P3')


df.columns = ['subject_id'] + [f'trial_{i}' for i in range(1, df.shape[1])]


# df_mean = df.groupby('subject_id').mean().reset_index()
# print(df_mean.shape)

# Initialize a list to store the features
features = []
times = np.linspace(-0.4, 1.2, 409) * 1000

# Process each sample
for index, row in df.iterrows():
    if index % 100 == 0:
        print(f'Progress: {index}')

    # sample_id = row[0]
    signal = row[0:].values
    save_path = os.path.join(save_dir, f'vs_ot_real_{index + 1}.png')
    # 计算并绘制Wst矩阵
    Wst, tlag, scale = compute_Wst(signal, times, plot_on=True,save_path=save_path)
    # print(Wst.shape, tlag.shape, scale.shape)
    p1_amp, n1_amp, p3_amp, p1_lat, n1_lat, p3_lat = extract_features(Wst, tlag, scale, times)
    # print(p1_amp, n1_amp, p3_amp, p1_lat, n1_lat, p3_lat)
    features.append([ p1_amp, n1_amp, p3_amp, p1_lat, n1_lat, p3_lat]) #sample_id,

# 转换为DataFrame并保存结果
features_df = pd.DataFrame(features, columns=[ 'P1 Amplitude', 'N1 Amplitude', 'P3 Amplitude', 'P1 Latency', 'N1 Latency', 'P3 Latency']) #Sample ID',
features_df.to_csv('../P3.csv', index=False)
