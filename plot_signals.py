
import matplotlib.font_manager as fm


def plot_losses(d_loss_epoch, g_loss_epoch, plot_fold, save_path):
    plt.figure(figsize=(3, 3))

    if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
        mpl.rcParams['font.family'] = 'Arial'
    elif 'Nimbus Sans' in [f.name for f in fm.fontManager.ttflist]:
        mpl.rcParams['font.family'] = 'Nimbus Sans'
    else:

        mpl.rcParams['font.family'] = 'DejaVu Sans'

    mpl.rcParams['font.size'] = 12


    plt.plot(d_loss_epoch, label='D Loss', color='firebrick', linewidth=1, linestyle='-', alpha=0.7)
    plt.plot(g_loss_epoch, label='G Loss', color='navy', linewidth=1, linestyle='-', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')
    plt.legend(loc='best', frameon=False)

    plt.tight_layout()

    save_path = os.path.join(save_path, plot_fold)
    plot_name = "loss.tiff"

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    plt.savefig(os.path.join(save_path, plot_name),format='png',dpi=600)

    plt.close()

def plot_losses_smooth(d_loss_epoch, g_loss_epoch, plot_fold, save_path):
    plt.figure(figsize=(3, 3))

    if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
        mpl.rcParams['font.family'] = 'Arial'
    elif 'Nimbus Sans' in [f.name for f in fm.fontManager.ttflist]:
        mpl.rcParams['font.family'] = 'Nimbus Sans'
    else:

        mpl.rcParams['font.family'] = 'DejaVu Sans'

    mpl.rcParams['font.size'] = 12


    d_loss_epoch = np.array(d_loss_epoch)
    g_loss_epoch = np.array(g_loss_epoch)

    max_d_loss_finite = np.nanmax(np.where(np.isfinite(d_loss_epoch), d_loss_epoch, -np.inf))
    max_g_loss_finite = np.nanmax(np.where(np.isfinite(g_loss_epoch), g_loss_epoch, -np.inf))
    d_loss_epoch_clean = np.nan_to_num(d_loss_epoch, nan=0.0, posinf=max_d_loss_finite, neginf=-max_d_loss_finite)
    g_loss_epoch_clean = np.nan_to_num(g_loss_epoch, nan=0.0, posinf=max_g_loss_finite, neginf=-max_g_loss_finite)

    epochs = np.arange(len(d_loss_epoch_clean))
    epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)
    d_loss_epoch_spline = make_interp_spline(epochs, d_loss_epoch_clean, k=3)(epochs_smooth)
    g_loss_epoch_spline = make_interp_spline(epochs, g_loss_epoch_clean, k=3)(epochs_smooth)

    plt.plot(epochs_smooth, d_loss_epoch_spline, label='D Loss', color='firebrick', linewidth=1, alpha=0.7)
    plt.plot(epochs_smooth, g_loss_epoch_spline, label='G Loss', color='navy', linewidth=1, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')
    plt.legend(loc='best', frameon=False)

    plt.tight_layout()

    save_path = os.path.join(save_path, plot_fold)
    plot_name = "loss_smooth.tiff"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, plot_name), format='png',dpi=600)


    plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import make_interp_spline

def plot_losses_G_smooth(g_loss_epoch, wd_gp_loss_epoch, psd_loss_epoch, time_loss_epoch, plot_fold, save_path):
    plt.figure(figsize=(3, 3))

    if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
        mpl.rcParams['font.family'] = 'Arial'
    elif 'Nimbus Sans' in [f.name for f in fm.fontManager.ttflist]:
        mpl.rcParams['font.family'] = 'Nimbus Sans'
    else:
        mpl.rcParams['font.family'] = 'DejaVu Sans'

    mpl.rcParams['font.size'] = 12


    g_loss_epoch = np.array(g_loss_epoch)
    wd_gp_loss_epoch = np.array(wd_gp_loss_epoch)
    psd_loss_epoch = np.array(psd_loss_epoch)
    time_loss_epoch = np.array(time_loss_epoch)


    def clean_loss_epoch(loss_epoch):
        max_loss_finite = np.nanmax(np.where(np.isfinite(loss_epoch), loss_epoch, -np.inf))
        return np.nan_to_num(loss_epoch, nan=0.0, posinf=max_loss_finite, neginf=-max_loss_finite)

    g_loss_epoch_clean = clean_loss_epoch(g_loss_epoch)
    wd_gp_loss_epoch_clean = clean_loss_epoch(wd_gp_loss_epoch)
    psd_loss_epoch_clean = clean_loss_epoch(psd_loss_epoch)
    time_loss_epoch_clean = clean_loss_epoch(time_loss_epoch)


    epochs = np.arange(len(g_loss_epoch_clean))
    epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)

    def smooth_loss_epoch(loss_epoch_clean):
        return make_interp_spline(epochs, loss_epoch_clean, k=3)(epochs_smooth)

    g_loss_epoch_spline = smooth_loss_epoch(g_loss_epoch_clean)
    wd_gp_loss_epoch_spline = smooth_loss_epoch(wd_gp_loss_epoch_clean)
    psd_loss_epoch_spline = smooth_loss_epoch(psd_loss_epoch_clean)
    time_loss_epoch_spline = smooth_loss_epoch(time_loss_epoch_clean)


    plt.plot(epochs_smooth, g_loss_epoch_spline, label='G Loss', color='navy', linewidth=1, alpha=0.7)
    plt.plot(epochs_smooth, wd_gp_loss_epoch_spline, label='WD(GP) Loss', color='red', linewidth=1, alpha=0.7)
    plt.plot(epochs_smooth, psd_loss_epoch_spline, label='PSD Loss', color='green', linewidth=1, alpha=0.7)
    plt.plot(epochs_smooth, time_loss_epoch_spline, label='Time Loss', color='purple', linewidth=1, alpha=0.7)

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')
    plt.legend(loc='best', frameon=False,fontsize = 8)
    plt.tight_layout()


    save_path = os.path.join(save_path, plot_fold)
    plot_name = "loss_smooth_G.tiff"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, plot_name), format='png',dpi=600)
    plt.close()



