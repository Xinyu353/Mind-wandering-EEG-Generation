import torch
import torch.nn as nn
import torch.fft


class FrequencyFilterLayer(nn.Module):
    def __init__(self, cutoff_frequency, fs):
        super(FrequencyFilterLayer, self).__init__()
        self.cutoff_frequency = cutoff_frequency
        self.fs = fs

    def forward(self, x):
        # x shape: (batch_size, seq_num, num_features)
        batch_size, seq_num = x.shape

        # Apply FFT along the sequence dimension
        fft = torch.fft.rfft(x, dim=1)

        # Calculate the frequency components
        freqs = torch.fft.rfftfreq(seq_num, d=1. / self.fs)

        # Create a frequency mask
        mask = (freqs >= 0.5) & (freqs <= self.cutoff_frequency)
        mask = mask.to(x.device)

        # Apply the mask to each feature
        mask = mask.unsqueeze(0) # Shape: (1, seq_num//2+1, 1)
        fft = fft * mask

        # Apply inverse FFT to get the filtered signal
        filtered_signal = torch.fft.irfft(fft, n=seq_num, dim=1)

        return filtered_signal
