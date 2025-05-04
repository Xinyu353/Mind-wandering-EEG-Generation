from torch import nn
from filter import FrequencyFilterLayer


# Generator Network using 1D CNNs
class Generator(nn.Module):
    def __init__(self, latent_dim, eeg_length=409,  cutoff_frequency=40, fs=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.eeg_length = eeg_length

        self.model = nn.Sequential(
            # Layer 1: Input (1 → 16)
            nn.ConvTranspose1d(in_channels=latent_dim, out_channels=256, kernel_size=8, stride=1, padding=0),  # (bs, 256, 8)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: (16 → 32)
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=3),  # (bs, 128, 13)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: (32 → 64)
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3),  # (bs, 64, 26)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: (64 → 128)
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3),  # (bs, 32, 52)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: (128 → 256)
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=2, padding=3),  # (bs, 16, 104)
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Final Layer: (256 → 409)
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=2, padding=3),  # (bs, 1, 205)
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final Layer: (256 → 409)
            nn.ConvTranspose1d(8, 1, kernel_size=5, stride=2, padding=2),  # (bs, 1, 409)
            nn.Tanh()
        )
        self.filter_layer = FrequencyFilterLayer(cutoff_frequency=cutoff_frequency, fs=fs)

    def forward(self, z):
        z = z.unsqueeze(2)
        # print(f"Input z: {z.size()}")

        for layer in self.model:
            z = layer(z)
            # print(f"After {layer.__class__.__name__}: {z.size()}")

        z = z.squeeze(1)
        z = self.filter_layer(z)
        # print("Output z: ", output.size())


        return z


class Discriminator(nn.Module):
    def __init__(self, input_dim=409):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: Extract fine-grained features
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),  # (batch_size, 16, 409)
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: Downsample by a factor of 2
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # (batch_size, 32, 205)
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: Further downsample
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),  # (batch_size, 64, 103)
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: Further downsample
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),  # (batch_size, 128, 52)
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: Further downsample
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),  # (batch_size, 256, 26)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 26, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.model(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
