"""
model.py
Define el CVAE 1D con cálculo dinámico del tamaño de flatten.
"""
import torch
import torch.nn as nn

# class CVAE(nn.Module):
#     def __init__(self, in_channels=1, latent_dim=60, input_length=2048):
#         super().__init__()
#         # Encoder: 9 capas Conv1d
#         kernels = [19]*6 + [9]*3
#         layers = []
#         ch = in_channels
#         for k_i, k in enumerate(kernels):
#             out_ch = 16 if k_i < 6 else 32
#             layers += [
#                 nn.Conv1d(ch, out_ch, k, stride=2, padding=k//2),
#                 nn.BatchNorm1d(out_ch),
#                 nn.LeakyReLU()
#             ]
#             ch = out_ch
#         self.encoder = nn.Sequential(*layers)

#         # Calcular tamaño flattened
#         length = input_length
#         for k in kernels:
#             length = (length + 2*(k//2) - (k-1) - 1)//2 + 1
#         self.flat_size = ch * length

#         # Proyecciones latentes
#         self.fc_mu     = nn.Linear(self.flat_size, latent_dim)
#         self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

#         # Decoder
#         self.fc_dec = nn.Linear(latent_dim, self.flat_size)
#         rev_layers = []
#         ch_dec = ch
#         for idx, k in reversed(list(enumerate(kernels))):
#             in_ch = ch_dec
#             out_ch = in_channels if idx==0 else (16 if idx<6 else 32)
#             rev_layers += [
#                 nn.ConvTranspose1d(in_ch, out_ch, k, stride=2, padding=k//2, output_padding=1),
#                 nn.BatchNorm1d(out_ch),
#                 nn.LeakyReLU()
#             ]
#             ch_dec = out_ch
#         self.decoder = nn.Sequential(*rev_layers)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         return mu + torch.randn_like(std)*std

#     def forward(self, x):
#         enc = self.encoder(x)
#         flat = enc.view(x.size(0), -1)
#         mu, logvar = self.fc_mu(flat), self.fc_logvar(flat)
#         z = self.reparameterize(mu, logvar)
#         dec_in = self.fc_dec(z).view_as(enc)
#         recon = self.decoder(dec_in)
#         return recon, mu, logvar


class CVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=60, input_length=2048):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=19, stride=2, padding=9), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3), nn.ReLU(),
        )
        # Calcular dinámicamente flat_size con tensor dummy
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            enc_out = self.encoder(dummy)
            flat_size = enc_out.numel()  # 1 * channels * length

        # Flatten y proyecciones latentes
        self.flatten = nn.Flatten()
        self.fc_mu     = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flat_size)
        # Reconstrucción a partir de encoder dummy shape
        channels, length = enc_out.shape[1], enc_out.shape[2]
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (channels, length)),
            nn.ConvTranspose1d(channels, 64, kernel_size=7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=11, stride=2, padding=5, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, in_channels, kernel_size=19, stride=2, padding=9, output_padding=1), nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, epoch=None):
        enc = self.encoder(x)
        flat = self.flatten(enc)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)
        dec_in = self.fc_dec(z)
        recon = self.decoder(dec_in)
        return recon, mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld
