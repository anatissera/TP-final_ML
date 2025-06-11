"""
model.py
Define el CVAE 1D con cálculo dinámico del tamaño de flatten.
"""
import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, in_channels=12, latent_dim=60, input_length=2048):
        super().__init__()
        # Encoder: 9 capas Conv1d
        kernels = [19]*6 + [9]*3
        layers = []
        ch = in_channels
        for k_i, k in enumerate(kernels):
            out_ch = 16 if k_i < 6 else 32
            layers += [
                nn.Conv1d(ch, out_ch, k, stride=2, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU()
            ]
            ch = out_ch
        self.encoder = nn.Sequential(*layers)

        # Calcular tamaño flattened
        length = input_length
        for k in kernels:
            length = (length + 2*(k//2) - (k-1) - 1)//2 + 1
        self.flat_size = ch * length

        # Proyecciones latentes
        self.fc_mu     = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.flat_size)
        rev_layers = []
        ch_dec = ch
        for idx, k in reversed(list(enumerate(kernels))):
            in_ch = ch_dec
            out_ch = in_channels if idx==0 else (16 if idx<6 else 32)
            rev_layers += [
                nn.ConvTranspose1d(in_ch, out_ch, k, stride=2, padding=k//2, output_padding=1),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU()
            ]
            ch_dec = out_ch
        self.decoder = nn.Sequential(*rev_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + torch.randn_like(std)*std

    def forward(self, x):
        enc = self.encoder(x)
        flat = enc.view(x.size(0), -1)
        mu, logvar = self.fc_mu(flat), self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)
        dec_in = self.fc_dec(z).view_as(enc)
        recon = self.decoder(dec_in)
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld
