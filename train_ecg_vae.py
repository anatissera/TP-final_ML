# train_ecg_vae.py
"""
Entrena un VAE convolucional sobre señales ECG normales (PTB-XL y Chapman).
Preprocesa (bandpass 0.5-40Hz, min-max a [-1,1]) y entrena sólo con registros normales.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
import wfdb
from wfdb import rdrecord

# ----- Preprocesamiento -----
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, fs, lowcut=0.5, highcut=40.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def normalize_minmax(data):
    # escala cada canal a [-1,1]
    minv = data.min(axis=1, keepdims=True)
    maxv = data.max(axis=1, keepdims=True)
    return 2*(data - minv)/(maxv - minv + 1e-8) - 1

# ----- Dataset ECG -----
class ECGDataset(Dataset):
    def __init__(self, ptb_dir, chap_dir, sample_length=5000):
        # Cargar metadata y filtrar normales
        ptb_csv = next((os.path.join(r,f) for r,_,fs in os.walk(ptb_dir) for f in fs if f=='ptbxl_database.csv'), None)
        df_ptb = pd.read_csv(ptb_csv)
        df_ptb = df_ptb[df_ptb['scp_codes'].str.contains('NORM', na=False)]
        # Chapman
        recs = []
        for root,_,files in os.walk(chap_dir):
            for f in files:
                if f.endswith('.hea') and 'Zone' not in f:
                    hea = os.path.join(root, f)
                    with open(hea) as h:
                        for line in h:
                            if line.startswith('#Dx:'):
                                dx = line.split()[1]
                                if dx=='426177001':
                                    recs.append(os.path.splitext(f)[0])
                                break
        self.ptb_meta = df_ptb.reset_index(drop=True)
        self.chap_meta = recs
        self.ptb_dir = ptb_dir
        self.chap_dir = chap_dir
        self.sample_length = sample_length
        # total items = sum
        self.total = len(self.ptb_meta) + len(self.chap_meta)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Decide si PTB o Chapman
        if idx < len(self.ptb_meta):
            row = self.ptb_meta.iloc[idx]
            fn = row['filename_lr']
            # buscar header
            hea_basename = os.path.basename(fn) + '.hea'
            hea_path = None
            for root,_,fs in os.walk(self.ptb_dir):
                if hea_basename in fs:
                    hea_path = os.path.join(root, hea_basename)
                    break
            rec_prefix = os.path.splitext(hea_path)[0]
            rec = rdrecord(rec_prefix)
            sig = rec.p_signal.T  # (12, N)
            fs = rec.fs
        else:
            rec_id = self.chap_meta[idx-len(self.ptb_meta)]
            mat_path = os.path.join(self.chap_dir, rec_id + '.mat')
            data = loadmat(mat_path)['val']
            sig = data
            fs = 500
        # recortar o pad a sample_length
        if sig.shape[1] >= self.sample_length:
            start = np.random.randint(0, sig.shape[1] - self.sample_length + 1)
            seg = sig[:, start:start+self.sample_length]
        else:
            # pad con ceros
            pad = np.zeros((sig.shape[0], self.sample_length))
            pad[:, :sig.shape[1]] = sig
            seg = pad
        # preproc
        seg = apply_bandpass(seg, fs)
        seg = normalize_minmax(seg)
        # convertir a tensor float32
        return torch.from_numpy(seg).float()

# ----- VAE Model -----
class ECG_VAE(nn.Module):
    def __init__(self, in_channels=12, z_dim=16, seq_len=5000):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128,7, stride=2, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
        )
        # calcular T'
        t_prime = seq_len // 8
        self.flatten = nn.Flatten()
        self.fc_mu  = nn.Linear(128 * t_prime, z_dim)
        self.fc_log = nn.Linear(128 * t_prime, z_dim)
        # Decoder
        self.fc_dec = nn.Linear(z_dim, 128 * t_prime)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 7, stride=2, padding=3, output_padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 7, stride=2, padding=3, output_padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, in_channels, 7, stride=2, padding=3, output_padding=1),
        )

    def reparam(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logv = self.fc_log(h)
        z = self.reparam(mu, logv)
        h2 = self.fc_dec(z).view(x.size(0), 128, -1)
        x_hat = self.dec(h2)
        return x_hat, mu, logv

# ----- Training Loop -----
def loss_fn(x, x_hat, mu, logv, beta=1.0):
    recon = torch.abs(x_hat - x).mean()
    kld = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
    return recon + beta * kld, recon, kld

if __name__ == '__main__':
    # Paths
    PTB_DIR = 'data/ptb-xl'
    CH_DIR  = 'data/ChapmanShaoxing'
    # Hyperparams
    batch_size = 16
    epochs = 50
    lr = 1e-3
    z_dim = 16
    seq_len = 5000

    # Dataset & Loader
    ds = ECGDataset(PTB_DIR, CH_DIR, sample_length=seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model, optim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = ECG_VAE(z_dim=z_dim, seq_len=seq_len).to(device)
    opt = optim.Adam(vae.parameters(), lr=lr)

    # Train
    for ep in range(1, epochs+1):
        vae.train()
        tot_loss = 0
        for batch in loader:
            x = batch.to(device)
            x_hat, mu, logv = vae(x)
            loss, recon, kld = loss_fn(x, x_hat, mu, logv)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()
            
        print(f"Epoch {ep}/{epochs} - Loss: {tot_loss/len(loader):.4f} (Recon {recon:.4f}, KLD {kld:.4f})")
    # Guardar
    torch.save(vae.state_dict(), 'ecg_vae.pth')
    print("Modelo guardado en ecg_vae.pth")
