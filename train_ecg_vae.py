# train_ecg_cvae_jang2021.py
"""
Implementación inspirada en Jang et al. (2021) PLOS ONE:
"Unsupervised feature learning for ECG data using the convolutional variational autoencoder".

Características clave:
- Datos: Use solo lead II de PTB-XL y Chapman, resampleado a 250 Hz y Z-score.
- Arquitectura:
  * Encoder: 9 bloques de Conv1d (6 capas con kernel=19, luego 3 capas con kernel=9), cada una con BatchNorm y LeakyReLU.
  * Latente: dimensión de 60 (mu y logvar de 60).
  * Decoder: simétrico con 2 capas FC + 2 capas ConvTranspose1d para reconstruir 2048 muestras (8.2 s a 250 Hz).
- Preprocesamiento:
  * Butterworth pasa-altos 0.5 Hz para remover línea base.
  * Resampleo a 250 Hz.
  * Cada señal centrada en segmento de longitud fija de 8.2 s (2048 muestras).
  * Normalización Z-score.
- Entrenamiento:
  * Optimizer: Adam, lr=1e-4.
  * Early stopping: detener si no mejora validación tras 5 épocas, máximo 50 épocas.
  * Batch size: 32.
  * Pérdida: ELBO = MSE + beta * KLD, con beta=1 (VAE estándar) o configurable.

Guarda el mejor modelo y scores de validación/test para análisis.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
import wfdb
from wfdb import rdrecord
from scipy.signal import resample

# ----- Preprocesamiento -----
def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='high')
    return b, a

def apply_preproc(sig, fs_src, target_fs=250, length=2048):
    # High-pass para remover baseline
    b, a = butter_highpass(0.5, fs_src)
    sig = filtfilt(b, a, sig, axis=1)
    # Resample a 250 Hz
    sig = resample(sig, int(sig.shape[1] * target_fs / fs_src), axis=1)
    # Centrar en medio y recortar/pad a length
    N = sig.shape[1]
    if N >= length:
        start = (N - length)//2
        seg = sig[:, start:start+length]
    else:
        seg = np.zeros((sig.shape[0], length))
        seg[:, (length-N)//2:(length-N)//2+N] = sig
    # Z-score por canal
    mu = seg.mean(axis=1, keepdims=True)
    std = seg.std(axis=1, keepdims=True) + 1e-6
    return (seg - mu) / std

# ----- Dataset -----
class ECGLeadIIDataset(Dataset):
    def __init__(self, ptb_dir, chap_dir, length=2048, include_ano=False):
        self.records = []
        # PTB-XL
        csv = next((os.path.join(r,f) for r,_,fs in os.walk(ptb_dir) for f in fs if f=='ptbxl_database.csv'), None)
        df = np.loadtxt(csv, delimiter=',', dtype=str, skiprows=1)  # adapt to your CSV structure
        for row in df:
            scp = row[3]
            idf = row[1]
            label = 1 if 'NORM' in scp else 0
            if label==1 or include_ano:
                self.records.append(('ptb', idf, label))
        # Chapman
        for root,_,files in os.walk(chap_dir):
            for f in files:
                if f.endswith('.hea') and 'Zone' not in f:
                    with open(os.path.join(root,f)) as h:
                        lines = h.readlines()
                    dx = next((l.split()[1] for l in lines if l.startswith('#Dx:')), '0')
                    label = 1 if dx=='426177001' else 0
                    if label==1 or include_ano:
                        rec_id = os.path.splitext(f)[0]
                        self.records.append(('chap', rec_id, label))
        self.ptb_dir, self.chap_dir = ptb_dir, chap_dir
        self.length = length

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        src, idf, label = self.records[idx]
        if src=='ptb':
            hea = os.path.basename(idf)+'.hea'
            path = next((os.path.join(r,hea) for r,_,_ in os.walk(self.ptb_dir) if hea in os.listdir(r)), None)
            rec = rdrecord(path.replace('.hea',''))
            sig = rec.p_signal.T; fs = rec.fs
        else:
            mat = loadmat(os.path.join(self.chap_dir, idf+'.mat'))
            sig = mat['val']; fs = 500
        seg = apply_preproc(sig, fs, target_fs=250, length=self.length)
        return torch.tensor(seg, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ----- cVAE Jang2021 -----
class cVAE1D(nn.Module):
    def __init__(self, in_ch=1, z_dim=60, seq_len=2048):
        super().__init__()
        layers = []
        ch = in_ch
        # 6 capas de kernel 19
        for _ in range(6):
            layers += [nn.Conv1d(ch, ch, kernel_size=19, stride=2, padding=9), nn.BatchNorm1d(ch), nn.LeakyReLU()]
        # 3 capas de kernel 9
        for _ in range(3):
            layers += [nn.Conv1d(ch, ch, kernel_size=9, stride=2, padding=4), nn.BatchNorm1d(ch), nn.LeakyReLU()]
        self.enc = nn.Sequential(*layers)
        t_p = seq_len // (2**9)  # reduce factor 2^9
        self.fc_mu  = nn.Linear(ch * t_p, z_dim)
        self.fc_log = nn.Linear(ch * t_p, z_dim)
        # decoder
        self.fc_dec = nn.Linear(z_dim, ch * t_p)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(ch, ch, kernel_size=9, stride=2, padding=4, output_padding=1), nn.BatchNorm1d(ch), nn.LeakyReLU(),
            nn.ConvTranspose1d(ch, in_ch, kernel_size=19, stride=2, padding=9, output_padding=1), nn.Tanh(),
        )

    def reparam(self, mu, logv):
        std = (0.5 * logv).exp(); eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x.unsqueeze(1))  # (B,1,L) -> channels
        h = h.flatten(1)
        mu = self.fc_mu(h); logv = self.fc_log(h)
        z = self.reparam(mu, logv)
        h2 = self.fc_dec(z).view(x.size(0), 1, -1)
        x_hat = self.dec(h2).squeeze(1)
        return x_hat, mu, logv

# ----- Entrenamiento con EarlyStopping -----
def train():
    PTB, CH = 'data/ptb-xl', 'data/ChapmanShaoxing'
    ds = ECGLeadIIDataset(PTB, CH, include_ano=True)
    n = len(ds)
    n_test, n_val = int(0.1*n), int(0.1*n)
    ds_tr, ds_val, ds_te = random_split(ds, [n-n_val-n_test, n_val, n_test])
    lt_tr = DataLoader(ds_tr, 32, shuffle=True)
    lt_val = DataLoader(ds_val, 32)
    lt_te = DataLoader(ds_te, 32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cVAE1D(in_ch=1, z_dim=60, seq_len=2048).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    best_val = float('inf'); no_imp = 0

    for ep in range(1, 51):
        model.train(); train_loss=0
        for xb, _ in lt_tr:
            xb = xb.to(device)
            xh, mu, logv = model(xb)
            recon = nn.functional.mse_loss(xh, xb, reduction='mean')
            kld   = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
            loss = recon + kld
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
        train_loss /= len(lt_tr)
        # valid
        model.eval(); val_loss=0
        with torch.no_grad():
            for xb,_ in lt_val:
                xb = xb.to(device)
                xh, mu, logv = model(xb)
                recon = nn.functional.mse_loss(xh, xb, reduction='mean')
                kld   = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
                val_loss += (recon + kld).item()
        val_loss /= len(lt_val)
        print(f"Epoch {ep}  Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        # early stop
        if val_loss < best_val:
            best_val = val_loss; no_imp = 0
            torch.save(model.state_dict(), 'cvae_jang_best.pth')
        else:
            no_imp += 1
            if no_imp >=5:
                print("No improvement for 5 epochs, stopping.")
                break

    # test final
    model.load_state_dict(torch.load('cvae_jang_best.pth'))
    recons, klds = [], []
    model.eval()
    with torch.no_grad():
        for xb,_ in lt_te:
            xb = xb.to(device)
            xh, mu, logv = model(xb)
            recons.append(nn.functional.mse_loss(xh, xb, reduction='none').mean(dim=1).cpu().numpy())
            klds.append((-0.5*(1+logv-logv.exp()-mu.pow(2))).sum(dim=1).cpu().numpy())
    np.save('test_scores.npy', np.stack([np.concatenate(recons), np.concatenate(klds)],1))
    print("Entrenamiento y evaluación completos.")

if __name__=='__main__':
    train()
