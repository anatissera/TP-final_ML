# train_ecg_cVAE.py
"""
Entrena un cVAE convolucional sobre señales ECG (PTB-XL y Chapman).
Conditioning binaria: 1 = normal, 0 = anomalous.
- Reconstrucción con MSE.
- Regularización KL con factor β.
- Grad clipping.
- Split train/val/test incluyendo normales y anomalías.
- Guarda mejor modelo y scores de recon/KL para análisis.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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

def normalize_zscore(data):
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)
    return (data - mu) / (sigma + 1e-6)

# ----- Dataset ECG con condition -----
class ECGDataset(Dataset):
    def __init__(self, ptb_dir, chap_dir, sample_length=5000, include_anomalies=False):
        self.samples = []  # list of (src, identifier, label)
        # Normales label=1
        csv = next((os.path.join(r,f) for r,_,fs in os.walk(ptb_dir) for f in fs if f=='ptbxl_database.csv'), None)
        df = pd.read_csv(csv)
        df_norm = df[df['scp_codes'].str.contains('NORM', na=False)].reset_index(drop=True)
        for _, row in df_norm.iterrows():
            self.samples.append(('ptb', row['filename_lr'], 1))
        # Anómalos label=0 si se incluyen
        if include_anomalies:
            df_ano = df[~df['scp_codes'].str.contains('NORM', na=False)].reset_index(drop=True)
            for _, row in df_ano.iterrows():
                self.samples.append(('ptb', row['filename_lr'], 0))
        # Chapman solo normales (para cVAE podemos incluir anomalías si tenemos etiqueta)
        for root,_,files in os.walk(chap_dir):
            for f in files:
                if f.endswith('.hea') and 'Zone' not in f:
                    with open(os.path.join(root,f)) as h:
                        for line in h:
                            if line.startswith('#Dx:'):
                                dx = line.split()[1]
                                label = 1 if dx=='426177001' else 0
                                if label==1 or include_anomalies:
                                    rec_id = os.path.splitext(f)[0]
                                    self.samples.append(('chap', rec_id, label))
                                break
        self.ptb_dir = ptb_dir
        self.chap_dir = chap_dir
        self.sample_length = sample_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, idf, label = self.samples[idx]
        # cargar señal
        if src=='ptb':
            hea = os.path.basename(idf) + '.hea'
            path = next((os.path.join(r,hea) for r,_,fs in os.walk(self.ptb_dir) if hea in fs), None)
            rec = rdrecord(os.path.splitext(path)[0])
            sig, fs = rec.p_signal.T, rec.fs
        else:
            mat = loadmat(os.path.join(self.chap_dir, idf+'.mat'))
            sig, fs = mat['val'], 500
        # segmentar
        if sig.shape[1] >= self.sample_length:
            i = np.random.randint(0, sig.shape[1] - self.sample_length + 1)
            seg = sig[:, i:i+self.sample_length]
        else:
            pad = np.zeros((sig.shape[0], self.sample_length))
            pad[:, :sig.shape[1]] = sig
            seg = pad
        # preproc
        seg = apply_bandpass(seg, fs)
        seg = normalize_zscore(seg)
        # condition tensor
        c = torch.tensor([label], dtype=torch.float32)
        return torch.from_numpy(seg).float(), c

# ----- cVAE Model -----
class ECG_cVAE(nn.Module):
    def __init__(self, in_ch=12, cond_dim=1, z_dim=16, seq_len=5000):
        super().__init__()
        # encoder conv
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch,32,7,2,3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,64,7,2,3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,7,2,3), nn.BatchNorm1d(128), nn.ReLU(),
        )
        t_p = seq_len // 8
        # mu/logvar con condicionamiento
        self.fc_mu  = nn.Linear(128*t_p + cond_dim, z_dim)
        self.fc_log = nn.Linear(128*t_p + cond_dim, z_dim)
        # decoder inicial
        self.fc_dec = nn.Linear(z_dim + cond_dim, 128*t_p)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(128,64,7,2,3,1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64,32,7,2,3,1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32,in_ch,7,2,3,1), nn.Tanh(),
        )

    def reparam(self, mu, logv):
        std=(0.5*logv).exp(); eps=torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, c):
        # x: (B,12,N), c: (B,1)
        h = self.enc(x).flatten(1)
        h_c = torch.cat([h, c], dim=1)
        mu = self.fc_mu(h_c)
        logv = self.fc_log(h_c)
        z = self.reparam(mu, logv)
        z_c = torch.cat([z, c], dim=1)
        h2 = self.fc_dec(z_c).view(x.size(0),128,-1)
        x_hat = self.dec(h2)
        return x_hat, mu, logv

# ----- Pérdida β-VAE -----
def loss_fn(x, xh, mu, logv, beta):
    recon = nn.functional.mse_loss(xh, x, reduction='mean')
    kld   = -0.5*torch.mean(1 + logv - mu.pow(2) - logv.exp())
    return recon + beta*kld, recon, kld

if __name__=='__main__':
    PTB_DIR='data/ptb-xl'; CHAP_DIR='data/ChapmanShaoxing'
    bs, seq, zdim, cond_dim = 16, 5000, 16, 1
    epochs, lr, beta = 20, 1e-4, 5.0

    # Dataset con normales + anomalías para cVAE
    ds = ECGDataset(PTB_DIR, CHAP_DIR, sample_length=seq, include_anomalies=True)
    n = len(ds)
    n_test = int(0.1*n); n_val=int(0.1*n); n_train=n-n_val-n_test
    ds_tr, ds_val, ds_te = random_split(ds, [n_train,n_val,n_test])
    lt_tr = DataLoader(ds_tr, bs, shuffle=True, num_workers=4)
    lt_val= DataLoader(ds_val, bs, shuffle=False, num_workers=2)
    lt_te = DataLoader(ds_te, bs, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cvae = ECG_cVAE(in_ch=12, cond_dim=cond_dim, z_dim=zdim, seq_len=seq).to(device)
    opt = optim.Adam(cvae.parameters(), lr=lr)

    best_val = float('inf')
    for ep in range(1, epochs+1):
        cvae.train(); train_loss=0.0
        for xb, c in lt_tr:
            xb, c = xb.to(device), c.to(device)
            xh, mu, logv = cvae(xb, c)
            loss, recon, kld = loss_fn(xb, xh, mu, logv, beta)
            opt.zero_grad(); loss.backward();
            nn.utils.clip_grad_norm_(cvae.parameters(),1.0); opt.step()
            train_loss += loss.item()
        train_loss /= len(lt_tr)
        # valid
        cvae.eval(); val_loss, recs, klds = 0.0, [], []
        with torch.no_grad():
            for xb, c in lt_val:
                xb, c = xb.to(device), c.to(device)
                xh, mu, logv = cvae(xb, c)
                loss, recon, kld = loss_fn(xb, xh, mu, logv, beta)
                val_loss += loss.item(); recs.append(recon.item()); klds.append(kld.item())
        val_loss /= len(lt_val)
        print(f"Epoch {ep}/{epochs}  Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        if val_loss < best_val:
            best_val=val_loss
            torch.save(cvae.state_dict(),'ecg_cvae_best.pth')
            np.save('val_scores.npy', np.stack([recs,klds],1))

    print("Entrenamiento completo.")
    # Test final
    cvae.load_state_dict(torch.load('ecg_cvae_best.pth',map_location=device))
    all_recs, all_klds = [], []
    with torch.no_grad():
        for xb, c in lt_te:
            xb, c = xb.to(device), c.to(device)
            xh, mu, logv = cvae(xb, c)
            rec = nn.functional.mse_loss(xh, xb, reduction='none').mean(dim=(1,2))
            kld = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=1)
            all_recs.extend(rec.cpu().numpy()); all_klds.extend(kld.cpu().numpy())
    np.save('test_scores.npy', np.stack([all_recs,all_klds],1))
    print(f"Test final MSE mean: {np.mean(all_recs):.4f} | KLD mean: {np.mean(all_klds):.4f}")
