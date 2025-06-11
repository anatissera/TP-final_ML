import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from ecg_data_exploration_scaled_filtered import apply_bandpass, normalize_signal
from scipy.io import loadmat
import wfdb
from wfdb import rdrecord
import pandas as pd
import matplotlib.pyplot as plt

# Cargar modelo VAE (usá misma clase ECG_VAE de entrenamiento)
from train_ecg_vae import ECG_VAE

# Reutilizá tu preprocesamiento y funciones
def load_ptbxl_anomalous(ptb_dir, sample_length=5000, n_samples=100):
    csv_path = next((os.path.join(r, f) for r, _, fs in os.walk(ptb_dir)
                     for f in fs if f == 'ptbxl_database.csv'), None)
    df = pd.read_csv(csv_path)
    df = df[~df['scp_codes'].str.contains('NORM', na=False)].sample(n=n_samples).reset_index(drop=True)
    signals = []
    for _, row in df.iterrows():
        fn = row['filename_lr']
        hea_basename = os.path.basename(fn) + '.hea'
        hea_path = None
        for root, _, fs in os.walk(ptb_dir):
            if hea_basename in fs:
                hea_path = os.path.join(root, hea_basename)
                break
        if hea_path is None:
            continue
        rec_prefix = os.path.splitext(hea_path)[0]
        rec = rdrecord(rec_prefix)
        sig = rec.p_signal.T
        fs = rec.fs
        if sig.shape[1] >= sample_length:
            start = np.random.randint(0, sig.shape[1] - sample_length + 1)
            seg = sig[:, start:start+sample_length]
        else:
            seg = np.pad(sig, ((0, 0), (0, sample_length - sig.shape[1])))
        seg = apply_bandpass(seg, fs)
        seg = normalize_signal(seg, method='minmax')
        signals.append(seg)
    return torch.tensor(np.stack(signals)).float()

# ===== MAIN =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = ECG_VAE(z_dim=16, seq_len=5000).to(device)
vae.load_state_dict(torch.load('ecg_vae.pth', map_location=device))
vae.eval()

# Datos anómalos
ptb_dir = 'data/ptb-xl'
test_data = load_ptbxl_anomalous(ptb_dir, n_samples=100)
test_data = test_data.to(device)

# Evaluar MAE
with torch.no_grad():
    recon, _, _ = vae(test_data)
    mae = torch.mean(torch.abs(recon - test_data), dim=(1, 2))  # por muestra
    print(f'MAE promedio en test anómalo: {mae.mean().item():.4f}')
    print(f'MAE mínimo: {mae.min().item():.4f}, máximo: {mae.max().item():.4f}')

# (opcional) graficar original vs reconstruido
idx = mae.argmax().item()  # elegir el más fallido
orig = test_data[idx].cpu().numpy()
reco = recon[idx].cpu().numpy()
plt.plot(orig[0], label='original')
plt.plot(reco[0], label='reconstruido')
plt.title(f'Lead 0 - MAE={mae[idx].item():.4f}')
plt.legend()
plt.grid(True)
plt.show()
