# test_anomalies.py
"""
Evalúa el VAE entrenado sobre señales anómalas y normales de test.
Calcula MAE por muestra y grafica distribuciones.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
import wfdb
from wfdb import rdrecord
from torch.utils.data import Dataset, DataLoader

# Import preprocesamiento y modelo
from ecg_data_exploration_scaled_filtered import apply_bandpass
from VAE import normalize_zscore
from VAE import ECG_VAE

# ----- Dataset de anomalías y normales de test -----
class AnomalyTestDataset(Dataset):
    def __init__(self, ptb_dir, sample_length=5000, n_samples=100, normal=False):
        """
        Si normal=True, toma registros normales de PTB-XL;
        de lo contrario, toma registros anómalos.
        """
        # Cargar metadata
        csv = next((os.path.join(r, f) for r, _, fs in os.walk(ptb_dir) for f in fs if f=='ptbxl_database.csv'), None)
        import pandas as pd
        df = pd.read_csv(csv)
        if normal:
            df = df[df['scp_codes'].str.contains('NORM', na=False)]
        else:
            df = df[~df['scp_codes'].str.contains('NORM', na=False)]
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        self.records = df['filename_lr'].tolist()
        self.ptb_dir = ptb_dir
        self.sample_length = sample_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fn = self.records[idx]
        hea = os.path.basename(fn) + '.hea'
        # buscar header
        hea_path = None
        for root, _, files in os.walk(self.ptb_dir):
            if hea in files:
                hea_path = os.path.join(root, hea)
                break
        if hea_path is None:
            raise FileNotFoundError(f"Header no encontrado: {hea}")
        rec_prefix = os.path.splitext(hea_path)[0]
        rec = rdrecord(rec_prefix)
        sig = rec.p_signal.T  # (12, N)
        fs = rec.fs
        # segmentar al centro
        if sig.shape[1] >= self.sample_length:
            start = (sig.shape[1] - self.sample_length) // 2
            seg = sig[:, start:start+self.sample_length]
        else:
            pad = np.zeros((sig.shape[0], self.sample_length))
            pad[:, :sig.shape[1]] = sig
            seg = pad
        # preproc igual que en train
        seg = apply_bandpass(seg, fs)
        seg = normalize_zscore(seg)
        return torch.from_numpy(seg).float()

# ----- Funciones de test -----
def compute_mae(loader, model, device):
    maes = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            xh, _, _ = model(xb)
            batch_mae = torch.mean(torch.abs(xh - xb), dim=(1,2)).cpu().numpy()
            maes.extend(batch_mae.tolist())
    return np.array(maes)

if __name__ == '__main__':
    PTB_DIR = 'data/ptb-xl'
    model_path = 'ecg_vae_best.pth'  # usa el mejor guardado en validación
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar modelo
    vae = ECG_VAE(z_dim=16, seq_len=5000).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()

    # Crear datasets y loaders
    N = 100
    ds_norm = AnomalyTestDataset(PTB_DIR, n_samples=N, normal=True)
    ds_ano  = AnomalyTestDataset(PTB_DIR, n_samples=N, normal=False)
    loader_norm = DataLoader(ds_norm, batch_size=16, shuffle=False)
    loader_ano  = DataLoader(ds_ano,  batch_size=16, shuffle=False)

    # Calcular MAE
    mae_norm = compute_mae(loader_norm, vae, device)
    mae_ano  = compute_mae(loader_ano,  vae, device)
    print(f"MAE Normales: mean={mae_norm.mean():.4f}  std={mae_norm.std():.4f}")
    print(f"MAE Anómalos:  mean={mae_ano.mean():.4f}   std={mae_ano.std():.4f}")

    # Graficar histogramas
    plt.hist(mae_norm, bins=30, alpha=0.7, label='Normales')
    plt.hist(mae_ano,  bins=30, alpha=0.7, label='Anómalos')
    plt.xlabel('MAE por muestra')
    plt.ylabel('Frecuencia')
    plt.title('Distribución MAE: Normales vs Anómalos')
    plt.legend()
    plt.grid(True)
    plt.show()
