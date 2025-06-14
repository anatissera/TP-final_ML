import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm
from scipy.io import loadmat

# Importa funciones de preprocess y test
from preprocess import (
    load_ptbxl, load_chapman, load_mitbih,
    apply_highpass, resample_signal, zscore_normalize, segment_signal, find_file
)
from model import CVAE, loss_function
from test_functions import compute_reconstruction_error, save_metrics
from utils.metrics import compute_roc, compute_pr, optimal_threshold

import os

def find_data_subfolder(subfolder_name, start_path='.'):
    current_path = os.path.abspath(start_path)
    while True:
        candidate = os.path.join(current_path, 'data', subfolder_name)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current_path)
        if parent == current_path:
            break
        current_path = parent
    return None

# Ahora buscás las rutas relativas automáticamente:
PTB_DIR = find_data_subfolder('ptb-xl/1.0.3')
CHAP_DIR = find_data_subfolder('ChapmanShaoxing')
MIT_DIR = find_data_subfolder('mitdb')


EPOCHS      = 18
BATCH       = 32
LR          = 1e-3
LATENT      = 60
MODEL_OUT   = 'best_cvae_leadII.pth'
METRICS_OUT = 'metrics_combined_lead_II.json'

# Entrenamiento y validación de CVAE

def preprocess_and_segment(sig, fs):
    sig = apply_highpass(sig, fs)
    sig = resample_signal(sig, fs)
    sig = zscore_normalize(sig)
    sig = segment_signal(sig)
    return sig

# Carga solo sanos (PTB+Chapman)
def load_normals():
    normals_ptb  = load_ptbxl(PTB_DIR, healthy_only=True)
    normals_chap = load_chapman(CHAP_DIR, healthy_only=True)
    return np.concatenate([normals_ptb, normals_chap], axis=0)

# Entrena el modelo CVAE
def train_cvae(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Split
    n_val = int(0.2 * len(data))
    train_set, val_set = random_split(data, [len(data)-n_val, n_val])
    train_loader = DataLoader(TensorDataset(torch.tensor(train_set)), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(val_set)), batch_size=BATCH)

    model = CVAE(in_channels=1, latent_dim=LATENT, input_length=2048).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mae = float('inf')

    for ep in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        for (x,) in tqdm(train_loader, desc=f"Train Ep{ep}/{EPOCHS}", leave=False):
            x = x.float().to(device)
            recon, mu, logvar = model(x)
            # aquí podrías usar beta_schedule(ep) si lo implementas
            loss = loss_function(recon, x, mu, logvar)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validación MAE
        model.eval()
        mae_vals = []
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.float().to(device)
                recon, _, _ = model(x)
                mae_vals.append(torch.mean(torch.abs(recon - x)).item())
        val_mae = np.mean(mae_vals)
        print(f"Epoch {ep}: Train Loss={train_loss:.4f} — Val MAE={val_mae:.4f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), MODEL_OUT)

    return model, val_loader, device

# Extrae errores de reconstrucción
def get_errors(model, loader, device):
    return compute_reconstruction_error(model, loader, device)

# Carga y evalúa anómalos

def get_anomalies(model, df_meta, device):
    # PTB
    anom_ptb = load_ptbxl(PTB_DIR, healthy_only=False)
    ptb_loader = DataLoader(TensorDataset(torch.tensor(anom_ptb)), batch_size=BATCH)
    ptb_errors = compute_reconstruction_error(model, ptb_loader, device)
    print(f"Anómalos PTB-XL: {len(ptb_errors)}")
    # Chapman
    anom_chap = load_chapman(CHAP_DIR, healthy_only=False)
    chap_loader = DataLoader(TensorDataset(torch.tensor(anom_chap)), batch_size=BATCH)
    chap_errors = compute_reconstruction_error(model, chap_loader, device)
    print(f"Anómalos Chapman: {len(chap_errors)}")
    return ptb_errors, chap_errors

# Evaluación con ROC y PR

def evaluate_all(healthy_errors, ptb_errors, chap_errors):
    # concatena
    errors = np.concatenate([healthy_errors, ptb_errors, chap_errors])
    y_true = np.concatenate([
        np.zeros_like(healthy_errors),
        np.ones_like(ptb_errors),
        np.ones_like(chap_errors)
    ])
    # ROC
    fpr, tpr, thr_roc, roc_auc = compute_roc(y_true, errors)
    # PR
    precision, recall, thr_pr, pr_auc = compute_pr(y_true, errors)
    # umbral Youden
    best_thr = optimal_threshold(fpr, tpr, thr_roc)

    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR  AUC: {pr_auc:.3f}")
    print(f"Best threshold (Youden): {best_thr:.4f}")

    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'youden_threshold': float(best_thr)
    }
    save_metrics(metrics, METRICS_OUT)
    print(f"Metrics saved to {METRICS_OUT}")

if __name__ == '__main__':
    # 1) Carga normales y entrena
    normals = load_normals()
    model, val_loader, device = train_cvae(normals)
    # 2) Errores sanos
    healthy_errors = get_errors(model, val_loader, device)
    # 3) Anómalos
    df_meta = pd.read_csv(find_file(PTB_DIR, 'ptbxl_database.csv'))
    ptb_errors, chap_errors = get_anomalies(model, df_meta, device)
    # 4) Evaluación final
    evaluate_all(healthy_errors, ptb_errors, chap_errors)
