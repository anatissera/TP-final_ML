

# pipeline.py
"""
Pipeline de entrenamiento, validación y test en sanos vs anomalías PTB-XL y Chapman.
Basado en Jang et al. (2021): 18 epochs, lr=1e-3, latent=60, MSE+KL, segmentos de 8.192s a 250Hz.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm
from scipy.io import loadmat


# Importa TODO de tu preprocess.py
from preprocess import (
    load_ptbxl, load_chapman,             # cargas en batch
    load_ptbxl_signal,                    # carga señal individual PTB-XL
    apply_highpass, resample_signal,      # funciones de preprocesamiento
    zscore_normalize, segment_signal,
    find_file
)
from model import CVAE, loss_function
from test_functions import (
    compute_reconstruction_error,
    evaluate_detection,
    save_metrics
)


# RUTAS y HYPERPARAMS

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


EPOCHS    = 18
BATCH     = 32
LR        = 1e-3
LATENT    = 60
MODEL_OUT = 'best_cvae.pth'
METRICS_OUT = 'metrics_combined_lead_II.json'


def preprocess_and_segment(sig, fs):
    """
    Aplica filtro, remuestreo y normalización a una señal (12, L).
    """
    sig = apply_highpass(sig, fs)
    sig = resample_signal(sig, fs)
    sig = zscore_normalize(sig)
    sig = segment_signal(sig)
    return sig


def load_ptbxl_anomalies(ptb_dir, meta_df):
    """
    Recorre el metadata DF de PTB-XL para cargar solo las señales con SCP != NORM.
    Devuelve array (N,12,2048).
    """
    out = []
    for _, row in meta_df.iterrows():
        fn = row['filename_lr']
        hea = os.path.basename(fn) + '.hea'
        hea_path = find_file(ptb_dir, hea)
        prefix = os.path.splitext(hea_path)[0]
        rec = load_ptbxl_signal(ptb_dir, fn)  # p_signal, fs
        sig, fs = rec
        if not row['scp_codes'].count('NORM'):
            sig = preprocess_and_segment(sig, fs)
            out.append(sig)
    return np.stack(out) if out else np.empty((0,12,2048))


# def load_chapman_anomalies(chap_dir):
#     """
#     Carga todas las señales Chapman y filtra solo las anomalías (Dx != 426177001).
#     """
#     out = []
#     for root, _, files in os.walk(chap_dir):
#         for f in files:
#             if f.endswith('.hea') and 'Zone' not in f:
#                 hea_path = os.path.join(root, f)
#                 # extrae Dx
#                 with open(hea_path) as fh:
#                     lines = fh.read().splitlines()
#                 dx = next((l.split()[1] for l in lines if l.startswith('#Dx:')), '')
#                 if dx != '426177001':
#                     rec_id = f.replace('.hea','')
#                     mat = os.path.join(root, rec_id + '.mat')
#                     sig = np.load(mat)['val'] if mat.endswith('.npz') else __import__('scipy.io').loadmat(mat)['val']
#                     sig = preprocess_and_segment(sig, 500)
#                     out.append(sig)
#     return np.stack(out) if out else np.empty((0,12,2048))


def load_chapman_anomalies(chap_dir):
    """
    Carga todas las señales Chapman y filtra solo las anomalías (Dx != 426177001).
    """
    out = []
    for root, _, files in os.walk(chap_dir):
        for f in files:
            if f.endswith('.hea') and 'Zone' not in f:
                hea_path = os.path.join(root, f)
                # extrae Dx
                with open(hea_path) as fh:
                    lines = fh.read().splitlines()
                dx = next((l.split()[1] for l in lines if l.startswith('#Dx:')), '')
                if dx != '426177001':
                    rec_id = f.replace('.hea','')
                    mat_path = os.path.join(root, rec_id + '.mat')
                    # aquí usamos loadmat importado
                    sig = loadmat(mat_path)['val']   # ya es (12, L)
                    sig = preprocess_and_segment(sig, 500)
                    out.append(sig)
    return np.stack(out) if out else np.empty((0,12,2048))


def load_sanos():
    # --- Carga y prepara sanos PTB+Chapman ---
    print("Cargando sanos PTB-XL y Chapman…")
    normals_ptb  = load_ptbxl(PTB_DIR, healthy_only=True)
    normals_chap = load_chapman(CHAP_DIR, healthy_only=True)
    data = np.concatenate([normals_ptb, normals_chap], axis=0)
    print(f"Total sanos disponibles: {len(data)} señales")
    
    return data

def main():
    data = load_sanos()
    model, val_loader, device= training(data)

def training(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Train/Val split 80/20
    n_val = int(0.2 * len(data))
    n_train = len(data) - n_val
    train_set, val_set = random_split(data, [n_train, n_val])
    train_loader = DataLoader(TensorDataset(torch.tensor(train_set)), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_set)), batch_size=BATCH)

    # --- Modelo ---
    # model = CVAE(in_channels=1, latent_dim=LATENT, input_length=2048).to(device)
    model = CVAE(in_channels=1, latent_dim=60, input_length=2048).to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    best_mae = float('inf')

    # Entrenamiento + validación
    print("Iniciando entrenamiento…")
    for ep in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        for (x,) in tqdm(train_loader, desc=f"Ep {ep}/{EPOCHS} Train", leave=False):
            x = x.float().to(device)
            recon, mu, logvar = model(x)
            loss = loss_function(recon, x, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
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



# test

def load(model, val_loader, device):
    # --- Test: sanos hold-out vs anomalías ---
    print("Cargando metadata PTB-XL para anomalías…")
    df_meta = pd.read_csv(find_file(PTB_DIR, 'ptbxl_database.csv'))
    # # Hold-out sanos
    # healthy_errors = compute_reconstruction_error(model, val_loader, device)
    # print(f"Hold-out sanos MAE mean={healthy_errors.mean():.4f}, std={healthy_errors.std():.4f}")
    
    # hold-out sanos
    healthy_errors = compute_reconstruction_error(model, val_loader, device)
    thr = np.percentile(healthy_errors, 12.16)
    print(f"Threshold @12pct de sanos: {thr:.4f}")

    
    return df_meta, healthy_errors, thr


def anomalos_ptb(df_meta, model, device):
    # Anómalos PTB-XL
    anom_ptb = load_ptbxl_anomalies(PTB_DIR, df_meta)
    ptb_loader = DataLoader(TensorDataset(torch.tensor(anom_ptb)), batch_size=BATCH)
    ptb_errors = compute_reconstruction_error(model, ptb_loader, device)
    print(f"Anómalos PTB-XL: {len(anom_ptb)} señales")
    
    return ptb_errors
    

def anomalos_chap(model, device):

    # Anómalos Chapman
    anom_chap = load_chapman_anomalies(CHAP_DIR)
    chap_loader = DataLoader(TensorDataset(torch.tensor(anom_chap)), batch_size=BATCH)
    chap_errors = compute_reconstruction_error(model, chap_loader, device)
    print(f"Anómalos Chapman: {len(anom_chap)} señales")
    
    return chap_errors


def metrics(healthy_errors, ptb_errors, chap_errors):

    # Métricas combinadas
    y_true = np.concatenate([
        np.zeros_like(healthy_errors),
        np.ones_like(ptb_errors),
        np.ones_like(chap_errors)
    ])
    y_pred = np.concatenate([
        healthy_errors,
        ptb_errors,
        chap_errors
    ]) > (healthy_errors.mean() + 3 * healthy_errors.std())

    metrics = evaluate_detection(y_true, y_pred)
    print("Métricas final (sanos vs PTB+Chapman anomalías):", metrics)
    save_metrics(metrics, METRICS_OUT)
    print(f"Resultados guardados en {METRICS_OUT}")


def metrics_fixed_threshold(healthy_errors, ptb_errors, chap_errors, threshold):
    y_true = np.concatenate([
        np.zeros_like(healthy_errors),
        np.ones_like(ptb_errors),
        np.ones_like(chap_errors)
    ])
    y_pred = np.concatenate([
        healthy_errors,
        ptb_errors,
        chap_errors
    ]) > threshold

    metrics = evaluate_detection(y_true, y_pred)
    print("Métricas (con umbral percentil 12 de sanos):", metrics)
    save_metrics(metrics, METRICS_OUT)
    print(f"Resultados guardados en {METRICS_OUT}")
    

if __name__ == '__main__':
    main()

