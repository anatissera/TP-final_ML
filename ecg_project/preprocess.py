"""
preprocess.py
Cargado y preprocesamiento de ECG:
  - PTB-XL (batch y señal individual)
  - Chapman-Shaoxing
  - MIT-BIH Arrhythmia (opcional)
Operaciones:
  - High-pass (0.5 Hz)
  - Resample a 250 Hz
  - Z-score por canal
  - Segmentación fija a 8.192 s → 2048 muestras
  - Búsqueda recursiva de archivos
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
import wfdb
from wfdb import rdrecord
from scipy.io import loadmat

# Parámetros globales de preprocessing
TARGET_FS       = 250      # frecuencia deseada
SEGMENT_SEC     = 8.192    # segundos de segmentación
SEGMENT_SAMPLES = int(TARGET_FS * SEGMENT_SEC)  # 2048 muestras

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)

def apply_highpass(sig, fs, cutoff=0.5, order=4):
    """
    Filtrado pasa-altos canal-a-canal para remover drift de línea base.
    """
    b, a = butter_highpass(cutoff, fs, order)
    return filtfilt(b, a, sig, axis=1)

def resample_signal(sig, orig_fs):
    """
    Remuestrea la señal multicanal a TARGET_FS.
    """
    n_samples = int(sig.shape[1] * TARGET_FS / orig_fs)
    return resample(sig, n_samples, axis=1)

def zscore_normalize(sig):
    """
    Normalización Z-score por canal (derivación).
    """
    mean = np.mean(sig, axis=1, keepdims=True)
    std  = np.std(sig,  axis=1, keepdims=True) + 1e-6
    return (sig - mean) / std


def segment_signal(sig):
    # sig: (12, L)
    # seleccionamos solo lead II, que suele ser el canal 1 (ajustá si fuera otro índice)
    sig = sig[1:2, :]   # ahora (1, L)
    if sig.shape[1] >= SEGMENT_SAMPLES:
        return sig[:, :SEGMENT_SAMPLES]
    pad = SEGMENT_SAMPLES - sig.shape[1]
    return np.pad(sig, ((0,0),(0,pad)), mode='constant')



def find_file(root, filename):
    """
    Busca recursivamente `filename` bajo `root` y devuelve la ruta completa.
    """
    for r, _, files in os.walk(root):
        if filename in files:
            return os.path.join(r, filename)
    return None


# ---------------------------------------
#  PTB-XL
# ---------------------------------------

def load_ptbxl(ptb_dir, healthy_only=True):
    """
    Carga en batch todos los ECG de PTB-XL sanos (o todos si healthy_only=False).
    Retorna array (N, 12, 2048).
    """
    # Encuentra CSV de metadata
    csv_path = find_file(ptb_dir, 'ptbxl_database.csv')
    if csv_path is None:
        raise FileNotFoundError(f"ptbxl_database.csv no encontrado en {ptb_dir}")
    df = pd.read_csv(csv_path)
    if healthy_only:
        df = df[df['scp_codes'].str.contains('NORM', na=False)]
    records = []
    for _, row in df.iterrows():
        sig, fs = load_ptbxl_signal(ptb_dir, row['filename_lr'])
        sig = apply_highpass(sig, fs)
        sig = resample_signal(sig, fs)
        sig = zscore_normalize(sig)
        sig = segment_signal(sig)
        records.append(sig)
    return np.stack(records, axis=0)

def load_ptbxl_signal(ptb_dir, filename_lr):
    """
    Carga un solo registro PTB-XL (prefix = filename_lr):
    devuelve (sig.T, fs) con sig de forma (12, L).
    """
    hea_name = os.path.basename(filename_lr) + '.hea'
    hea_path = find_file(ptb_dir, hea_name)
    if hea_path is None:
        raise FileNotFoundError(f"No encontré header {hea_name} en {ptb_dir}")
    prefix = os.path.splitext(hea_path)[0]
    rec = rdrecord(prefix)
    # p_signal es (L, 12), lo queremos (12, L)
    return rec.p_signal.T, rec.fs


# ---------------------------------------
#  Chapman-Shaoxing
# ---------------------------------------

def load_chapman(chap_dir, healthy_only=True):
    """
    Carga en batch señales de Chapman:
      - healthy_only=True → filtra Dx=426177001 (sinus rhythm)
      - healthy_only=False → carga todas
    Retorna array (N,12,2048).
    """
    records = []
    for root, _, files in os.walk(chap_dir):
        for f in files:
            if not f.endswith('.hea') or 'Zone' in f:
                continue
            hea_path = os.path.join(root, f)
            # Parsea diagnóstico
            with open(hea_path) as fh:
                lines = fh.read().splitlines()
            dx = next((l.split()[1] for l in lines if l.startswith('#Dx:')), '')
            if healthy_only and dx != '426177001':
                continue
            # Carga .mat asociado
            rec_id = f.replace('.hea','')
            mat_path = os.path.join(root, rec_id + '.mat')
            val = loadmat(mat_path)['val']  # forma (12, L)
            sig, fs0 = val, 500
            # Preprocesamiento idéntico
            sig = apply_highpass(sig, fs0)
            sig = resample_signal(sig, fs0)
            sig = zscore_normalize(sig)
            sig = segment_signal(sig)
            records.append(sig)
    return np.stack(records, axis=0) if records else np.empty((0,12,SEGMENT_SAMPLES))

# ---------------------------------------
#  MIT-BIH (opcional)
# ---------------------------------------

def load_mitbih(mit_dir):
    """
    Carga todos los .dat de MIT-BIH recursivamente.
    Retorna array (N,12,2048) o vacío si no hay.
    """
    records = []
    for root, _, files in os.walk(mit_dir):
        for f in files:
            if f.endswith('.dat'):
                prefix = os.path.splitext(os.path.join(root, f))[0]
                try:
                    rec = rdrecord(prefix)
                    sig, fs = rec.p_signal.T, rec.fs
                    sig = apply_highpass(sig, fs)
                    sig = resample_signal(sig, fs)
                    sig = zscore_normalize(sig)
                    sig = segment_signal(sig)
                    records.append(sig)
                except Exception as e:
                    print(f"Warning: no pude leer {prefix}: {e}")
    return np.stack(records, axis=0) if records else np.empty((0,12,SEGMENT_SAMPLES))








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
