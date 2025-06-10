import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import wfdb
from wfdb import rdrecord
from scipy.io import loadmat

# ---------- Utilities ----------
def list_directory(start_dir, depth=2):
    """Prints directory structure up to given depth."""
    for root, dirs, files in os.walk(start_dir):
        level = root.replace(start_dir, '').count(os.sep)
        if level > depth:
            continue
        indent = '  ' * level
        # print(f"{indent}{os.path.basename(root)}/")
        # if level < depth:
        #     for f in files:
        #         print(f"{indent}  - {f}")

# ---------- Parse Chapman Header ----------
def parse_chapman_header(hea_path):
    with open(hea_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    parts = lines[0].split()
    record_id = parts[0].replace('.mat', '')
    age = next((l.split()[1] for l in lines if l.startswith('#Age:')), None)
    sex = next((l.split()[1] for l in lines if l.startswith('#Sex:')), None)
    dx  = next((l.split()[1] for l in lines if l.startswith('#Dx:')), None)
    return {'record': record_id, 'age': age, 'sex': sex, 'diagnosis': dx}

# ---------- Loading Metadata ----------
def load_ptbxl_metadata(ptb_dir):
    csv = next((os.path.join(r, f) for r, _, fs in os.walk(ptb_dir)
                for f in fs if f == 'ptbxl_database.csv'), None)
    df = pd.read_csv(csv)
    return df[['ecg_id', 'filename_lr', 'scp_codes', 'age', 'sex']]

def load_chapman_metadata(path):
    recs = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.hea') and 'Zone' not in file:
                recs.append(parse_chapman_header(os.path.join(root, file)))
    return pd.DataFrame(recs)

def filter_chapman_normal(df):
    return df[df['diagnosis'].str.contains('426177001', na=False)].reset_index(drop=True)

# ---------- Signal Loading ----------
def load_ptbxl_signal(ptb_dir, filename_lr):
    hea_basename = os.path.basename(filename_lr) + '.hea'
    hea_path = None
    for root, _, files in os.walk(ptb_dir):
        if hea_basename in files:
            hea_path = os.path.join(root, hea_basename)
            break
    if hea_path is None:
        raise FileNotFoundError(f"No se encontró header PTB-XL para '{filename_lr}' (buscando '{hea_basename}') en {ptb_dir}")
    record_prefix = os.path.splitext(hea_path)[0]
    record = rdrecord(record_prefix)
    return record.p_signal.T, record.fs

def load_chapman_signal(chap_dir, record_id):
    mat_path = os.path.join(chap_dir, f"{record_id}.mat")
    data = loadmat(mat_path)['val']
    return data, 500

# ---------- Preprocessing ----------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, fs, lowcut=0.5, highcut=40.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def normalize_signal(data, method='zscore'):
    """Normalize each channel: 'zscore' or 'minmax'."""
    if method == 'zscore':
        mu = np.mean(data, axis=1, keepdims=True)
        sigma = np.std(data, axis=1, keepdims=True)
        return (data - mu) / (sigma + 1e-8)
    elif method == 'minmax':
        minv = np.min(data, axis=1, keepdims=True)
        maxv = np.max(data, axis=1, keepdims=True)
        return 2 * (data - minv) / (maxv - minv + 1e-8) - 1
    else:
        raise ValueError(f"Método de normalización desconocido: {method}")

# ---------- Visualization ----------
def plot_signal(time, signal, lead=0, title="ECG Signal"):
    plt.figure()
    plt.plot(time, signal[lead])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f"{title} - Lead {lead}")
    plt.grid(True)
    plt.show()

# ---------- Main Analysis ----------
if __name__ == '__main__':
    PTBXL_DIR = 'data/ptb-xl'
    CHAPMAN_DIR = 'data/ChapmanShaoxing'

    print("PTB-XL structure:")
    list_directory(PTBXL_DIR, depth=3)
    print("\nChapman-Shaoxing structure:")
    list_directory(CHAPMAN_DIR, depth=2)

    # Load metadata and filter normals
    ptb_meta = load_ptbxl_metadata(PTBXL_DIR)
    ptb_norm = ptb_meta[ptb_meta['scp_codes'].str.contains('NORM', na=False)]
    print(f"PTB-XL normales: {len(ptb_norm)} registros")

    chap_meta = load_chapman_metadata(CHAPMAN_DIR)
    chap_norm = filter_chapman_normal(chap_meta)
    print(f"Chapman-Shaoxing normales: {len(chap_norm)} registros")

    # PTB-XL sample
    sample = ptb_norm.iloc[0]
    sig_ptb, fs_ptb = load_ptbxl_signal(PTBXL_DIR, sample['filename_lr'])
    sig_ptb = apply_bandpass(sig_ptb, fs_ptb)
    sig_ptb = normalize_signal(sig_ptb, method='minmax')  # escala [-1,1]
    time_ptb = np.arange(sig_ptb.shape[1]) / fs_ptb
    plot_signal(time_ptb, sig_ptb, title="Preprocessed PTB-XL ECG [-1,1]")

    # Chapman sample
    if len(chap_norm) > 0:
        rec = chap_norm.iloc[0]['record']
        sig_chap, fs_chap = load_chapman_signal(CHAPMAN_DIR, rec)
        sig_chap = apply_bandpass(sig_chap, fs_chap)
        sig_chap = normalize_signal(sig_chap, method='minmax')  # escala [-1,1]
        time_chap = np.arange(sig_chap.shape[1]) / fs_chap
        plot_signal(time_chap, sig_chap, title="Preprocessed Chapman ECG [-1,1]")
    else:
        print("No hay registros normales de Chapman para procesar.")
