import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import wfdb
from wfdb import rdrecord
from scipy.io import loadmat

# ---------- Utilities ----------
def list_directory(start_dir, depth=2):
    """
    Prints directory structure up to given depth.
    """
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
    """
    Extracts metadata from Chapman-Shaoxing .hea header.
    """
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
    """
    Loads PTB-XL record by locating the .hea header matching filename_lr.
    filename_lr is metadata string like 'records100/00000/00001_lr'.
    """
    # Derive expected header basename
    hea_basename = os.path.basename(filename_lr) + '.hea'
    hea_path = None
    for root, _, files in os.walk(ptb_dir):
        if hea_basename in files:
            hea_path = os.path.join(root, hea_basename)
            break
    if hea_path is None:
        raise FileNotFoundError(f"No se encontró header PTB-XL para '{filename_lr}' (buscando '{hea_basename}') en {ptb_dir}")
    # Record prefix is full path without extension
    record_prefix = os.path.splitext(hea_path)[0]
    record = rdrecord(record_prefix)
    return record.p_signal.T, record.fs

def load_chapman_signal(chap_dir, record_id):
    mat_path = os.path.join(chap_dir, f"{record_id}.mat")
    data = loadmat(mat_path)['val']
    return data, 500

# ---------- Visualization ----------

def plot_signal(time, signal, lead=0, title="ECG Signal"):
    plt.figure()
    plt.plot(time, signal[lead])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f"{title} - Lead {lead}")
    plt.grid(True)
    plt.show()

def plot_fft(signal, fs, lead=0, title="FFT of ECG Signal"):
    n = signal.shape[1]
    yf = fft(signal[lead])
    xf = fftfreq(n, 1 / fs)
    plt.figure()
    plt.plot(xf[:n//2], np.abs(yf[:n//2]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f"{title} - Lead {lead}")
    plt.grid(True)
    plt.show()

# ---------- Filtering ----------

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def apply_lowpass(data, fs, cutoff=40.0, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# ---------- Main Analysis ----------

if __name__ == '__main__':
    PTBXL_DIR = 'data/ptb-xl'
    CHAPMAN_DIR = 'data/ChapmanShaoxing'

    # Directory trees
    print("PTB-XL structure:")
    list_directory(PTBXL_DIR, depth=3)
    print("\nChapman-Shaoxing structure:")
    list_directory(CHAPMAN_DIR, depth=2)

    # Metadata
    ptb_meta = load_ptbxl_metadata(PTBXL_DIR)
    ptb_norm = ptb_meta[ptb_meta['scp_codes'].str.contains('NORM', na=False)]
    print(f"PTB-XL normales: {len(ptb_norm)} registros")

    chap_meta = load_chapman_metadata(CHAPMAN_DIR)
    chap_norm = filter_chapman_normal(chap_meta)
    print(f"Chapman-Shaoxing normales: {len(chap_norm)} registros")

    # PTB-XL sample
    sample = ptb_norm.iloc[0]
    sig, fs = load_ptbxl_signal(PTBXL_DIR, sample['filename_lr'])
    time = np.arange(sig.shape[1]) / fs
    plot_signal(time, sig, title="Raw PTB-XL ECG")

    # Chapman sample
    if len(chap_norm):
        rec = chap_norm.iloc[0]['record']
        chap_sig, chap_fs = load_chapman_signal(CHAPMAN_DIR, rec)
        chap_time = np.arange(chap_sig.shape[1]) / chap_fs
        plot_signal(chap_time, chap_sig, title="Raw Chapman ECG")
    else:
        print("No hay registros normales de Chapman para plotear.")
