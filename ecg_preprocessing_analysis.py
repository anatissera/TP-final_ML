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
def list_directory(start_dir, depth=1):
    """
    Prints directory structure up to given depth.
    """
    for root, dirs, files in os.walk(start_dir):
        level = root.replace(start_dir, '').count(os.sep)
        if level > depth:
            continue
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < depth:
            for f in files:
                print(f"{indent}  - {f}")

# ---------- Parse Chapman Header ----------
def parse_chapman_header(hea_path):
    """
    Extracts metadata from Chapman-Shaoxing .hea header.
    Returns dict with record id, fs, samples, age, sex, diagnosis, and lead boundaries.
    """
    with open(hea_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    # Header line example: JS00001.mat 16+24 1000/mV 16 0 -254 21756 0 I
    parts = lines[0].split()
    record_id = parts[0].replace('.mat', '')
    n_leads = parts[1]
    fs = parts[2]
    total_samples = parts[3]
    # Following 12 lines: lead info
    leads = {}
    for i in range(1, 13):
        row = lines[i].split()
        first_val = row[5]
        last_val = row[6]
        lead_name = row[7]
        leads[lead_name] = {'first': first_val, 'last': last_val}
    # Clinical metadata at end
    # Last line format: Aged=XX Sex=M dx=426177001 ...
    meta_line = next((l for l in lines if 'dx=' in l), '')
    age = None
    sex = None
    dx = None
    for token in meta_line.split():
        if token.startswith('Age=') or token.startswith('Age='):
            age = token.split('=')[1]
        if token.startswith('Sex='):
            sex = token.split('=')[1]
        if token.startswith('dx='):
            dx = token.split('=')[1]
    metadata = {
        'record': record_id,
        'n_leads': n_leads,
        'fs': fs,
        'samples': total_samples,
        'age': age,
        'sex': sex,
        'diagnosis': dx,
    }
    # Include lead first/last values
    for lead, v in leads.items():
        metadata[f'{lead}_first'] = v['first']
        metadata[f'{lead}_last'] = v['last']
    return metadata

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
                hea = os.path.join(root, file)
                recs.append(parse_chapman_header(hea))
    return pd.DataFrame(recs)

# ---------- Signal Loading ----------
def load_ptbxl_signal(ptb_dir, filename_lr):
    record = rdrecord(os.path.join(ptb_dir, filename_lr))
    return record.p_signal.T, record.fs  # shape: (n_leads, n_samples)

def load_chapman_signal(chap_dir, record_id):
    mat_path = os.path.join(chap_dir, f"{record_id}.mat")
    data = loadmat(mat_path)['val']  # shape: (n_leads, n_samples)
    fs = 500  # known sampling rate
    return data, fs

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
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass(data, fs, cutoff=40.0, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# ---------- Main Analysis ----------
if __name__ == '__main__':
    # Paths
    PTBXL_DIR = 'data/ptb-xl'
    CHAPMAN_DIR = 'data/ChapmanShaoxing'

    # Inspect directory structures
    print("PTB-XL structure:")
    list_directory(PTBXL_DIR)
    print("\nChapman-Shaoxing structure:")
    list_directory(CHAPMAN_DIR)

    # Load metadata and filter normals
    ptb_meta = load_ptbxl_metadata(PTBXL_DIR)
    ptb_norm = ptb_meta[ptb_meta['scp_codes'].str.contains('NORM', na=False)]
    print(f"PTB-XL normales: {len(ptb_norm)} registros")

    chap_meta = load_chapman_metadata(CHAPMAN_DIR)
    chap_norm = chap_meta[chap_meta['diagnosis'] == '426177001']  # SR code
    print(f"Chapman-Shaoxing normales: {len(chap_norm)} registros")

    # Select first normal record from PTB-XL
    sample = ptb_norm.iloc[0]
    sig, fs = load_ptbxl_signal(PTBXL_DIR, sample['filename_lr'])
    time = np.arange(sig.shape[1]) / fs

    # Plot raw signal
    plot_signal(time, sig, lead=0, title="Raw PTB-XL ECG")

    # Plot FFT
    plot_fft(sig, fs, lead=0, title="Spectrum PTB-XL ECG")

    # Apply lowpass filter and plot
    filtered = apply_lowpass(sig, fs, cutoff=40.0)
    plot_signal(time, np.vstack([sig, filtered]), lead=0, title="Raw vs Filtered (Lead 0)")
    
    # Similarly for Chapman-Shaoxing
    chap_sample = chap_norm.iloc[0]
    chap_sig, chap_fs = load_chapman_signal(CHAPMAN_DIR, chap_sample['record'])
    chap_time = np.arange(chap_sig.shape[1]) / chap_fs

    plot_signal(chap_time, chap_sig, lead=0, title="Raw Chapman ECG")
    plot_fft(chap_sig, chap_fs, lead=0, title="Spectrum Chapman ECG")
    chap_filt = apply_lowpass(chap_sig, chap_fs, cutoff=40.0)
    plot_signal(chap_time, np.vstack([chap_sig, chap_filt]), lead=0, title="Raw vs Filtered Chapman")
