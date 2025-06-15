import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
import wfdb
from wfdb import rdrecord
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, r2_score
from xgboost import XGBClassifier


# Global parameters
target_fs       = 250      # Desired sampling frequency
segment_sec     = 8.192    # Seconds per segment
segment_samples = int(target_fs * segment_sec)  # ~2048 samples

# --- File finder ---
def find_file(root: str, filename: str) -> str:
    """
    Recursively search for `filename` under `root` directory.
    """
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"{filename} not found under {root}")

# --- Preprocessing building blocks ---
def butter_highpass(cutoff: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)


def apply_highpass(sig: np.ndarray, fs: float, cutoff: float = 0.5, order: int = 4) -> np.ndarray:
    """Highpass filter channel-wise to remove baseline drift."""
    b, a = butter_highpass(cutoff, fs, order)
    return filtfilt(b, a, sig, axis=1)


def resample_signal(sig: np.ndarray, orig_fs: float) -> np.ndarray:
    """Resample multichannel signal to target_fs."""
    n_samples = int(sig.shape[1] * target_fs / orig_fs)
    return resample(sig, n_samples, axis=1)


def segment_signal(sig: np.ndarray) -> np.ndarray:
    """Segment to fixed length, selecting second lead (index 1)."""
    # Expect sig shape (n_leads, L)
    if sig.shape[0] > 1:
        lead = sig[1:2, :]
    else:
        lead = sig
    L = lead.shape[1]
    if L >= segment_samples:
        return lead[:, :segment_samples]
    pad = segment_samples - L
    return np.pad(lead, ((0,0),(0,pad)), mode='constant')


def preprocess_basic(sig: np.ndarray, fs: float) -> np.ndarray:
    """Filter, resample, and segment, but do not normalize."""
    x = apply_highpass(sig, fs)
    x = resample_signal(x, fs)
    x = segment_signal(x)
    return x

# --- Z-score normalization (train stats only) ---
def compute_zscore_stats(signals: np.ndarray):
    """
    Compute per-channel mean/std for array of shape (N,1,L).
    Returns mean and std arrays of shape (1,1,L).
    """
    mean = np.mean(signals, axis=(0,2), keepdims=True)
    std  = np.std(signals,  axis=(0,2), keepdims=True) + 1e-6
    return mean, std


def apply_zscore(signals: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply Z-score using provided mean and std."""
    return (signals - mean) / std

# --- PTB-XL loading ---
def load_ptbxl_signal(ptb_dir: str, filename_lr: str):
    """Load one PTB-XL record, return (12,L) and sampling rate."""
    # Find header
    hea_path = find_file(ptb_dir, os.path.basename(filename_lr) + '.hea')
    rec = rdrecord(os.path.splitext(hea_path)[0])
    sig = rec.p_signal.T  # (12, L)
    fs = rec.fs
    return sig, fs


def load_ptbxl(ptb_dir: str, healthy_only: bool = True) -> np.ndarray:
    """Load all PTB-XL signals, filter healthy if requested."""
    csv_path = find_file(ptb_dir, 'ptbxl_database.csv')
    df = pd.read_csv(csv_path)
    if healthy_only:
        df = df[df['scp_codes'].str.contains('NORM', na=False)]
    records = []
    for _, row in df.iterrows():
        sig, fs = load_ptbxl_signal(ptb_dir, row['filename_lr'])
        records.append(preprocess_basic(sig, fs))
    return np.stack(records, axis=0)  # (N,1,L)

# --- Chapman-Shaoxing loading ---
def load_chapman(chap_dir: str, healthy_only: bool = True) -> np.ndarray:
    """Load Chapman-Shaoxing signals, healthy_only filters sinus rhythm."""
    records = []
    for root, _, files in os.walk(chap_dir):
        for f in files:
            if not f.endswith('.hea') or 'Zone' in f:
                continue
            hea_path = os.path.join(root, f)
            with open(hea_path) as fh:
                lines = fh.read().splitlines()
            dx = next((l.split()[1] for l in lines if l.startswith('#Dx:')), '')
            if healthy_only and dx != '426177001':
                continue
            mat_path = os.path.join(root, f.replace('.hea','.mat'))
            val = loadmat(mat_path)['val']  # (12,L)
            records.append(preprocess_basic(val, 500))
    return np.stack(records, axis=0) if records else np.empty((0,1,segment_samples))

# --- MIT-BIH loading (example) ---
def load_mitbih(mit_dir: str) -> np.ndarray:
    """Load MIT-BIH arrhythmia database signals."""
    records = []
    for root, _, files in os.walk(mit_dir):
        for f in files:
            if not f.endswith('.hea'):
                continue
            rec = rdrecord(os.path.splitext(os.path.join(root,f))[0])
            sig = rec.p_signal.T
            records.append(preprocess_basic(sig, rec.fs))
    return np.stack(records, axis=0) if records else np.empty((0,1,segment_samples))

# --- Combined loader for healthy normals ---
def load_sanos(ptb_dir: str, chap_dir: str) -> np.ndarray:
    """
    Load all healthy signals from PTB-XL and Chapman-Shaoxing.
    Returns array of shape (N,1,segment_samples).
    """
    ptb = load_ptbxl(ptb_dir, healthy_only=True)
    chap = load_chapman(chap_dir, healthy_only=True)
    if ptb.size == 0 and chap.size == 0:
        return np.empty((0,1,segment_samples))
    return np.concatenate([ptb, chap], axis=0)


# --- Combined loader for all anomalies ---
def load_all_anomalies(ptb_dir: str, chap_dir: str, meta_df: pd.DataFrame) -> np.ndarray:
    """
    Load all anomalous signals from PTB-XL (using meta_df) and Chapman-Shaoxing.
    """
    ptb_ano = load_ptbxl(ptb_dir, healthy_only=False)
    chap_ano = load_chapman(chap_dir, healthy_only=False)
    if ptb_ano.size == 0 and chap_ano.size == 0:
        return np.empty((0,1,segment_samples))
    return np.concatenate([ptb_ano, chap_ano], axis=0)

