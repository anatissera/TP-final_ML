import numpy as np
from torch.utils.data import Dataset
import torch

# class ECGWithFMM(Dataset):
#     def __init__(self, signals: np.ndarray, coeffs: np.ndarray):
#         """
#         signals: array (N, 1, 2048)
#         coeffs:  array (N, C)   donde C = num_coefs (55)
#         """
#         assert signals.shape[0] == coeffs.shape[0]
#         self.signals = signals.astype(np.float32)
#         self.coeffs  = coeffs.astype(np.float32)

#     def __len__(self):
#         return len(self.signals)

#     def __getitem__(self, idx):
#         # señal original (1, 2048)
#         x = self.signals[idx]
#         # coeficientes (C,)
#         c = self.coeffs[idx]
#         # repetir cada coeficiente a lo largo de los 2048 pasos
#         c_exp = np.repeat(c[:, None], x.shape[-1], axis=1)  # (C, 2048)
#         # apilar señal + coef (se crea un tensor (1+C, 2048))
#         x_cat = np.vstack([x, c_exp])
#         return x_cat

    
    
    
from torch.utils.data import Dataset

class ECGWithFMM(Dataset):
    def __init__(self, signals: np.ndarray, coeffs: np.ndarray):
        # signals: (N, 2048, 1) → torch espera (N, C, L) para Conv1D
        self.signals = np.transpose(signals, (0, 2, 1)).astype(np.float32)
        # coeffs: (N, C_fmm)
        self.coeffs  = coeffs.astype(np.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]   # (1, 2048)
        c = self.coeffs[idx]    # (C_fmm,)
        # repetir c para que quede (C_fmm, 2048)
        c_exp = np.repeat(c[:, None], x.shape[-1], axis=1)
        # apilar canal ECG + canales FMM → (1 + C_fmm, 2048)
        x_cat = np.vstack([x, c_exp])
        return torch.from_numpy(x_cat)

    
class ECGWithLabels(Dataset):
    def __init__(self, signals, coeffs, labels):
        self.base   = ECGWithFMM(signals, coeffs)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        return self.base[idx], self.labels[idx]