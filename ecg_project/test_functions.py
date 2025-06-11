"""
test_functions.py
Functions to evaluate the CVAE.
"""
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# def compute_reconstruction_error(model, loader, device):
#     model.eval()
#     errors = []
#     with torch.no_grad():
#         for (x,) in loader:
#             x = x.to(device).float()  # <- ESTE CASTEO ES CLAVE
#             recon, _, _ = model(x)
#             err = torch.mean((recon - x) ** 2, dim=[1, 2])  # por señal
#             errors.append(err.cpu().numpy())
#     return np.concatenate(errors)


def compute_reconstruction_error(model, loader, device):
    model.eval()
    errs = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device).float()
            recon, _, _ = model(x)
            # error absoluto medio por muestra
            err = torch.mean(torch.abs(recon - x), dim=[1,2])
            errs.append(err.cpu().numpy())
    return np.concatenate(errs)


def evaluate_detection(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred)
    }

def save_metrics(metrics, outfile):
    import json
    with open(outfile, 'w') as f:
        json.dump(metrics, f, indent=2)
