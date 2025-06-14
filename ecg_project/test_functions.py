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


# test_functions.py (o un nuevo módulo de scores)

import torch
import numpy as np

def compute_anomaly_scores(model, loader, device, beta=1.0, sigma=1.0):
    """
    Para cada muestra devuelve un diccionario con:
      mse, logp, kl, elbo, score
    """
    model.eval()
    all_scores = {k: [] for k in ('mse','logp','kl','elbo','score')}
    with torch.no_grad():
        for (x,) in loader:
            x = x.float().to(device)
            recon, mu, logvar = model(x)
            # 1) MSE por muestra
            mse = torch.mean((recon-x)**2, dim=[1,2])
            # 2) recon log-prob (Gaussiano isotrópico σ)
            recon_logprob = -0.5 * torch.sum(
                (recon - x)**2 / sigma**2 + torch.log(2*np.pi*sigma**2),
                dim=[1,2]
            )
            # 3) kl divergencia
            kl = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1)
            # 4) elbo y score
            elbo = recon_logprob - beta * kl
            score = -elbo
            # guardar
            for k, v in zip(('mse','logp','kl','elbo','score'),
                            (mse, recon_logprob, kl, elbo, score)):
                all_scores[k].append(v.cpu().numpy())
    # concatenar
    for k in all_scores:
        all_scores[k] = np.concatenate(all_scores[k])
    return all_scores
