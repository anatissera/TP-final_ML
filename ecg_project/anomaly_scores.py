# anomaly_scores.py
import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance

def compute_elbo_score(model, loader, device, beta=1.0, sigma=1.0):
    model.eval()
    scores, z_means = [], []
    with torch.no_grad():
        for (x,) in loader:
            x = x.float().to(device)
            recon, mu, logvar = model(x)
            # recon log-likelihood
            recon_term = -0.5 * torch.sum(
                ((x - recon)**2)/(sigma**2)
                + torch.log(2 * np.pi * (sigma**2)),
                dim=[1,2]
            )
            # KL divergence
            kl_term = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(),
                dim=1
            )
            elbo = recon_term - beta * kl_term
            score = -elbo
            scores.append(score.cpu().numpy())
            z_means.append(mu.cpu().numpy())
    return np.concatenate(scores), np.concatenate(z_means)


class LatentMahalanobis:
    def __init__(self):
        self.emp_cov = None
        self.mean = None

    def fit(self, z_normals):
        self.emp_cov = EmpiricalCovariance().fit(z_normals)
        self.mean = self.emp_cov.location_

    def score(self, z_batch):
        md2 = self.emp_cov.mahalanobis(z_batch)
        return np.sqrt(md2)
