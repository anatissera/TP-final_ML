import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
import pandas as pd

from preprocess import (
    load_sanos,
    load_ptbxl_anomalies,
    load_chapman_anomalies
)

# Augmentation
def augment_ecg(sig):
    sig = sig + 0.01 * np.random.randn(*sig.shape)
    scale = np.random.uniform(0.8, 1.2)
    return sig * scale

# Beta schedules
def beta_linear(epoch, epochs):
    return 1.0 + 4.0 * (epoch / (epochs-1))

def beta_cyclic(epoch, cycle=10, beta_max=4.0):
    phase = epoch % cycle
    return beta_max * (phase / (cycle-1))

# Residual block
class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size//2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(),
            nn.Conv1d(ch, ch, kernel_size, padding=padding, dilation=dilation)
        )
    def forward(self, x):
        return x + self.block(x)

# VAE model
torch.manual_seed(0)
class VAE1D(nn.Module):
    def __init__(self, input_ch=1, latent_dim=16, seq_len=2048, n_blocks=3):
        super().__init__()
        layers = [nn.Conv1d(input_ch, 32, 19, 2, 9), nn.LeakyReLU()]
        for i in range(n_blocks):
            layers.append(ResBlock1D(32, 3, 2**i))
        self.encoder = nn.Sequential(*layers)
        self.out_len = seq_len // 2
        flat_size = 32 * self.out_len
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logv = nn.Linear(flat_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, self.out_len)),
            nn.ConvTranspose1d(32, input_ch, 19, 2, 9, output_padding=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(self.flatten(h)), self.fc_logv(self.flatten(h))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(self.fc_dec(z))

    def forward(self, x):
        mu, logv = self.encode(x)
        return mu, logv

# Compute scores
def compute_scores(model, loader, device, beta):
    model.eval()
    errs, zs = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mu, logv = model.encode(x)
            z = model.reparameterize(mu, logv)
            rec = model.decode(z)
            mse = ((rec - x)**2).mean(dim=[1,2]).cpu().numpy()
            kl = (-0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum(dim=1)).cpu().numpy()
            elbo = -mse - beta * kl
            errs.append(elbo)
            zs.append(mu.cpu().numpy())
    return np.concatenate(errs), np.vstack(zs)

# Main hyperparameter sweep
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1) Load data: only normals for training, anomalies for testing
    norms = load_sanos()
    ptb   = load_ptbxl_anomalies()
    chap  = load_chapman_anomalies()
    anos  = np.concatenate([ptb, chap], axis=0)

    # Convert to tensors with proper channel dim
    X_norm = torch.tensor(norms[:,1:2,:], dtype=torch.float32)
    X_ano  = torch.tensor(anos[:,1:2,:], dtype=torch.float32)

    # Split normals into train/val
    n_norm = len(X_norm)
    i1 = int(0.6 * n_norm)
    i2 = int(0.8 * n_norm)
    train_norm = X_norm[:i1]
    val_norm   = X_norm[i1:i2]

    # Hyperparameter grid
    grid = {
        'latent_dim': [8, 16, 32],
        'lr': [1e-3, 1e-4],
        'n_blocks': [2, 3],
        'beta_fn': [('linear', beta_linear), ('cyclic', beta_cyclic)]
    }
    results = []

    for ld in grid['latent_dim']:
        for lr in grid['lr']:
            for nb in grid['n_blocks']:
                for name, beta_fn in grid['beta_fn']:
                    # a) Train VAE on normals only
                    train_ds = TensorDataset(train_norm, torch.zeros(len(train_norm)))
                    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                    model = VAE1D(input_ch=1, latent_dim=ld, n_blocks=nb).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    for epoch in range(10):
                        model.train()
                        for x, _ in train_loader:
                            x = x.to(device)
                            mu, logv = model.encode(x)
                            z = model.reparameterize(mu, logv)
                            rec = model.decode(z)
                            recon_loss = ((rec - x)**2).mean()
                            kl_loss = (-0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum())/x.size(0)
                            beta = beta_fn(epoch, 10)
                            loss = recon_loss + beta * kl_loss
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # b) Prepare validation set: equal normals and anomalies
                    N = len(val_norm)
                    val_ano = X_ano[:N]
                    X_val = torch.cat([val_norm, val_ano], dim=0)
                    y_val = np.concatenate([np.zeros(N), np.ones(N)])
                    val_ds = TensorDataset(X_val, torch.tensor(y_val, dtype=torch.long))
                    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

                    # c) Compute scores on validation
                    beta_last = beta_fn(9, 10)
                    errs, zs = compute_scores(model, val_loader, device, beta_last)

                    # d) Mahalanobis
                    mu_bar = zs[:N].mean(axis=0)
                    cov = np.cov(zs[:N].T) + 1e-6 * np.eye(ld)
                    invcov = np.linalg.inv(cov)
                    dM = np.sqrt(((zs - mu_bar) @ invcov * (zs - mu_bar)).sum(axis=1))

                    # e) Combined sweep
                    best_a, best_auc = 0, 0
                    for a in [0, 0.25, 0.5, 0.75, 1]:
                        comb = a * errs + (1 - a) * dM
                        auc = roc_auc_score(y_val, -comb)
                        if auc > best_auc:
                            best_a, best_auc = a, auc

                    # f) XGBoost classifier on validation set
                    Xf = np.hstack([errs.reshape(-1,1), zs])
                    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    clf.fit(Xf, y_val)
                    probs = clf.predict_proba(Xf)[:,1]
                    clf_auc = roc_auc_score(y_val, probs)

                    results.append({
                        'latent_dim': ld,
                        'lr': lr,
                        'n_blocks': nb,
                        'beta': name,
                        'best_alpha': best_a,
                        'auc_comb': best_auc,
                        'auc_xgb': clf_auc
                    })
    # Save sweep results
    pd.DataFrame(results).to_csv('hyperparam_sweep_results.csv', index=False)
    print("Sweep completo. Resultados en hyperparam_sweep_results.csv")
