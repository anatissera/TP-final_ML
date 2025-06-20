import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
import pandas as pd

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, r2_score



# from preprocess import (
#     load_sanos,
#     load_ptbxl_anomalies,
#     load_chapman_anomalies
# )

from preprocess import (
    load_sanos,
    load_all_anomalies,
    compute_zscore_stats,
    apply_zscore,
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

# # Training and evaluation with best hyperparameters
# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # 1) Load and prepare data
#     norms = load_sanos()
#     ptb   = load_ptbxl_anomalies()
#     chap  = load_chapman_anomalies()
#     anos  = np.concatenate([ptb, chap], axis=0)

#     # Select lead II and reshape
#     norms_lead = norms[:, 0, :]
#     anos_lead  = anos[:, 0, :]
#     X_norm = torch.tensor(norms_lead, dtype=torch.float32).unsqueeze(1)
#     X_ano  = torch.tensor(anos_lead,  dtype=torch.float32).unsqueeze(1)

#     # Split normals
#     n_norm = len(X_norm)
#     i1 = int(0.8 * n_norm)
#     train_norm = X_norm[:i1]
#     val_norm   = X_norm[i1:]

#     # Chosen hyperparameters
#     latent_dim = 32
#     lr         = 1e-3
#     n_blocks   = 3
#     beta_fn    = beta_cyclic
#     epochs     = 50

#     # a) Train VAE on normals only
#     train_ds    = TensorDataset(train_norm, torch.zeros(len(train_norm)))
#     train_loader= DataLoader(train_ds, batch_size=32, shuffle=True)
#     model = VAE1D(input_ch=1, latent_dim=latent_dim, n_blocks=n_blocks).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for x, _ in train_loader:
#             x = x.to(device)
#             mu, logv = model.encode(x)
#             z = model.reparameterize(mu, logv)
#             rec = model.decode(z)
#             recon_loss = ((rec - x)**2).mean()
#             kl_loss = (-0.5*(1+logv-mu.pow(2)-logv.exp()).sum())/x.size(0)
#             beta = beta_fn(epoch, epochs)
#             loss = recon_loss + beta * kl_loss
#             optimizer.zero_grad(); loss.backward(); optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

#     # b) Prepare balanced test set
#     N = len(val_norm)
#     val_ano = X_ano[:N]
#     X_test  = torch.cat([val_norm, val_ano], dim=0)
#     y_test  = np.concatenate([np.zeros(N), np.ones(N)])
#     test_ds = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.long))
#     test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

#     # c) Compute ELBO scores and latents
#     beta_last = beta_fn(epochs-1, epochs)
#     errs, zs = compute_scores(model, test_loader, device, beta_last)

#     # d) Mahalanobis distance
#     mu_bar = zs[:N].mean(axis=0)
#     cov = np.cov(zs[:N].T) + 1e-6*np.eye(latent_dim)
#     invcov = np.linalg.inv(cov)
#     dM = np.sqrt(((zs-mu_bar)@invcov*(zs-mu_bar)).sum(axis=1))

#     # e) Combine with alpha=1 (ELBO) or tuned alpha, here use best_alpha from sweep
#     alpha = 1.0
#     combined_score = alpha*errs + (1-alpha)*dM

#     # f) XGBoost classifier on [err, latents]
#     Xf = np.hstack([errs.reshape(-1,1), zs])
#     clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#     clf.fit(Xf, y_test)
#     prob = clf.predict_proba(Xf)[:,1]

#     # g) Compute metrics
#     from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score
#     # Binarize at 0.5 for classification metrics
#     y_pred = (prob >= 0.5).astype(int)
#     metrics = {
#         'roc_auc': roc_auc_score(y_test, prob),
#         'precision': precision_score(y_test, y_pred),
#         'recall': recall_score(y_test, y_pred),
#         'f1': f1_score(y_test, y_pred),
#         'accuracy': accuracy_score(y_test, y_pred),
#         'r2': r2_score(y_test, prob)
#     }
#     print("Final metrics:", metrics)
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

def find_file(root, filename):
    """
    Busca recursivamente `filename` bajo `root` y devuelve la ruta completa.
    """
    for r, _, files in os.walk(root):
        if filename in files:
            return os.path.join(r, filename)
    return None


# Ahora buscás las rutas relativas automáticamente:
PTB_DIR = find_data_subfolder('ptb-xl/1.0.3')
CHAP_DIR = find_data_subfolder('ChapmanShaoxing')
MIT_DIR = find_data_subfolder('mitdb')





if __name__ == '__main__':
    # 1) Cargar datos sanos y anomalías
    normals   = load_sanos(PTB_DIR, CHAP_DIR)  # (N_norm,1,L)
    ptb_df    = pd.read_csv(find_file(PTB_DIR, 'ptbxl_database.csv'))
    anomalies = load_all_anomalies(PTB_DIR, CHAP_DIR, ptb_df)  # (N_ano,1,L)

        # 2) Split normales en DEV y TEST (80/20)
    n_norm = normals.shape[0]
    split_dev = int(0.8 * n_norm)
    dev_norm  = normals[:split_dev]
    test_norm = normals[split_dev:]

    # 3) Dentro de DEV, split en TRAIN y VAL (80/20 of DEV)
    n_dev = dev_norm.shape[0]
    split_train = int(0.8 * n_dev)
    train_norm = dev_norm[:split_train]
    val_norm   = dev_norm[split_train:]

    # 4) Normalización Z-score usando solo train_norm) Normalización Z-score usando solo train_norm
    mean, std = compute_zscore_stats(train_norm)
    train_norm = apply_zscore(train_norm, mean, std)
    val_norm   = apply_zscore(val_norm,   mean, std)
    test_norm  = apply_zscore(test_norm,  mean, std)
    anomalies  = apply_zscore(anomalies,  mean, std)

    # 4) Convertir a tensores
    train_tensor = torch.tensor(train_norm, dtype=torch.float32)
    val_tensor   = torch.tensor(val_norm,   dtype=torch.float32)
    test_tensor  = torch.tensor(test_norm,  dtype=torch.float32)
    ano_tensor   = torch.tensor(anomalies,  dtype=torch.float32)

    # Parámetros de entrenamiento final
    latent_dim = 32
    lr         = 1e-3
    n_blocks   = 3
    epochs     = 50
    batch_size = 32

    # 5) Entrena VAE sobre conjunto DEV (train_norm + val_norm)
    dev_tensor = torch.cat([train_tensor, val_tensor], dim=0)
    dev_ds     = TensorDataset(dev_tensor, torch.zeros(len(dev_tensor)))
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE1D(input_ch=1, latent_dim=latent_dim, n_blocks=n_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, _ in dev_loader:
            x = x.to(device)
            mu, logv = model.encode(x)
            z = model.reparameterize(mu, logv)
            rec = model.decode(z)
            recon_loss = ((rec - x)**2).mean()
            kl_loss = (-0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum()) / x.size(0)
            beta = beta_cyclic(epoch, cycle=10, beta_max=4.0)
            loss = recon_loss + beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dev_loader):.4f}")

    # 6) Prepara test final balanceado: test_norm vs primeras anomalías: test_norm vs primeras anomalías
    N_test = test_tensor.shape[0]
    test_x = torch.cat([test_tensor, ano_tensor[:N_test]], dim=0)
    y_test = np.concatenate([np.zeros(N_test), np.ones(N_test)])
    test_loader = DataLoader(TensorDataset(test_x, torch.tensor(y_test, dtype=torch.long)), batch_size=batch_size)

    # 7) Obtiene scores ELBO y z_mean en test
    beta_last = beta_cyclic(epochs-1, cycle=10, beta_max=4.0)
    errs, zs = compute_scores(model, test_loader, device, beta_last)

    # 8) Mahalanobis distance en test_norm
    mu_bar = zs[:N_test].mean(axis=0)
    cov    = np.cov(zs[:N_test].T) + 1e-6 * np.eye(latent_dim)
    invcov = np.linalg.inv(cov)
    dM     = np.sqrt(((zs - mu_bar) @ invcov * (zs - mu_bar)).sum(axis=1))

    # 9) Clasificador XGBoost en test
    # Limpiar infinities
    max_val = np.finfo(np.float32).max/10
    min_val = np.finfo(np.float32).min/10
    errs = np.nan_to_num(errs, nan=0.0, posinf=max_val, neginf=min_val)
    zs   = np.nan_to_num(zs,   nan=0.0, posinf=max_val, neginf=min_val)
    Xf   = np.hstack([errs.reshape(-1,1), zs])
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(Xf, y_test)
    probs = clf.predict_proba(Xf)[:,1]

    # 10) Métricas finales
    auc      = roc_auc_score(y_test, probs)
    y_pred   = (probs >= 0.5).astype(int)
    metrics  = {
        'roc_auc':   auc,
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'accuracy':  accuracy_score(y_test, y_pred),
        'r2':        r2_score(y_test, probs)
    }
    print('Final metrics:', metrics)
