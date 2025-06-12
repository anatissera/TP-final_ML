# vae_pipeline.py
"""
Pipeline para entrenar y evaluar un VAE 1D en señales ECG de 12 derivaciones:
- Carga metadata de PTB-XL o Chapman para filtrar solo señales normales.
- Dataset y DataLoader con splits train/val/test.
- Modelo VAE con encoder/decoder 1D.
- Métricas de reconstrucción: MSE y MAE en train, val y test.
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
import pandas as pd




# -------------------- Configuraciones --------------------
# Seleccionar dataset: 'ptbxl' o 'chapman'
DATASET = 'ptbxl/1.0.3'
BASE_DIR = 'data'
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 1000
SEQ_LEN = 5000      # longitud fija (10 s * 500 Hz)
LATENT_DIM = 64
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- Metadata loading --------------------
def get_normal_file_list():
    files = []
    if DATASET=='ptbxl':
        # Cargar metadata PTB-XL
        df = pd.read_csv(os.path.join(BASE_DIR,'ptb-xl','ptbxl_database.csv'))
        normals = df[df['scp_codes'].str.contains('NORM',na=False)]['filename_lr']
        for fn in normals:
            pat = os.path.join(BASE_DIR,'ptb-xl','records100',fn+'.mat')
            if os.path.exists(pat): files.append(pat)
    else:
        # ChapmanSHAOXING: buscar código 426177001 en headers
        for hea in glob.glob(os.path.join(BASE_DIR,'ChapmanShaoxing','*.hea')):
            with open(hea) as f:
                lines = f.readlines()
            dx = next((l.split()[1] for l in lines if l.startswith('#Dx:')),None)
            if dx and '426177001' in dx:
                mat = hea.replace('.hea','.mat')
                if os.path.exists(mat): files.append(mat)
    return files

# -------------------- Dataset --------------------
class ECGDataset(Dataset):
    def __init__(self, file_list, seq_len=SEQ_LEN):
        self.files = file_list
        self.seq_len = seq_len

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        mat = loadmat(self.files[idx])['val']  # (12, T)
        # trim o padder
        if mat.shape[1]>=self.seq_len:
            sig = mat[:,:self.seq_len]
        else:
            pad = np.zeros((12,self.seq_len-mat.shape[1]))
            sig = np.concatenate([mat,pad],axis=1)
        # normalización canal a canal
        mean = sig.mean(axis=1,keepdims=True)
        std = sig.std(axis=1,keepdims=True)+1e-6
        sig = (sig-mean)/std
        return torch.tensor(sig,dtype=torch.float32)

# -------------------- VAE --------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12,32,4,2,1), nn.ReLU(),
            nn.Conv1d(32,64,4,2,1), nn.ReLU(),
            nn.Conv1d(64,128,4,2,1), nn.ReLU()
        )
        n = SEQ_LEN//8
        self.fc_mu = nn.Linear(128*n, latent_dim)
        self.fc_logvar = nn.Linear(128*n, latent_dim)

    def forward(self,x):
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        n = SEQ_LEN//8
        self.fc = nn.Linear(latent_dim,128*n)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose1d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose1d(32,12,4,2,1)
        )

    def forward(self,z):
        h = self.fc(z).view(-1,128,SEQ_LEN//8)
        return self.deconv(h)

class VAE(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.enc=Encoder(latent_dim)
        self.dec=Decoder(latent_dim)
    def reparam(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        return mu+std*torch.randn_like(std)
    def forward(self,x):
        mu,lv=self.enc(x)
        z=self.reparam(mu,lv)
        return self.dec(z), mu, lv

# -------------------- Loss --------------------
def loss_fn(recon,x,mu,lv):
    mse = nn.functional.mse_loss(recon,x,reduction='mean')
    mae = nn.functional.l1_loss(recon,x,reduction='mean')
    kld = -0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
    return mse+kld, mse, mae, kld

# -------------------- Training/Evaluación --------------------
def train_val_test_split(dataset):
    n=len(dataset)
    n_test=int(n*TEST_SPLIT)
    n_val=int(n*VAL_SPLIT)
    n_train=n-n_test-n_val
    return random_split(dataset,[n_train,n_val,n_test])

def run():
    files = get_normal_file_list()
    print(f"Archivos normales: {len(files)}")
    ds=ECGDataset(files)
    train_ds,val_ds,test_ds = train_val_test_split(ds)
    print(f"Split -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    loaders={
        'train':DataLoader(train_ds,BATCH_SIZE,shuffle=True),
        'val':DataLoader(val_ds,BATCH_SIZE),
        'test':DataLoader(test_ds,BATCH_SIZE)
    }
    model=VAE(LATENT_DIM).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

    # Training loop
    for ep in range(1,EPOCHS+1):
        model.train()
        tot_loss=0
        for x in loaders['train']:
            x=x.to(DEVICE)
            recon,mu,lv=model(x)
            loss,mse,mae,kld=loss_fn(recon,x,mu,lv)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss+=loss.item()
        avg=tot_loss/len(loaders['train'])
        # Validation
        model.eval()
        with torch.no_grad():
            metrics={'mse':0,'mae':0,'kld':0}
            for x in loaders['val']:
                x=x.to(DEVICE)
                recon,mu,lv=model(x)
                _,mse_i,mae_i,kld_i=loss_fn(recon,x,mu,lv)
                metrics['mse']+=mse_i.item(); metrics['mae']+=mae_i.item(); metrics['kld']+=kld_i.item()
            for k in metrics: metrics[k]/=len(loaders['val'])
        print(f"Ep {ep}/{EPOCHS} Train Loss {avg:.4f} | Val MSE {metrics['mse']:.4f} MAE {metrics['mae']:.4f}")

    # Test final
    model.eval()
    with torch.no_grad():
        mae_t=0
        for x in loaders['test']:
            x=x.to(DEVICE)
            recon,mu,lv=model(x)
            mae_t+=nn.functional.l1_loss(recon,x,reduction='mean').item()
        mae_t/=len(loaders['test'])
    print(f"Test MAE: {mae_t:.4f}")

if __name__=='__main__':
    run()
