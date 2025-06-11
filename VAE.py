# train_ecg_vae.py
"""
Entrena un VAE convolucional sobre señales ECG normales (PTB-XL y Chapman).
Reserva un subconjunto de registros normales para validación y otro para test.
Preprocesa (bandpass 0.5-40Hz, min-max a [-1,1]) y entrena sólo con train normales.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
import wfdb
from wfdb import rdrecord

# ----- Preprocesamiento -----
from scipy.signal import butter, filtfilt
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, fs, lowcut=0.5, highcut=40.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def normalize_zscore(data):
    """Z-score normalization per channel."""
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)
    return (data - mu) / (sigma + 1e-6)

# ----- Dataset ECG -----
class ECGDataset(Dataset):
    def __init__(self, ptb_dir, chap_dir, sample_length=5000, normal_only=True):
        self.samples = []
        # PTB-XL normales
        csv = next((os.path.join(r,f) for r,_,fs in os.walk(ptb_dir) for f in fs if f=='ptbxl_database.csv'), None)
        df = pd.read_csv(csv)
        df_norm = df[df['scp_codes'].str.contains('NORM', na=False)].reset_index(drop=True)
        for _, row in df_norm.iterrows():
            self.samples.append(('ptb', row['filename_lr']))
        # Chapman normales
        for root,_,files in os.walk(chap_dir):
            for f in files:
                if f.endswith('.hea') and 'Zone' not in f:
                    with open(os.path.join(root,f)) as h:
                        for line in h:
                            if line.startswith('#Dx:') and '426177001' in line:
                                rec_id = os.path.splitext(f)[0]
                                self.samples.append(('chap', rec_id))
                                break
        self.ptb_dir=ptb_dir; self.chap_dir=chap_dir; self.sample_length=sample_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, identifier = self.samples[idx]
        # cargar señal
        if src=='ptb':
            fn = identifier
            hea = os.path.basename(fn)+'.hea'
            hea_path = [os.path.join(r,hea) for r,_,fs in os.walk(self.ptb_dir) if hea in fs]
            rec = rdrecord(os.path.splitext(hea_path[0])[0])
            sig = rec.p_signal.T; fs=rec.fs
        else:
            rec_id = identifier
            mat = loadmat(os.path.join(self.chap_dir, rec_id+'.mat'))
            sig = mat['val']; fs=500
        # segmentar
        if sig.shape[1]>=self.sample_length:
            start=np.random.randint(0, sig.shape[1]-self.sample_length+1)
            seg=sig[:, start:start+self.sample_length]
        else:
            pad=np.zeros((sig.shape[0], self.sample_length)); pad[:,:sig.shape[1]]=sig; seg=pad
        # preproc
        seg = apply_bandpass(seg, fs)
        # seg = normalize_minmax(seg)
        seg = normalize_zscore(seg)
        return torch.from_numpy(seg).float()

# ----- VAE Model -----
class ECG_VAE(nn.Module):
    def __init__(self, in_channels=12, z_dim=16, seq_len=5000):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels,32,7,2,3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,64,7,2,3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,7,2,3), nn.BatchNorm1d(128), nn.ReLU(),
        )
        t_prime = seq_len//8
        self.flatten = nn.Flatten()
        self.fc_mu  = nn.Linear(128*t_prime, z_dim)
        self.fc_log = nn.Linear(128*t_prime, z_dim)
        self.fc_dec = nn.Linear(z_dim, 128*t_prime)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(128,64,7,2,3,1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64,32,7,2,3,1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32,in_channels,7,2,3,1),
        )

    def reparam(self, mu, logv):
        std=(0.5*logv).exp(); eps=torch.randn_like(std)
        return mu+eps*std

    def forward(self,x):
        h=self.enc(x); h=self.flatten(h)
        mu=self.fc_mu(h); logv=self.fc_log(h)
        z=self.reparam(mu, logv)
        h2=self.fc_dec(z).view(x.size(0),128,-1)
        x_hat=self.dec(h2)
        return x_hat, mu, logv

# ----- Loss & Training -----
def loss_fn(x,x_hat,mu,logv,beta=1.0):
    recon=torch.abs(x_hat-x).mean(); kld=-0.5*torch.mean(1+logv-mu.pow(2)-logv.exp())
    return recon+beta*kld, recon, kld

if __name__=='__main__':
    PTB_DIR='data/ptb-xl'; CH_DIR='data/ChapmanShaoxing'
    batch=16; epochs=15; lr=1e-3; z_dim=16; seq_len=5000
    # dataset completo
    ds=ECGDataset(PTB_DIR,CH_DIR,seq_len)
    # split: 80% train, 10% val, 10% test
    n=len(ds); n_test=int(0.1*n); n_val=int(0.1*n); n_train=n-n_val-n_test
    ds_train, ds_val, ds_test = random_split(ds, [n_train,n_val,n_test])
    loader_tr=DataLoader(ds_train,batch,shuffle=True,num_workers=4)
    loader_val=DataLoader(ds_val,batch,shuffle=False,num_workers=2)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae=ECG_VAE(z_dim=z_dim, seq_len=seq_len).to(device)
    opt=optim.Adam(vae.parameters(), lr=lr)

    best_val=float('inf')
    for ep in range(1,epochs+1):
        vae.train(); train_loss=0
        for xb in loader_tr:
            xb=xb.to(device)
            xh, mu, lv=vae(xb)
            loss,_,_ = loss_fn(xb,xh,mu,lv)
            opt.zero_grad(); loss.backward();
            nn.utils.clip_grad_norm_(vae.parameters(),1.0)
            opt.step(); train_loss+=loss.item()
        train_loss/=len(loader_tr)
        # validación
        vae.eval(); val_loss=0
        with torch.no_grad():
            for xb in loader_val:
                xb=xb.to(device)
                xh,mu,lv=vae(xb)
                loss,_,_=loss_fn(xb,xh,mu,lv)
                val_loss+=loss.item()
        val_loss/=len(loader_val)
        print(f"Epoch {ep}/{epochs} - Train L: {train_loss:.4f}  Val L: {val_loss:.4f}")
        if val_loss<best_val:
            best_val=val_loss
            torch.save(vae.state_dict(),'ecg_vae_best.pth')
    print("Entrenamiento completo. Mejor modelo en ecg_vae_best.pth")
    # test: MAE sobre ds_test
    from torch.utils.data import DataLoader
    loader_te=DataLoader(ds_test,batch,shuffle=False)
    tot_mae=[]
    vae.load_state_dict(torch.load('ecg_vae_best.pth',map_location=device))
    vae.eval()
    with torch.no_grad():
        for xb in loader_te:
            xb=xb.to(device)
            xh,_,_=vae(xb)
            mae=torch.abs(xh-xb).mean(dim=(1,2)).cpu().numpy()
            tot_mae.extend(mae.tolist())
    print(f"MAE promedio normales test: {np.mean(tot_mae):.4f}")
    # ds_test contiene sólo normales. Anómalos se evalúan en script aparte.
