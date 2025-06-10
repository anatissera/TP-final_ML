import os
import pandas as pd
import numpy as np
import wfdb
from wfdb import rdrecord
from scipy.io import loadmat


# ---------- Utilidades ----------
def list_directory(start_dir, depth=1):
    print(f"\nEstructura de {start_dir} (profundidad {depth}):")
    for root, dirs, files in os.walk(start_dir):
        level = root.replace(start_dir, '').count(os.sep)
        if level > depth: continue
        indent = '  ' * level
        # print(f"{indent}{os.path.basename(root)}/")
        # for f in files:
            # print(f"{indent}- {f}")

# ---------- PTB-XL ----------
def load_ptbxl_metadata(ptb_dir):
    # Carga CSV oficial ptbxl_database.csv
    csv = next((os.path.join(r,f) for r,_,fs in os.walk(ptb_dir)
                for f in fs if f=='ptbxl_database.csv'), None)
    if csv is None:
        raise FileNotFoundError("No se encontró ptbxl_database.csv en PTBXL_DIR")
    df = pd.read_csv(csv)
    return df[['ecg_id','filename_lr','scp_codes','age','sex']]

def filter_ptbxl_normal(df):
    return df[df['scp_codes'].str.contains('NORM', na=False)].reset_index(drop=True)

# ---------- Chapman-Shaoxing (parseando .hea) ----------
def parse_chapman_header(hea_path):
    """
    Extrae metadata completa del header .hea:
      - id, n_leads, fs, total_samples
      - first/last values y nombre de cada lead
      - age, sex, dx (diagnóstico)
    """
    with open(hea_path,'r') as f:
        lines = [l.strip() for l in f.readlines()]
    # Línea 0: JS00001.mat 16+24 1000/mV 16 0 -254 21756 0 I
    parts = lines[0].split()
    record_id = parts[0].replace('.mat','')
    n_leads = parts[1]
    fs = parts[2]
    total_samples = parts[3]
    # A cada lead corresponde una línea siguiente
    leads = {}
    for i in range(1,13):
        row = lines[i].split()
        # índices 5 y 6: primer y último valor crudo, 7: nombre de derivación
        leads[row[7]] = {'first': row[5], 'last': row[6]}
    # metadata demográficas en líneas marcadas
    age = next((l.split()[1] for l in lines if l.startswith('#Age:')), None)
    sex = next((l.split()[1] for l in lines if l.startswith('#Sex:')), None)
    dx  = next((l.split()[1] for l in lines if l.startswith('#Dx:')), None)
    return {
        'record': record_id,
        'n_leads': n_leads,
        'fs': fs,
        'samples': total_samples,
        'age': age,
        'sex': sex,
        'diagnosis': dx,
        **{f'{lead}_first': v['first'] for lead,v in leads.items()},
        **{f'{lead}_last': v['last'] for lead,v in leads.items()}
    }

def load_chapman_metadata(path):
    """
    Recorre todos los .hea y construye un DataFrame con metadata parseada.
    """
    recs = []
    for root,_,files in os.walk(path):
        for file in files:
            if file.endswith('.hea') and 'Zone' not in file:
                hea = os.path.join(root,file)
                recs.append(parse_chapman_header(hea))
    return pd.DataFrame(recs)

def filter_chapman_normal(df):
    # Filtra diagnosis == 'SR' (Sinus Rhythm)
    
    # # Buscar registros con solo el código 426177001 (ritmo sinusal normal)
    # return df[df['diagnosis'] == '426177001'].reset_index(drop=True)
    
    # Incluye cualquier fila donde uno de los diagnósticos sea 426177001
    return df[df['diagnosis'].str.contains('426177001', na=False)].reset_index(drop=True)


# ---------- Main ----------
if __name__=='__main__':
    PTBXL_DIR = 'data/ptb-xl'
    CHAPMAN_DIR = 'data/ChapmanShaoxing'

    # PTB-XL
    print("# PTB-XL #")
    list_directory(PTBXL_DIR)
    ptb = load_ptbxl_metadata(PTBXL_DIR)
    print(f"Total PTB-XL: {len(ptb)} registros, normales: {len(filter_ptbxl_normal(ptb))}")

    # Chapman
    print("\n# Chapman-Shaoxing #")
    list_directory(CHAPMAN_DIR)
    chap = load_chapman_metadata(CHAPMAN_DIR)
    print(f"Total Chapman: {len(chap)} registros, normales (SR): {len(filter_chapman_normal(chap))}")

    # Ejemplo load señal
    if not chap.empty:
        mat_file = chap.iloc[0]['record'] + '.mat'
        path = os.path.join(CHAPMAN_DIR, mat_file)
        data = loadmat(path)['val'] if os.path.exists(path) else None
        print(f"Señal {chap.iloc[0]['record']} cargada: {None if data is None else data.shape}")
