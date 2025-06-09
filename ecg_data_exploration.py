import os
import pandas as pd
import numpy as np
import wfdb
from wfdb import rdrecord
from scipy.io import loadmat

# ==============================================================================
# Configuration: Rutas a los directorios descomprimidos
# =====================================================================
PTBXL_DIR = 'data/ptb-xl'        
CHAPMAN_DIR = 'data/ChapmanShaoxing'

# ==============================================================================
# Funciones para explorar PTB-XL (WFDB)
# =====================================================================
def explore_ptbxl(ptb_dir):
    """
    Recorre el directorio PTB-XL y devuelve un DataFrame con metadatos de cada registro.
    """
    records = []
    for root, _, files in os.walk(ptb_dir):
        for file in files:
            if file.endswith('.hea') and not file.startswith('._'):
                base = file[:-4]
                hea_path = os.path.join(root, file)
                try:
                    rec = wfdb.rdheader(hea_path)
                    # Extraer etiquetas SCP-ECG
                    scp_codes = rec.comments[-1] if rec.comments else ''
                    records.append({
                        'record_name': base,
                        'path': hea_path,
                        'n_leads': rec.n_sig,
                        'fs': rec.fs,
                        'sig_len': rec.sig_len,
                        'scp_codes': scp_codes
                    })
                except Exception as e:
                    print(f"Error leyendo {hea_path}: {e}")

    return pd.DataFrame(records)


def filter_ptbxl_normal(df):
    """
    Filtra registros normales (superclase 'NORM') en PTB-XL.
    """
    # Cada registro puede tener múltiples códigos separados por ','
    mask = df['scp_codes'].str.contains('NORM', na=False)
    return df[mask].reset_index(drop=True)

# ==============================================================================
# Funciones para explorar Chapman-Shaoxing
# =====================================================================

def parse_chapman_header(hea_path):
    """
    Lee un archivo .hea de Chapman-Shaoxing y extrae metadatos (ritmo, fs, duración).
    """
    meta = {}
    with open(hea_path, 'r') as f:
        line = f.readline().strip().split()
        # Ej: JS00001.mat 16+24 1000/mV 16 0 -254 21756 0 I
        # Última columna contiene la derivación, penúltima el código de segmento
        # Asumimos que el ritmo aparece en comentarios o en el nombre del archivo zone
        # Para este ejemplo, extraemos ritmo desde el nombre del archivo zone (.hea?Zone)
    return meta


def explore_chapman(chap_dir):
    """
    Recorre el directorio Chapman-Shaoxing y devuelve DataFrame con registros y etiquetas.
    """
    data = []
    for root, _, files in os.walk(chap_dir):
        for file in files:
            if file.endswith('.hea') and '.heaZone' not in file:
                base = file.replace('.hea', '')
                hea_path = os.path.join(root, file)
                # Tratamos de abrir solo la primera línea para metadata
                with open(hea_path, 'r') as f:
                    header = f.readline().strip().split()
                # Label: extraer ritmo desde zona (busca archivo .heaZoneXX)
                rhythm = 'Unknown'
                # Escanea zonas
                for zone_file in files:
                    if zone_file.startswith(base) and 'Zone' in zone_file:
                        # Ej: JS00001.heaIII.Zone
                        parts = zone_file.split('.')
                        if len(parts) >= 3:
                            rhythm = parts[2]
                        break
                data.append({
                    'record_name': base,
                    'hea_path': hea_path,
                    'fs': int(header[2].split('/')[0]),
                    'gain': header[2].split('/')[1] if '/' in header[2] else None,
                    'n_leads': int(header[1].split('+')[0]),
                    'rhythm': rhythm
                })
    return pd.DataFrame(data)


def filter_chapman_normal(df):
    """
    Filtra registros con ritmo de "Sinus Rhythm" (SR) en Chapman-Shaoxing.
    """
    mask = df['rhythm'].str.upper().isin(['SR', 'SINUS', 'SINUSRHYTHM'])
    return df[mask].reset_index(drop=True)

# ==============================================================================
# Ejecución de ejemplo
# =====================================================================

if __name__ == '__main__':
    # Explorar PTB-XL
    ptb_meta = explore_ptbxl(PTBXL_DIR)
    print(f'Total PTB-XL records: {len(ptb_meta)}')
    ptb_norm = filter_ptbxl_normal(ptb_meta)
    print(f'PTB-XL registros normales: {len(ptb_norm)}')

    # Explorar Chapman-Shaoxing
    chap_meta = explore_chapman(CHAPMAN_DIR)
    print(f'Total Chapman records: {len(chap_meta)}')
    chap_norm = filter_chapman_normal(chap_meta)
    print(f'Chapman registros normales (SR): {len(chap_norm)}')

    # Mostrar primeras filas
    print(ptb_meta.head())
    print(chap_meta.head())
