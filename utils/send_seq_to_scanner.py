# -*- coding: utf-8 -*-
"""
MRcourse Dropbox Workflow
Upload .seq, wait for .dat, download .dat
"""

import os
import time
import dropbox
import torch
import numpy as np
from .config import ACCESS_TOKEN, DROPBOX_FOLDER

# -----------------------------
# Settings
# -----------------------------
DROPBOX_FOLDER = DROPBOX_FOLDER       # Ordner in Dropbox
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(SCRIPT_DIR, "seq")
os.makedirs(LOCAL_DIR, exist_ok=True)

# -----------------------------
# Helper: load measurement
# -----------------------------
def load_measurement(seq_dat_file: str, adc_length: int, NCoils: int = 20, heuristic_shift: int = 4) -> torch.Tensor:
    data = np.loadtxt(seq_dat_file)
    data = data[:,0] + 1j*data[:,1]
    data = data.reshape([-1, NCoils, adc_length + heuristic_shift])
    signal = data.transpose([0,2,1])[:, :adc_length, :].reshape([-1, NCoils])
    return torch.tensor(signal, dtype=torch.complex64)

# -----------------------------
# Main function
# -----------------------------
def send_seq_to_scanner(seq, seq_file_name, adc_length):
    dbx = dropbox.Dropbox(ACCESS_TOKEN)

    LOCAL_UPLOAD_PATH = os.path.join(LOCAL_DIR, seq_file_name)
    LOCAL_DOWNLOAD_PATH = os.path.join(LOCAL_DIR, seq_file_name + ".dat")

    # Write seq file locally
    seq.write(LOCAL_UPLOAD_PATH)

    # --- Upload .seq ---
    dropbox_path = f"{DROPBOX_FOLDER}/{seq_file_name}"
    with open(LOCAL_UPLOAD_PATH, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
    print(f"Upload successful: {seq_file_name} → {dropbox_path}")

    # --- Wait for .dat ---
    target_file = f"{DROPBOX_FOLDER}/{seq_file_name}.dat"
    print(f"Wait for data '{target_file}'...")

    downloaded = False
    while not downloaded:
        try:
            metadata, res = dbx.files_download(target_file)
            with open(LOCAL_DOWNLOAD_PATH, "wb") as f:
                f.write(res.content)
            print(f"Download finished: {LOCAL_DOWNLOAD_PATH}")
            downloaded = True
        except dropbox.exceptions.ApiError:
            # Datei existiert noch nicht
            time.sleep(5)

    # --- Load measurement ---
    return load_measurement(LOCAL_DOWNLOAD_PATH, adc_length)