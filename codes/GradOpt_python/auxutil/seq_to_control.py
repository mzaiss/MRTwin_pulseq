# -*- coding: utf-8 -*-

# reads .seq files from folder and writes in control file
import os
import sys
import numpy as np
import scipy
import scipy.io
import time


control_filename = "control.txt"

# FAU FAu FAU FAU
basepath = "//141.67.249.47/MRtransfer/pulseq_zero/sequences"
dp_control = "//141.67.249.47//MRtransfer//pulseq_zero//control"
# path_seq_files = "//141.67.249.47//MRtransfer//pulseq_zero//sequences//seq190927//e24_tgtRARE_tskRARE96_lowSAR_highpass_scaler_brainphantom_sardiv100"
# path_seq_files = str(sys.argv[1])
path_seq_files = "//141.67.249.47//MRtransfer//pulseq_zero//sequences//seq190925//e24_tgtRARE_tskRARE64_lowSAR_phantom_supervised"
path_seq_files = r"\\141.67.249.47\MRTransfer\pulseq_zero\sequences\seq200519\q30_tgtT1T2_tskT2prepT1wait_supervised_NN_grappa"

# MPI MPI MPI MPIMPI MPIMPI MPIMPI MPIMPI MPIMPI MPI
#path_seq_files ="//mrz3t/Upload/CEST_seq/pulseq_zero//sequences//seq191014//p06_tgtGRESP_tskFLASH_FA_G_48_lowsar_supervised2px_adaptive_frelax"
#basepath = "//mrz3t/Upload/CEST_seq/pulseq_zero//sequences"
#dp_control = "//mrz3t/Upload/CEST_seq/pulseq_zero//control"
#control_filename = "control.txt"

all_seq_files = os.listdir(path_seq_files)
meas_target = "target.seq" in all_seq_files
all_seq_files = [os.path.join(path_seq_files, filename) for filename in all_seq_files if filename[-3:]=="seq" and  "target" not in filename]
if meas_target:
    all_seq_files.insert(0,os.path.join(path_seq_files,'target.seq')) #set target.seq at beginning of the list
all_seq_files = [l.replace(r'\\',r'//') for l in all_seq_files]
all_seq_files = [l.replace('\\','//')+"\n" for l in all_seq_files]


with open(os.path.join(dp_control,control_filename),"r") as f:
    control_lines = f.readlines()
    
control_lines = control_lines[:-1]

for seq in all_seq_files:
    if seq not in control_lines:
        control_lines.append(seq)
     
#control_lines.extend(all_seq_files)
    
#control_lines = [l.replace('\\','//')+"\n" for l in control_lines]
control_lines.append("wait")

with open(os.path.join(dp_control,control_filename),"w") as f:
    f.writelines(control_lines)

time.sleep(2)