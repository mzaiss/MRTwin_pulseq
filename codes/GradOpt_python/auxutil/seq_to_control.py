# -*- coding: utf-8 -*-

# reads .seq files from folder and writes in control file
import os
import numpy as np
import scipy
import scipy.io


basepath = "//141.67.249.47/MRtransfer/pulseq_zero/sequences"
dp_control = "//141.67.249.47//MRtransfer//pulseq_zero//control"
control_filename = "control.txt"


path_seq_files = "//141.67.249.47//MRtransfer//pulseq_zero//sequences//seq190927//e24_tgtRARE_tskRARE96_lowSAR_highpass_scaler_brainphantom_sardiv100"
#path_seq_files = "//141.67.249.47//MRtransfer//pulseq_zero//sequences//seq190925//e24_tgtRARE_tskRARE64_lowSAR_phantom_supervised"
all_seq_files = os.listdir(path_seq_files)
all_seq_files = [os.path.join(path_seq_files, filename) for filename in all_seq_files if filename[-3:]=="seq" and  "target" not in filename]
all_seq_files.insert(0,os.path.join(path_seq_files,'target.seq')) #set target.seq at beginning of the list

with open(os.path.join(dp_control,control_filename),"r") as f:
    control_lines = f.readlines()
    
control_lines = [l.strip() for l in control_lines[:-1]]

control_lines.extend(all_seq_files)
    
control_lines = [l.replace('\\','//')+"\n" for l in control_lines]
control_lines.append("wait")

with open(os.path.join(dp_control,control_filename),"w") as f:
    f.writelines(control_lines)