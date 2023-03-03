#!/bin/bash

git clone -b mr0-core https://github.com/mzaiss/MRTwin_pulseq.git

module load python3/anaconda-2022.05
pip install MRTwin_pulseq/data/MRzeroCore-0.1.0-cp39-cp39-manylinux_2_28_x86_64.whl --force-reinstall
pip install pypulseq
