# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import torch
from importlib import reload


module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import core.nnreco; reload(core.nnreco)


def load_NN(fullpath_seq,extraRep = 10):
    
    nmb_hidden_neurons_list = [extraRep,16,32,16,1]
    NN = core.nnreco.VoxelwiseNet(None, nmb_hidden_neurons_list, use_gpu=False)
    state_dict = torch.load(os.path.join(fullpath_seq, 'NN_last.pt'), map_location=torch.device('cpu'))
    NN.load_state_dict(state_dict)
    return NN