import numpy as np
import matplotlib.pyplot as plt
import os, sys
import scipy
import math

from sys import platform
from shutil import copyfile
from core.pulseq_exporter import pulseq_write_GRE
from core.pulseq_exporter import pulseq_write_GRE_DREAM
from core.pulseq_exporter import pulseq_write_RARE
from core.pulseq_exporter import pulseq_write_BSSFP
from core.pulseq_exporter import pulseq_write_EPI

fullpath_seq = r"C:\Users\danghi\Documents\MRzero\q14_tgtT1_tskT1invrec_supervised_seqNNvivo_time_zero.py"

fn_alliter_array = "alliter_arr.npy"
alliter_array = np.load(os.path.join(os.path.join(fullpath_seq, fn_alliter_array)), allow_pickle=True)
alliter_array = alliter_array.item()

iteration_idx = [4999,6079,7159,8239,9329]
iteration_idx = [*(np.linspace(5000,5079,15).astype('int')),*(np.linspace(6080,6159,15).astype('int')),*(np.linspace(7160,7239,15).astype('int')),*(np.linspace(8230,8209,15).astype('int'))]
iteration_idx.extend([4999,6079,7159,8239,9329])
iteration_idx.sort()

for iter_idx in (iteration_idx):
    fn_pulseq = "iter" + str(iter_idx).zfill(6) + ".seq"
    
    flips_numpy = alliter_array['flips'][iter_idx]
    event_time_numpy = alliter_array['event_times'][iter_idx]
    grad_moms_numpy = alliter_array['grad_moms'][iter_idx]
    
    seq_params = flips_numpy, event_time_numpy, grad_moms_numpy
    
    if sequence_class.lower() == "gre":
        pulseq_write_GRE(seq_params, os.path.join(fullpath_seq, fn_pulseq), plot_seq=False)
    elif sequence_class.lower() == "gre_dream":
        pulseq_write_GRE_DREAM(seq_params, os.path.join(fullpath_seq, fn_pulseq), plot_seq=False)
    elif sequence_class.lower() == "rare":
        pulseq_write_RARE(seq_params, os.path.join(fullpath_seq, fn_pulseq), plot_seq=False)
    elif sequence_class.lower() == "bssfp":
        pulseq_write_BSSFP(seq_params, os.path.join(fullpath_seq, fn_pulseq), plot_seq=False)
    elif sequence_class.lower() == "epi":
        pulseq_write_EPI(seq_params, os.path.join(fullpath_seq, fn_pulseq), plot_seq=False)          


spins.set_system(real_phantom_resized)

for iter_idx in (iteration_idx):
    
    flips_numpy = alliter_array['flips'][iter_idx]
    event_time_numpy = alliter_array['event_times'][iter_idx]
    grad_moms_numpy = alliter_array['grad_moms'][iter_idx]
    
    scanner.set_adc_mask(adc_mask=setdevice(adc_mask))
    scanner.init_flip_tensor_holder()      
    scanner.set_flip_tensor_withB1plus(flips)
    # rotate ADC according to excitation phase
    rfsign = ((flips[3,:,0]) < 0).float()
    scanner.set_ADC_rot_tensor(-flips[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific
    scanner.init_gradient_tensor_holder()
    scanner.set_gradient_precession_tensor(grad_moms,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes
 
T1_maps = torch.zeros(len(iteration_idx),32,32)

for i,iter_idx in enumerate(iteration_idx):
    scanner.signal = torch.unsqueeze(torch.from_numpy(alliter_array['all_signals'][iter_idx]),4).float()
    reco_sep = scanner.adjoint_separable()
    reco_all_rep=torch.zeros((extraRep,reco_sep.shape[1],2))
    for j in range(0,extraRep):
        reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)
    reco_test = torch.sqrt((reco_all_rep**2).sum(2))
    scale = torch.max(reco_test)
    reco_test = reco_test / scale
    
    
    fn_NN_paramlist = "alliter_NNparamlist_" + str(iter_idx) + '.pt'
    nmb_hidden_neurons_list = [extraRep,16,32,16,1]
    NN = core.nnreco.VoxelwiseNet(scanner.sz, nmb_hidden_neurons_list, use_gpu=use_gpu, gpu_device=gpu_dev)
    state_dict = torch.load(os.path.join(fullpath_seq, fn_NN_paramlist), map_location=torch.device('cpu'))
    #state_dict = torch.load(os.path.join(opt.get_base_path(experiment_id, today_datestr)[1], 'last_NN.pt'), map_location=torch.device('cpu'))
    NN.load_state_dict(state_dict)
    
    reco_test = reco_test.permute([1,0])
    plt.imshow(tonumpy(reco_test.reshape(32,32,10)[:,:,9]))
    plt.show()
    T1_maps[i,:,:] = NN(reco_test).reshape([sz[0],sz[1]])