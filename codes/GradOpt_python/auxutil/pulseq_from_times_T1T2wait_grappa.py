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
from core.pulseq_exporter import pulseq_write_super


def tonumpy(x):
    return x.detach().cpu().numpy()

# fullpath_seq = r"C:\Users\danghi\Documents\MRzero\q14_tgtT1_tskT1invrec_supervised_seqNNvivo_time_zero.py"
# fullpath_seq = r"\\141.67.249.47\MRTransfer\pulseq_zero\sequences\seq200507\q14_tgtT1_64"
fullpath_seq = r"\\141.67.249.47\MRTransfer\pulseq_zero\sequences\seq200519\q30_tgtT1T2_tskT2prepT1wait_supervised_NN_grappa"


fn_alliter_array = "alliter_arr.npy"
alliter_array = np.load(os.path.join(os.path.join(fullpath_seq, fn_alliter_array)), allow_pickle=True)
alliter_array = alliter_array.item()

# iteration_idx = [4999,6079,7159,8239,9329]
# iteration_idx = [*(np.linspace(5000,5079,15).astype('int')),*(np.linspace(6080,6159,15).astype('int')),*(np.linspace(7160,7239,15).astype('int')),*(np.linspace(8230,8209,15).astype('int'))]
# iteration_idx.extend([4999,6079,7159,8239,9329])
iteration_idx = [0,*np.arange(49,650,50)]
# iteration_idx.sort()

sz = np.array([126,126])
extraRep = 6
NRep = extraRep*sz[1]
T = sz[0] + 10 
sequence_class = 'super'
R=3
Nlines = int(np.ceil(sz[1]/R)*extraRep) # number of acquired lines given undersampling factor R

event_time_numpy = alliter_array['event_times'][iteration_idx]
TE_preps_1 = np.abs(event_time_numpy[:,1,[i*32 for i in range(extraRep)]])
TE_preps_2 = np.abs(event_time_numpy[:,3,[i*32 for i in range(extraRep)]])
TI_waits = np.abs(event_time_numpy[:,5,[i*32 for i in range(extraRep)]])

waitings = np.abs(event_time_numpy[:,-1,[i*32-1 for i in range(extraRep)]])
waitings[:,0] = 2.9*1e-3

scan_time = np.sum(np.abs(event_time_numpy))
print('total scan time: '+str(scan_time))

flips = torch.zeros((T,Nlines,4), dtype=torch.float32)
flips[6,:,0] = 5*np.pi/180  # GRE/FID specific, GRE preparation part 1 : 90 degree excitation 
flips[6,:,3] = 1
# randomize RF phases
measRepStep = Nlines//extraRep
for i in range(0,extraRep):
    flips[0,i*measRepStep,0] = 90*np.pi/180
    flips[0,i*measRepStep,1] = 0*np.pi/180
    flips[2,i*measRepStep,0] = 180*np.pi/180
    flips[2,i*measRepStep,1] = 90*np.pi/180
    flips[4,i*measRepStep,0] = 90*np.pi/180
    flips[4,i*measRepStep,1] = 180*np.pi/180
    flips[6,i*measRepStep:(i+1)*measRepStep,1] = torch.tensor(scanner.phase_cycler[:(measRepStep)]).float()*np.pi/180

grad_moms = torch.zeros((T,Nlines,2), dtype=torch.float32)
meas_indices=np.zeros((extraRep,measRepStep))
for i in range(0,extraRep):
    meas_indices[i,:] = np.arange(i*measRepStep,(i+1)*measRepStep)

for j in range(0,extraRep):
    # third action after second 90deg pulse (chrusher)
    grad_moms[5,j*measRepStep] = np.float(sz[0])*2.0#1e-2

    #  measurement
    grad_moms[7,meas_indices[j,:] ,0] = -sz[0]/2         # GRE/FID specific, rewinder in second event block
    grad_moms[7,meas_indices[j,:] ,1] = torch.linspace(-int(sz[1]/2),int(sz[1]/2-1),int(meas_indices[j,:].size))  # phase encoding blip in second event block
    grad_moms[8:-2,meas_indices[j,:] ,0] = torch.ones(int(sz[0])).view(int(sz[0]),1).repeat([1,meas_indices[j,:].size]) # ADC open, readout, freq encoding
    grad_moms[-2,meas_indices[j,:] ,0] = torch.ones(1)*sz[0]*2  # GRE/FID specific, SPOILER
    grad_moms[-2,meas_indices[j,:] ,1] = -grad_moms[1,meas_indices[j,:] ,1]      # GRE/FID specific, yblip rewinder
    
    odd_index = np.arange(int(Nlines/extraRep/2),Nlines/extraRep)
    even_index = np.flip(np.arange(0,int(Nlines/extraRep/2)))
    idx = np.ravel((odd_index,even_index),'F')
    temp_mom = grad_moms[7,meas_indices[j,:] ,1]
    grad_moms[7,meas_indices[j,:] ,1] = temp_mom[idx]
    grad_moms[-2,meas_indices[j,:] ,1] = -grad_moms[7,meas_indices[j,:] ,1]


for iter_idx in range(len(iteration_idx)):
    fn_pulseq = "iter" + str(iter_idx).zfill(6) + ".seq"
    
    event_time = torch.from_numpy(0.08*1e-3*np.ones((T,Nlines))).float()
    TE_prep1 = torch.from_numpy(TE_preps_1[iter_idx])
    TE_prep2 = torch.from_numpy(TE_preps_2[iter_idx])
    TI_wait = torch.from_numpy(TI_waits[iter_idx])+1e-3
    waiting = torch.from_numpy(waitings[iter_idx])
    
    event_time[5,:] =  5e-3
    event_time[6,:] =  2e-3
    event_time[7,:] =  5*1e-3   # for 96
    event_time[-2,:] = 5*1e-3
    event_time[-1,:] = 2.9*1e-3
    
    # first action
    for i in range(0,extraRep):
        event_time[0,i*measRepStep] =  2e-3
        event_time[2,i*measRepStep] =  2e-3
        event_time[4,i*measRepStep] =  2e-3
        event_time[1,i*measRepStep] = TE_prep1[i]
        event_time[3,i*measRepStep] = TE_prep2[i]
        event_time[5,i*measRepStep] = TI_wait[i]
        if i>0:
            event_time[-1,i*measRepStep-1] = waiting[i]       #delay after readout before next inversion ( only after the first event played out)
    event_time = setdevice(event_time)    
 
    
    flips_numpy = tonumpy(flips)
    event_time_numpy = tonumpy(event_time)
    grad_moms_numpy = tonumpy(grad_moms)
    adc_mask = torch.from_numpy(np.ones((T,1))).float()
    adc_mask[:8]  = 0
    adc_mask[-2:] = 0
    scanner.set_adc_mask(adc_mask=setdevice(adc_mask))
    
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
    elif sequence_class.lower() == "super":
        seq_params = flips_numpy, event_time_numpy, grad_moms_numpy, adc_mask.detach().cpu().numpy()
        pulseq_write_super(seq_params, os.path.join(fullpath_seq, fn_pulseq), plot_seq=False)
        

fn_pulseq = "full.seq"

event_time = torch.from_numpy(0.08*1e-3*np.ones((T,Nlines))).float()
TE_prep = torch.tensor(np.logspace(np.log10(0.01),np.log10(1.2),extraRep))
TI_wait = torch.tensor(np.logspace(np.log10(0.01),np.log10(2),extraRep))

event_time[5,:] =  5e-3
event_time[6,:] =  2e-3
event_time[7,:] =  5*1e-3   # for 96
event_time[-2,:] = 5*1e-3
event_time[-1,:] = 2.9*1e-3

# first action
for i in range(0,extraRep):
    event_time[0,i*measRepStep] =  2e-3
    event_time[2,i*measRepStep] =  2e-3
    event_time[4,i*measRepStep] =  2e-3
    event_time[1,i*measRepStep] = TE_prep[i]/2
    event_time[3,i*measRepStep] = TE_prep[i]/2
    event_time[5,i*measRepStep] = TI_wait[i]/2
    if i>0:
        event_time[-1,i*measRepStep-1] = 8       #delay after readout before next inversion ( only after the first event played out)
event_time = setdevice(event_time)    
 

flips_numpy = tonumpy(flips)
event_time_numpy = tonumpy(event_time)
grad_moms_numpy = tonumpy(grad_moms)
adc_mask = torch.from_numpy(np.ones((T,1))).float()
adc_mask[:8]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

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
elif sequence_class.lower() == "super":
    seq_params = flips_numpy, event_time_numpy, grad_moms_numpy, adc_mask.detach().cpu().numpy()
    pulseq_write_super(seq_params, os.path.join(fullpath_seq, fn_pulseq), plot_seq=False)        

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