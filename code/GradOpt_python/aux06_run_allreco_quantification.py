import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def tonumpy(x):
    return x.detach().cpu().numpy()

def signal_T1 (TI, S_0, T1):
    return np.abs(S_0 * (1 - 2*np.exp(-TI/T1)))

def signal2_T1 (TI, S_0, T1, Z_i):
    return np.abs(S_0 - (S_0 - Z_i)*np.exp(-TI/T1))

def signal3_T1 (TI, S_0, T1):
    return np.abs(S_0 * (1 - 2*np.exp(-TI/T1)+np.exp(-0.0152 / T1)))

def quantify_T1 (TI, S, p0):
    popt, pcov = curve_fit(signal2_T1, TI, S, p0 = p0, maxfev=1000000)#, bounds=([S[-1]/2,0.5,-S[-1]*2],[S[-1]*2,5,S[-1]]))
    return popt


fullpath_seq = r"C:\Users\danghi\Documents\MRzero\q14_tgtT1_tskT1invrec_supervised_seqNNvivo_short_TI.py"
#fullpath_seq = r"C:\Users\danghi\Documents\MRzero\q14_tgtT1_tskT1invrec_supervised_seqNNvivo_time_zero.py"


fn_alliter_array = "alliter_arr.npy"
alliter_array = np.load(os.path.join(os.path.join(fullpath_seq, fn_alliter_array)), allow_pickle=True)
alliter_array = alliter_array.item()

all_event_times = alliter_array['event_times']
all_TIs = all_event_times[:,2,[i*measRepStep for i in range(extraRep) ]]
all_waitings = all_event_times[:,-1,[i*measRepStep-1 for i in range(extraRep) ]]

# Target
fn_NN_paramlist = "alliter_NNparamlist_" + str(5001) + '.pt'
nmb_hidden_neurons_list = [extraRep,16,32,16,1]
NN = core.nnreco.VoxelwiseNet(scanner.sz, nmb_hidden_neurons_list, use_gpu=use_gpu, gpu_device=gpu_dev)
state_dict = torch.load(os.path.join(fullpath_seq, fn_NN_paramlist), map_location=torch.device('cpu'))

scanner.get_signal_from_real_system(experiment_id,today_datestr)
reco_sep = scanner.adjoint_separable()

reco_all_rep=torch.zeros((extraRep,reco_sep.shape[1],2))
for j in range(0,extraRep):
    reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)

reco_test = torch.sqrt((reco_all_rep**2).sum(2))
scale = torch.max(reco_test)
reco_test = reco_test / scale

S = reco_test.reshape(10,32,32)
S = tonumpy(S)

reco_test = reco_test.permute([1,0])
T1_map_CNN = NN(reco_test).reshape([sz[0],sz[1]])


xdata = all_TIs[0,:]
T1_map = np.zeros((sz[0],sz[1]))

for l in range(sz[0]):
    for m in range(sz[1]):
        try:
            popt = quantify_T1(xdata[0:], S[0:,l,m], p0=[S[-1,l,m],1,-S[-1,l,m]])
            #popt = quantify_T1(xdata[0:], S[0:,l,m], p0=None)
            T1_map[l,m] = popt[1]
        except:
            T1_map[l,m] = 0
            
plt.imshow(tonumpy(T1_map_CNN))
plt.colorbar()
plt.clim(-0.03,-0.05)   
plt.imshow(T1_map)
plt.colorbar()  
plt.clim(0,7)        

# Last iter

fn_NN_paramlist = "alliter_NNparamlist_" + str(10357) + '.pt'
nmb_hidden_neurons_list = [extraRep,16,32,16,1]
NN = core.nnreco.VoxelwiseNet(scanner.sz, nmb_hidden_neurons_list, use_gpu=use_gpu, gpu_device=gpu_dev)
state_dict = torch.load(os.path.join(fullpath_seq, fn_NN_paramlist), map_location=torch.device('cpu'))

scanner.get_signal_from_real_system(experiment_id,today_datestr,jobtype="lastiter")
reco_sep = scanner.adjoint_separable()

reco_all_rep=torch.zeros((extraRep,reco_sep.shape[1],2))
for j in range(0,extraRep):
    reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)

reco_test = torch.sqrt((reco_all_rep**2).sum(2))
scale = torch.max(reco_test)
reco_test = reco_test / scale

S = reco_test.reshape(10,32,32)
S = tonumpy(S)

reco_test = reco_test.permute([1,0])
T1_map_CNN = NN(reco_test).reshape([sz[0],sz[1]])


xdata = all_TIs[-1,:]
T1_map = np.zeros((sz[0],sz[1]))

for l in range(sz[0]):
    for m in range(sz[1]):
        try:
            popt = quantify_T1(np.abs(xdata[0:]), S[0:,l,m], p0=[S[-1,l,m],1,-S[-1,l,m]])
            #popt = quantify_T1(xdata[0:], S[0:,l,m], p0=None)
            T1_map[l,m] = popt[1]
        except:
            T1_map[l,m] = 0
            
plt.imshow(tonumpy(T1_map_CNN))
plt.colorbar()
plt.clim(0,4)   
plt.imshow(T1_map)
plt.colorbar()  
plt.clim(0,4)

signals = plt.plot(reco_test)
plt.legend(signals,[i for i in range (1,10)])

x,y = 11,17 #(R,C)
points = S[:,x,y]
plt.plot(np.abs(xdata),points,'x')

popt = quantify_T1(xdata, points, p0=[S[-1,x,y],1,-S[-1,x,y]])#, p0 = [np.flip(real_phantom_resized[:,:,1].transpose(),(0,1))[x,y],0.1,-0.1])
#popt = quantify_T1(np.abs(reduced_xdata), points, p0=None)
plt.plot(np.abs(xdata), points, '.b', marker='.')
plt.plot(np.abs(xdata), signal2_T1(np.abs(reduced_xdata), *popt), '-r', label='fit: S_0=%5.3f, T1=%5.3f, Z_i=%5.3f' % tuple(popt))
plt.legend()
plt.show()

(grey_roi_x,grey_roi_y) = (list(range(14,18)),list(range(10,13)))
(white_roi_x,white_roi_y) = (list(range(15,21)),list(range(5,10)))
(liquor_roi_x,liquor_roi_y) = (list(range(12,15)),list(range(14,16)))

import matplotlib.patches as patches
def rectangle(x_range,y_range):
    return patches.Rectangle((x_range[0],y_range[0]),x_range[-1]-x_range[0],y_range[-1]-y_range[0],linewidth=1,edgecolor='r',facecolor='none')

def plot_rectangle(im, x, y, lim = (0,4)):
    plt.imshow(im)
    plt.clim(lim)
    ax = plt.gca()
    rect = rectangle(x,y)
    ax.add_patch(rect)
    plt.show()


plot_rectangle(T1_map, grey_roi_x, grey_roi_y)
plot_rectangle(T1_map, white_roi_x, white_roi_y)
plot_rectangle(T1_map, liquor_roi_x, liquor_roi_y)


real_phantom = scipy.io.loadmat('../../data/numerical_brain_cropped.mat')['cropped_brain']
real_phantom_resized = np.zeros((sz[0],sz[1],5), dtype=np.float32)
for i in range(5):
    t = cv2.resize(real_phantom[:,:,i], dsize=(sz[0],sz[1]), interpolation=cv2.INTER_CUBIC)
    if i == 0:
        t[t < 0] = 0
    elif i == 1 or i == 2:
        t[t < cutoff] = cutoff
        
    real_phantom_resized[:,:,i] = t
    
plt.imshow(np.flip(real_phantom_resized[:,:,1].transpose(),(0,1)))
spins.set_system(real_phantom_resized)
scanner.forward_sparse_fast(spins, event_time,kill_transverse=kill_transverse)
reco_sep = scanner.adjoint_separable()

reco_all_rep=torch.zeros((extraRep,reco_sep.shape[1],2))
for j in range(0,extraRep):
    reco_all_rep[j,:,:] = reco_sep[meas_indices[j,:],:,:].sum(0)
    
scale = torch.max(tomag_torch(reco_all_rep))
phantom_signal = tomag_torch(reco_all_rep)/scale
NN_phantom = NN(phantom_signal.permute(1,0)).reshape([sz[0],sz[1]])
plt.imshow(tonumpy(NN_phantom))
plt.clim(0,2)
plt.colorbar()


phantom_signal = tonumpy(phantom_signal)
plt.imshow(phantom_signal[0,:].reshape(32,32))


(Pgrey_roi_x,Pgrey_roi_y) = (list(range(9,13)),list(range(11,15)))
(Pwhite_roi_x,Pwhite_roi_y) = (list(range(5,8)),list(range(11,15)))
(Pliquor_roi_x,Pliquor_roi_y) = (list(range(14,17)),list(range(10,12)))

plot_rectangle(phantom_signal[0,:].reshape(32,32), Pgrey_roi_x, Pgrey_roi_y)
plot_rectangle(phantom_signal[0,:].reshape(32,32), Pwhite_roi_x, Pwhite_roi_y)
plot_rectangle(phantom_signal[0,:].reshape(32,32), Pliquor_roi_x, Pliquor_roi_y)

phantom_signal_im = phantom_signal.reshape(10,32,32)

def plot_roi_signals (signal, x_list,y_list):
    x,y = np.meshgrid(x_list,y_list)
    reduced_signals = signal[:,y,x].reshape(10,-1)
    plt.plot(np.abs(xdata),reduced_signals,'.')
    plt.show()

def roi_stats (signal, x_list,y_list):
    x,y = np.meshgrid(x_list,y_list)
    reduced_signals = signal[:,y,x].reshape(10,-1)
    mean = np.mean(reduced_signals, axis=1)
    std = np.std(reduced_signals, axis=1)
    return (mean,std)
#Phantom
plot_roi_signals(phantom_signal_im,Pgrey_roi_x,Pgrey_roi_y) 
mean_Pgrey,std_Pgrey = roi_stats (phantom_signal_im, Pgrey_roi_x,Pgrey_roi_y)
plot_roi_signals(phantom_signal_im,Pwhite_roi_x,Pwhite_roi_y) 
mean_Pwhite,std_Pwhite = roi_stats (phantom_signal_im, Pwhite_roi_x,Pwhite_roi_y)
plot_roi_signals(phantom_signal_im,Pliquor_roi_x,Pliquor_roi_y) 
mean_Pliquor,std_Pliquor = roi_stats (phantom_signal_im, Pliquor_roi_x,Pliquor_roi_y)

plt.errorbar(np.abs(xdata),mean_Pgrey,std_Pgrey,ls = '',marker='x', capsize=3);plt.show()
plt.errorbar(np.abs(xdata),mean_Pwhite,std_Pwhite,ls = '',marker='x', capsize=3);plt.show()
plt.errorbar(np.abs(xdata),mean_Pliquor,std_Pliquor,ls = '',marker='x', capsize=3);plt.show()

#in vivo
vivo_signal = tonumpy(reco_test.permute(1,0)).reshape(10,32,32)
plot_roi_signals(vivo_signal,grey_roi_x,grey_roi_y) 
mean_grey,std_grey = roi_stats (vivo_signal, grey_roi_x,grey_roi_y)
plot_roi_signals(vivo_signal,white_roi_x,white_roi_y) 
mean_white,std_white = roi_stats (vivo_signal, white_roi_x,white_roi_y)
plot_roi_signals(vivo_signal,liquor_roi_x,liquor_roi_y) 
mean_liquor,std_liquor = roi_stats (vivo_signal, liquor_roi_x,liquor_roi_y)

plt.errorbar(np.abs(xdata),mean_grey,std_grey,ls = '',marker='x', capsize=3);plt.show()
plt.errorbar(np.abs(xdata),mean_white,std_white,ls = '',marker='x', capsize=3);plt.show()
plt.errorbar(np.abs(xdata),mean_liquor,std_liquor,ls = '',marker='x', capsize=3);plt.show()

x,y = 11,15
plt.plot(phantom_signal_im[:,x,y],'.')

plt.imshow(phantom_signal_im[0,:,:])
plt.clim(0,2)
def plot_all_stacks(im_stack):
    plt.subplot(341)
    plt.imshow(im_stack[0,:,:])
    plt.clim(0,1)
    plt.subplot(342)
    plt.imshow(im_stack[1,:,:])
    plt.clim(0,1)
    plt.subplot(343)
    plt.imshow(im_stack[2,:,:])
    plt.clim(0,1)
    plt.subplot(344)
    plt.imshow(im_stack[3,:,:])
    plt.clim(0,1)
    plt.subplot(345)
    plt.imshow(im_stack[4,:,:])
    plt.clim(0,1)
    plt.subplot(346)
    plt.imshow(im_stack[5,:,:])
    plt.clim(0,1)
    plt.subplot(347)
    plt.imshow(im_stack[6,:,:])
    plt.clim(0,1)
    plt.subplot(348)
    plt.imshow(im_stack[7,:,:])
    plt.clim(0,1)
    plt.subplot(349)
    plt.imshow(im_stack[8,:,:])
    plt.clim(0,1)
    plt.subplot(3,4,10)
    plt.imshow(im_stack[9,:,:])
    plt.clim(0,1)

plot_all_stacks(phantom_signal_im)
plot_all_stacks(vivo_signal)

plt.errorbar(np.abs(xdata),mean_Pgrey,std_Pgrey,ls = '',marker='x', capsize=3)
plt.errorbar(np.abs(xdata),mean_grey,std_grey,ls = '',marker='x', capsize=3)
plt.show()
plt.errorbar(np.abs(xdata),mean_Pwhite,std_Pwhite,ls = '',marker='x', capsize=3)
plt.errorbar(np.abs(xdata),mean_white,std_white,ls = '',marker='x', capsize=3)
plt.show()
plt.errorbar(np.abs(xdata),mean_Pliquor,std_Pliquor,ls = '',marker='x', capsize=3)
plt.errorbar(np.abs(xdata),mean_liquor,std_liquor,ls = '',marker='x', capsize=3)
plt.show()