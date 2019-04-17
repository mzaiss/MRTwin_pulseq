import numpy as np
import torch
from termcolor import colored
import matplotlib.pyplot as plt
import os
import scipy
# target images / sequence parameters holder
# torch to numpy
def tonumpy(x):
    return x.detach().cpu().numpy()

def magimg(x):
    return np.sqrt(np.sum(np.abs(x)**2,2))

def phaseimg(x):
    return np.angle(1j*x[:,:,1]+x[:,:,0])

class TargetSequenceHolder():
    def __init__(self,flips,event_time,grad_moms,scanner,spins,target):
        
        self.scanner = scanner
        self.target_image = target
        self.sz = scanner.sz
        self.flips = flips.clone()
        self.grad_moms = grad_moms.clone()
        self.event_time = event_time.clone()
        self.adc_mask = scanner.adc_mask.clone()
        self.ROI_signal=scanner.ROI_signal.clone()        
        
        self.ROI_def = 1
        self.PD0_mask = spins.PD0_mask
        
        self.batch_size = 1
        
        # batched mode
        if self.target_image.dim() == 3:
            self.batch_size = self.target_image.shape[0]
            self.target_image = self.target_image[0,:,:]
            self.PD0_mask = self.PD0_mask[0,:,:]
        
    def print_status(self, do_vis_image=False, reco=None):
        if do_vis_image:
            
            recoimg= (tonumpy(self.target_image).reshape([self.sz[0],self.sz[1],2]))
            recoimg_phase = tonumpy(self.PD0_mask)*phaseimg(recoimg)
    
            # clear previous figure stack            
            plt.clf()            
            
            ax1=plt.subplot(151)
            ax=plt.imshow(magimg(recoimg), interpolation='none')
            #plt.clim(0,1)
            fig = plt.gcf()
            fig.colorbar(ax)
            plt.title('target reco')
            plt.ion()
            
            plt.subplot(152, sharex=ax1, sharey=ax1)
            ax=plt.imshow(recoimg_phase, interpolation='none')
            plt.clim(-np.pi,np.pi)
            fig = plt.gcf()
            fig.colorbar(ax)
            plt.title('target reco phase')
            plt.ion()
               
            plt.subplot(153)
            if self.flips.dim() == 3:
                FA=self.flips[:,:,0]
            else:
                FA=self.flips
                
            ax=plt.imshow(np.transpose(tonumpy(FA*180/np.pi),[1,0]),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('FA [\N{DEGREE SIGN}]')
            plt.clim(-90,270)
            fig = plt.gcf()
            fig.colorbar(ax)
            fig.set_size_inches(18, 3)
            
            
            plt.subplot(154)
            ax=plt.imshow(tonumpy(torch.abs(self.event_time).permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('TR [s]')
            fig = plt.gcf()
            fig.set_size_inches(18, 3)
            fig.colorbar(ax)
              
            
            ax1=plt.subplot(2, 5, 5)
            ax=plt.imshow(tonumpy(self.grad_moms[:,:,0].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('gradx')
            fig = plt.gcf()
            fig.set_size_inches(18, 3)
            fig.colorbar(ax)
               
            
            ax1=plt.subplot(2, 5, 10)
            ax=plt.imshow(tonumpy(self.grad_moms[:,:,1].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            fig = plt.gcf()
            fig.set_size_inches(18, 3)
            fig.colorbar(ax)
            
            plt.show()
            plt.pause(0.02)
            
    # save current optimized parameter state to matlab array
    def export_to_matlab(self, experiment_id):
        scanner_dict = dict()
        scanner_dict['adc_mask'] = tonumpy(self.scanner.adc_mask)
        scanner_dict['B1'] = tonumpy(self.scanner.B1)
        scanner_dict['flips'] = tonumpy(self.flips)
        scanner_dict['event_times'] = np.abs(tonumpy(self.event_time))
        scanner_dict['grad_moms'] = tonumpy(self.grad_moms)
        scanner_dict['reco'] = tonumpy(self.target_image).reshape([self.scanner.sz[0],self.scanner.sz[1],2])
        scanner_dict['ROI'] = tonumpy(self.scanner.ROI_signal)
        scanner_dict['sz'] = self.scanner.sz
        #scanner_dict['adjoint_mtx'] = tonumpy(self.scanner.G_adj.permute([2,3,0,1,4]))
        scanner_dict['signal'] = tonumpy(self.scanner.signal)

        path=os.path.join('./out/',experiment_id)
        try:
            os.mkdir(path)
        except:
            print('export_to_matlab: directory already exists')
        scipy.io.savemat(os.path.join(path,"scanner_dict_tgt.mat"), scanner_dict)
