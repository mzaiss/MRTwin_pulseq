import numpy as np
import torch
from termcolor import colored
import matplotlib.pyplot as plt
import os
import scipy

from sys import platform
import time
from shutil import copyfile

from core.pulseq_exporter import pulseq_write_GRE
from core.pulseq_exporter import pulseq_write_RARE
from core.pulseq_exporter import pulseq_write_BSSFP
from core.pulseq_exporter import pulseq_write_EPI
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
              
            
#            ax1=plt.subplot(2, 5, 5)
#            ax=plt.imshow(tonumpy(self.grad_moms[:,:,0].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
#            plt.ion()
#            plt.title('gradx')
#            fig = plt.gcf()
#            fig.set_size_inches(18, 3)
#            fig.colorbar(ax)
                        
#            ax1=plt.subplot(2, 5, 10)
#            ax=plt.imshow(tonumpy(self.grad_moms[:,:,1].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
#            plt.ion()
#            fig = plt.gcf()
#            fig.set_size_inches(18, 3)
#            fig.colorbar(ax)
#            
#            plt.show()
#            plt.pause(0.02)
            
    # k-space plot             
            ax1=plt.subplot(155)
           
            kx= tonumpy(self.scanner.kspace_loc[:,:,0])
            ky= tonumpy(self.scanner.kspace_loc[:,:,1])
            for i in range(kx.shape[1]):
                plt.plot(kx[:,i],ky[:,i])
                
            fig.set_size_inches(18, 3)            
            
            plt.show()
            plt.pause(0.02)
            
    # save current optimized parameter state to matlab array
    def export_to_matlab(self, experiment_id):
        basepath = self.get_base_path(experiment_id)
        
        scanner_dict = dict()
        scanner_dict['adc_mask'] = tonumpy(self.scanner.adc_mask)
        scanner_dict['B1'] = tonumpy(self.scanner.B1)
        scanner_dict['flips'] = tonumpy(self.flips)
        scanner_dict['event_times'] = np.abs(tonumpy(self.event_time))
        scanner_dict['grad_moms'] = tonumpy(self.grad_moms)
        scanner_dict['reco'] = tonumpy(self.target_image).reshape([self.scanner.sz[0],self.scanner.sz[1],2])
        scanner_dict['ROI'] = tonumpy(self.scanner.ROI_signal)
        scanner_dict['sz'] = self.scanner.sz
        scanner_dict['signal'] = tonumpy(self.scanner.signal)
        
        fn_target_array = "scanner_dict_tgt.mat"
        
        try:
            os.makedirs(basepath)
            os.makedirs(os.path.join(basepath,"data"))
        except:
            pass
        scipy.io.savemat(os.path.join(basepath,fn_target_array), scanner_dict)

        
    def get_base_path(self, experiment_id):
        if platform == 'linux':
            basepath = '/media/upload3t/CEST_seq/pulseq_zero/sequences'
        else:
            basepath = 'K:\CEST_seq\pulseq_zero\sequences'

        today_datestr = time.strftime('%y%m%d')
        basepath = os.path.join(basepath, "seq" + today_datestr)
        basepath = os.path.join(basepath, experiment_id)

        return basepath

    def export_to_pulseq(self, experiment_id, sequence_class):
        basepath = self.get_base_path(experiment_id)
        
        fn_target_array = "target_arr.npy"
        fn_pulseq = "target.seq"
        
        # overwrite protection (gets trigger if pulseq file already exists)
#        today_datetimestr = time.strftime("%y%m%d%H%M%S")
#        if os.path.isfile(os.path.join(basepath, fn_pulseq)):
#            try:
#                copyfile(os.path.join(basepath, fn_pulseq), os.path.join(basepath, fn_pulseq + ".bak." + today_datetimestr))    
#                copyfile(os.path.join(basepath, fn_target_array), os.path.join(basepath, fn_target_array + ".bak." + today_datetimestr))    
#            except:
#                pass
        
        flips_numpy = tonumpy(self.flips)
        event_time_numpy = np.abs(tonumpy(self.event_time))
        grad_moms_numpy = tonumpy(self.grad_moms)
        
        # save target seq param array
        target_array = dict()
        target_array['adc_mask'] = tonumpy(self.scanner.adc_mask)
        target_array['B1'] = tonumpy(self.scanner.B1)
        target_array['flips'] = flips_numpy
        target_array['event_times'] = event_time_numpy
        target_array['grad_moms'] = grad_moms_numpy
        target_array['kloc'] = tonumpy(self.scanner.kspace_loc)
        target_array['reco'] = tonumpy(self.target_image).reshape([self.scanner.sz[0],self.scanner.sz[1],2])
        target_array['ROI'] = tonumpy(self.scanner.ROI_signal)
        target_array['sz'] = self.scanner.sz
        target_array['signal'] = tonumpy(self.scanner.signal)
        target_array['sequence_class'] = sequence_class
        
        try:
            os.makedirs(basepath)
            os.makedirs(os.path.join(basepath,"data"))
        except:
            pass
        np.save(os.path.join(os.path.join(basepath, fn_target_array)), target_array)
        
        # save sequence
        seq_params = flips_numpy, event_time_numpy, grad_moms_numpy
        
        if sequence_class.lower() == "gre":
            pulseq_write_GRE(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=True)
        elif sequence_class.lower() == "rare":
            pulseq_write_RARE(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=True)
        elif sequence_class.lower() == "bssfp":
            pulseq_write_BSSFP(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=True)
        elif sequence_class.lower() == "epi":
            pulseq_write_EPI(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=True)
        
        
        
        
        