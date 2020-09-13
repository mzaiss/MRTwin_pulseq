import numpy as np
import torch
from termcolor import colored
import matplotlib.pyplot as plt
import os
import scipy
import socket
from matplotlib.pyplot import cm



from sys import platform
import time
from shutil import copyfile

from core.pulseq_exporter import pulseq_write_GRE
from core.pulseq_exporter import pulseq_write_GRE_DREAM
from core.pulseq_exporter import pulseq_write_RARE
from core.pulseq_exporter import pulseq_write_BSSFP
from core.pulseq_exporter import pulseq_write_slBSSFP
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
    def __init__(self,rf_event,event_time,gradm_event,scanner,spins,target):
        
        self.scanner = scanner
        self.target_image = target
        self.sz = scanner.sz
        self.rf_event = rf_event.clone()
        self.gradm_event = gradm_event.clone()
        self.event_time = event_time.clone()
        self.adc_mask = scanner.adc_mask.clone()
        self.ROI_signal=scanner.ROI_signal.clone()
        
        self.sim_sig = self.scanner.signal.clone()
        
        self.meas_sig = None
        self.meas_reco = None
        
        self.ROI_def = 1
        self.PD0_mask = spins.PD0_mask
        
        self.batch_size = 1
        
        # batched mode
        if self.target_image.dim() == 3:
            self.batch_size = self.target_image.shape[0]
            self.target_image = self.target_image[0,:,:]
            self.PD0_mask = self.PD0_mask[0,:,:]
        
    def print_status(self, do_vis_image=False, kplot=False, reco=None, do_scanner_query=False):
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
               
            plt.subplot(253)
            if self.rf_event.dim() == 3:
                FA=self.rf_event[:,:,0]
            else:
                FA=self.rf_event
                
            ax=plt.imshow(np.transpose(tonumpy(FA*180/np.pi),[1,0]),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('FA [\N{DEGREE SIGN}]')
            plt.clim(-90,270)
            fig = plt.gcf()
            fig.colorbar(ax)
            fig.set_size_inches(18, 3)
            
            plt.subplot(258)
            if self.rf_event.dim() == 3:
                FA=self.rf_event[:,:,1]
            else:
                FA=self.rf_event
                
            ax=plt.imshow(np.transpose(tonumpy(FA*180/np.pi),[1,0]),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('phase [\N{DEGREE SIGN}]')
            plt.clim(0,360)
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
#            ax=plt.imshow(tonumpy(self.gradm_event[:,:,0].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
#            plt.ion()
#            plt.title('gradx')
#            fig = plt.gcf()
#            fig.set_size_inches(18, 3)
#            fig.colorbar(ax)
                        
#            ax1=plt.subplot(2, 5, 10)
#            ax=plt.imshow(tonumpy(self.gradm_event[:,:,1].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
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
            
            if do_scanner_query:
                plt.subplot(141)
                ax = plt.imshow(magimg(tonumpy(self.meas_reco.detach()).reshape([self.sz[0],self.sz[1],2])), interpolation='none')
                fig = plt.gcf()
                fig.colorbar(ax)
                plt.title("meas mag ADJ")
                
                plt.subplot(142)
                ax = plt.imshow(phaseimg(tonumpy(self.meas_reco.detach()).reshape([self.sz[0],self.sz[1],2])), interpolation='none')
                fig = plt.gcf()
                fig.colorbar(ax)
                plt.title("meas phase ADJ")
                
                NCol = self.scanner.NCol
                NRep = self.scanner.NRep
                
                coil_idx = 0
                adc_idx = np.where(self.scanner.adc_mask.cpu().numpy())[0]
                sim_kspace = self.sim_sig[coil_idx,adc_idx,:,:2,0]
                sim_kspace = magimg(tonumpy(sim_kspace.detach()).reshape([NCol,NRep,2]))
                
                plt.subplot(143)
                ax=plt.imshow(sim_kspace, interpolation='none')
                plt.title("sim kspace")   
                fig = plt.gcf()
                fig.colorbar(ax)
                
                meas_kspace = self.scanner.signal[coil_idx,adc_idx,:,:2,0]
                meas_kspace = magimg(tonumpy(meas_kspace.detach()).reshape([NCol,NRep,2]))     
                
                plt.subplot(144)
                ax=plt.imshow(meas_kspace, interpolation='none')
                plt.title("meas kspace")    
                fig = plt.gcf()
                fig.colorbar(ax)

                fig.set_size_inches(18, 3)
                
                plt.ion()
                plt.show()                
                plt.pause(0.02)      
                
    def print_seq_pic(self, kplot=False, plotsize=[20,2]):
            # clear previous figure stack            
            plt.clf()            
               
            plt.subplot(331); plt.title('event times [s]'); plt.ylabel('repetition'); plt.yticks(np.arange(0, self.scanner.NRep, 5))
#            ax=plt.pcolormesh(tonumpy(torch.abs(self.event_time).permute([1,0])), edgecolors='k', linewidth=1,cmap=plt.get_cmap('nipy_spectral'))
#            ax.set_aspect('equal')
            ax=plt.imshow(tonumpy(torch.abs(self.event_time).permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))        
            fig = plt.gcf();fig.colorbar(ax)
            
            plt.subplot(332); plt.title('ADC'); plt.yticks(np.arange(0, self.scanner.NRep, 5))
            ax=plt.imshow(np.tile(tonumpy(self.adc_mask),self.scanner.NRep).transpose(),cmap=plt.get_cmap('nipy_spectral'))        
            fig = plt.gcf();fig.colorbar(ax)
            
            
            plt.subplot(334); plt.title('rf flip [\N{DEGREE SIGN}]'); plt.ylabel('repetition'); plt.yticks(np.arange(0, self.scanner.NRep, 5))
            if self.rf_event.dim() == 3:
                flip_phase_event=self.rf_event
            else:
                flip_phase_event=self.rf_event
                
            ax=plt.imshow(np.transpose(tonumpy(flip_phase_event[:,:,0]*180/np.pi),[1,0]),cmap=plt.get_cmap('nipy_spectral'))
            plt.clim(-90,270)
            fig = plt.gcf(); fig.colorbar(ax)

            plt.subplot(335); plt.title('rf phase [\N{DEGREE SIGN}]'); plt.yticks(np.arange(0, self.scanner.NRep, 5))
            ax=plt.imshow(np.transpose(tonumpy(flip_phase_event[:,:,1]*180/np.pi),[1,0]),cmap=plt.get_cmap('nipy_spectral'))
            plt.clim(0,360)
            fig = plt.gcf();fig.colorbar(ax)

            plt.subplot(337); plt.title('grad_mom_x'); plt.xlabel('event index') ; plt.ylabel('repetition');plt.yticks(np.arange(0, self.scanner.NRep, 5))
            ax=plt.imshow(tonumpy(self.gradm_event[:,:,0].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            fig = plt.gcf();fig.colorbar(ax)
                        
            ax1=plt.subplot(338); plt.title('grad_mom_y'); plt.xlabel('event index');plt.yticks(np.arange(0, self.scanner.NRep, 5))
            ax=plt.imshow(tonumpy(self.gradm_event[:,:,1].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            fig = plt.gcf();fig.colorbar(ax)
            
            
            if kplot:  #k-space plot             
                plt.subplot(339) ; plt.title('k-space loc.')               
                kx= tonumpy(self.scanner.kspace_loc[:,:,0])
                ky= tonumpy(self.scanner.kspace_loc[:,:,1])
                color=cm.rainbow(np.linspace(0,1,kx.shape[1]))
                for i in range(kx.shape[1]):
                    plt.plot(kx[:,i],ky[:,i],c=color[i])
                plt.plot(kx[(tonumpy(self.adc_mask)).nonzero()[0],:],ky[(tonumpy(self.adc_mask)).nonzero()[0],:],'r.',markersize=0.75)
                plt.xlabel('k_x')
          
            fig = plt.gcf();  
            plt.ion()
            fig.set_size_inches(plotsize[0], plotsize[1])
            plt.show()
            plt.pause(0.02)
            
    def print_seq(self, plotsize=[20,2],time_axis=0):
        
        tfull=np.cumsum(tonumpy(self.event_time).transpose().ravel())
        tfull=np.insert(tfull, 0, 0)
        tfull=tfull[:-1]
        xlabel='time [s]'
        normg= 1/tonumpy(self.event_time).transpose().ravel() 
        normg[np.isnan(normg)] = 0
        normg[np.isinf(normg)] = 0
        
        if time_axis==0:
            tfull=np.arange(tfull.size)
            xlabel='event index'
            normg=1

        fig=plt.figure("""seq and image"""); fig.set_size_inches(plotsize); 
        plt.subplot(411); plt.ylabel('RF, time, ADC'); plt.title("Total acquisition time ={:.2} s".format(tonumpy(torch.sum(self.event_time))))
        plt.plot(tfull,np.tile(tonumpy(self.adc_mask),self.scanner.NRep).flatten('F'),'.',label='ADC')
        plt.plot(tfull,tonumpy(self.event_time).flatten('F'),'.',label='time')
#        plt.plot(tfull,tonumpy(self.rf_event[:,:,0]).flatten('F'),'.',label='RF')
        plt.stem(tfull, tonumpy(self.rf_event[:,:,0]).flatten('F'),'r',markerfmt ='ro',label='RF')
        major_ticks = np.arange(0, self.scanner.T*self.scanner.NRep, self.scanner.T) # this adds ticks at the correct position szread
        ax=plt.gca(); ax.set_xticks(tfull[major_ticks]); ax.grid()
        plt.legend()
        plt.subplot(412); plt.ylabel('gradients')
        #plt.plot(tfull,tonumpy(self.gradm_event[:,:,0]).flatten('F'),label='gx')
        #plt.plot(tfull,tonumpy(self.gradm_event[:,:,1]).flatten('F'),label='gy')
        
        plt.step(tfull, normg*tonumpy(self.gradm_event[:,:,0]).flatten('F'),label='gx', where='mid')
        plt.step(tfull,normg*tonumpy(self.gradm_event[:,:,1]).flatten('F'),label='gy', where='mid')
        ax=plt.gca(); ax.set_xticks(tfull[major_ticks]); ax.grid()
        plt.legend()
        plt.subplot(413); plt.ylabel('signal')
        plt.plot(tfull,tonumpy(self.scanner.signal[0,:,:,0,0]).flatten('F'),label='real')
        plt.plot(tfull,tonumpy(self.scanner.signal[0,:,:,1,0]).flatten('F'),label='imag')
        plt.xlabel(xlabel)
        ax=plt.gca(); ax.set_xticks(tfull[major_ticks]); ax.grid()
        plt.legend()
        plt.show()
        
                
                           
                
            
    # save current optimized parameter state to matlab array
    def export_to_matlab(self, experiment_id, today_datestr):
        basepath = self.get_base_path(experiment_id, today_datestr)
        
        scanner_dict = dict()
        scanner_dict['adc_mask'] = tonumpy(self.scanner.adc_mask)
        scanner_dict['B1'] = tonumpy(self.scanner.B1)
        scanner_dict['rf_event'] = tonumpy(self.rf_event)
        scanner_dict['event_times'] = np.abs(tonumpy(self.event_time))
        scanner_dict['gradm_event'] = tonumpy(self.gradm_event)
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

        
    def get_base_path(self, experiment_id, today_datestr):
        print(os.getcwd())
        if os.path.isfile(os.path.join('core','pathfile_local.txt')):
            pathfile ='pathfile_local.txt'
        else:
            pathfile ='pathfile.txt'
            print('You dont have a local pathfile in core/pathfile_local.txt, so we use standard file: pathfile.txt')
                
        with open(os.path.join('core',pathfile),"r") as f:
            path_from_file = f.readline()
        basepath = path_from_file
        basepath = os.path.join(basepath, 'sequences')
        basepath = os.path.join(basepath, "seq" + today_datestr)
        basepath = os.path.join(basepath, experiment_id)

        return basepath

    def export_to_pulseq(self, experiment_id, today_datestr, sequence_class, plot_seq=True,single_folder=False):
        basepath = self.get_base_path(experiment_id, today_datestr)
        
        if single_folder:
            basepath=os.path.dirname(basepath)
            fn_target_array = experiment_id+".npy"
            fn_pulseq = experiment_id+".seq"
        else:
            fn_target_array = "target_arr.npy"
            fn_pulseq = "target.seq"
        
        try:
            os.makedirs(basepath)
            os.makedirs(os.path.join(basepath,"data"))
        except:
            pass
        
        # overwrite protection (gets trigger if pulseq file already exists)
#        today_datetimestr = time.strftime("%y%m%d%H%M%S")
#        if os.path.isfile(os.path.join(basepath, fn_pulseq)):
#            try:
#                copyfile(os.path.join(basepath, fn_pulseq), os.path.join(basepath, fn_pulseq + ".bak." + today_datetimestr))    
#                copyfile(os.path.join(basepath, fn_target_array), os.path.join(basepath, fn_target_array + ".bak." + today_datetimestr))    
#            except:
#                pass
        
        rf_event_numpy = tonumpy(self.rf_event)
        event_time_numpy = np.abs(tonumpy(self.event_time))
        gradm_event_numpy = tonumpy(self.gradm_event)
        
        
        if not single_folder:
            # save target seq param array
            target_array = dict()
            target_array['adc_mask'] = tonumpy(self.scanner.adc_mask)
            target_array['B1'] = tonumpy(self.scanner.B1)
            target_array['rf_event'] = rf_event_numpy
            target_array['event_times'] = event_time_numpy
            target_array['gradm_event'] = gradm_event_numpy
            target_array['kloc'] = tonumpy(self.scanner.kspace_loc)
            try:
                target_array['reco'] = tonumpy(self.target_image).reshape([self.scanner.sz[0],self.scanner.sz[1],2])
            except:
                pass
                
            target_array['ROI'] = tonumpy(self.scanner.ROI_signal)
            target_array['sz'] = self.scanner.sz
            target_array['signal'] = tonumpy(self.scanner.signal)
            target_array['sequence_class'] = sequence_class
                   
            np.save(os.path.join(os.path.join(basepath, fn_target_array)), target_array)
        
        # save sequence
        seq_params = rf_event_numpy, event_time_numpy, gradm_event_numpy
        
        if sequence_class.lower() == "gre":
            pulseq_write_GRE(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=plot_seq)
        elif sequence_class.lower() == "gre_dream":
            pulseq_write_GRE_DREAM(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=plot_seq)
        elif sequence_class.lower() == "rare":
            pulseq_write_RARE(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=plot_seq)
        elif sequence_class.lower() == "se":
            pulseq_write_RARE(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=plot_seq)
        elif sequence_class.lower() == "bssfp":
            pulseq_write_BSSFP(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=plot_seq)
        elif sequence_class.lower() == "slbssfp":
            pulseq_write_slBSSFP(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=plot_seq)
        elif sequence_class.lower() == "epi":
            pulseq_write_EPI(seq_params, os.path.join(basepath, fn_pulseq), plot_seq=plot_seq)
        
        
        
        
        