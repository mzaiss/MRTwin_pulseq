import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fnn
from termcolor import colored
import matplotlib.pyplot as plt
from torch import optim

# torch to numpy
def tonumpy(x):
    return x.detach().cpu().numpy()

# optimization helper class
class OPT_helper():
    def __init__(self,scanner,spins,NN,nmb_total_samples_dataset):
        self.globepoch = 0
        self.subjidx = 0
        self.nmb_total_samples_dataset = nmb_total_samples_dataset
        
        self.use_periodic_grad_moms_cap = 1
        self.learning_rate = 0.02
        self.custom_learning_rate = None   # allow for differerent learning rates in different parameter groups
        
        self.opti_mode = 'seq'
        self.optimzer_type = 'Adam'
        
        self.scanner = scanner
        self.spins = spins
        self.NN = NN
        
        self.optimizer = None
        self.best_optimizer_state = None
        self.init_variables = None
        self.phi_FRP_model = None
        
        self.scanner_opt_params = None
        self.aux_params = None
        self.opt_param_idx = []
        
    def set_target(self,target):
        self.target = target
        
    def set_handles(self,init_variables,phi_FRP_model):
        self.init_variables = init_variables
        self.phi_FRP_model = phi_FRP_model
        
    def set_opt_param_idx(self,opt_param_idx):
        self.opt_param_idx  = opt_param_idx
        
    def new_batch(self):
        self.globepoch += 1
        
        if hasattr(self.spins, 'batch_size'):
            batch_size = self.spins.batch_size
        else:
            batch_size = 1
        
        self.subjidx = np.random.choice(self.nmb_total_samples_dataset, batch_size, replace=False)
        
    def weak_closure(self):
        self.optimizer.zero_grad()
        loss,_,_ = self.phi_FRP_model(self.scanner_opt_params, self.aux_params)
        loss.backward()
        
        return loss 
            
    def init_optimizer(self):
        WEIGHT_DECAY = 1e-8
        
        # only optimize a subset of params
        optimizable_params = []
        for i in range(len(self.opt_param_idx)):
            if self.custom_learning_rate == None:
                optimizable_params.append(self.scanner_opt_params[self.opt_param_idx[i]] )
            else:
                optimizable_params.append({'params':self.scanner_opt_params[self.opt_param_idx[i]], 'lr': self.custom_learning_rate[i]} )
            
        if self.opti_mode == 'seq':
            if self.optimzer_type == 'Adam':
                self.optimizer = optim.Adam(optimizable_params, lr=self.learning_rate, weight_decay=WEIGHT_DECAY)
            else:
                self.optimizer = optim.LBFGS(optimizable_params, lr=self.learning_rate)
        elif self.opti_mode == 'nn':
            if self.optimzer_type == 'Adam':
                self.optimizer = optim.Adam(list(self.NN.parameters()), lr=self.learning_rate, weight_decay=WEIGHT_DECAY)
            else:
                self.optimizer = optim.LBFGS(list(self.NN.parameters()), lr=self.learning_rate)
            
        elif self.opti_mode == 'seqnn':
            optimizable_params.append({'params':self.NN.parameters(), 'lr': self.learning_rate} )
            
            if self.optimzer_type == 'Adam':
                self.optimizer = optim.Adam(optimizable_params, lr=self.learning_rate, weight_decay=WEIGHT_DECAY)
            else:
                self.optimizer = optim.LBFGS(optimizable_params, lr=self.learning_rate)
            
            
            #self.optimizer = optim.Adam(list(self.NN.parameters())+optimizable_params, lr=self.learning_rate, weight_decay=WEIGHT_DECAY)            
        
        
        
    def train_model(self, training_iter = 100, show_par=False, do_vis_image=False):

        self.aux_params = [self.use_periodic_grad_moms_cap, self.opti_mode]
        self.init_optimizer()
        
        # continue optimization if state is saved
        if self.best_optimizer_state is not None:
            checkpoint = torch.load("../../results/optimizer_state.tmp")
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            print('Loading saved optimizer state....')
            

            
        for inner_iter in range(training_iter):
            
            _,reco,error = self.phi_FRP_model(self.scanner_opt_params, self.aux_params)
            print(colored("\033[93m iter %d, recon error = %f \033[0m" % (inner_iter,error), 'green'))
            
            
            
            if show_par:
                par_group = tonumpy(self.scanner_opt_params[self.opt_param_idx[0]])*180/np.pi
                par_group = np.round(100*par_group[0,:])/100
                print(par_group)
                
            
            self.print_status(do_vis_image,reco)
            

            self.new_batch()
            self.optimizer.step(self.weak_closure)

                
        
    
    def train_model_with_restarts(self, nmb_rnd_restart=15, training_iter=10, do_vis_image=False):
        
        # init gradients and flip events
        nmb_outer_iter = nmb_rnd_restart
        nmb_inner_iter = training_iter
        
        self.aux_params = [self.use_periodic_grad_moms_cap, self.opti_mode]
        
        best_error = 1000
        
        for outer_iter in range(nmb_outer_iter):
            #print('restarting... %i%% ready' %(100*outer_iter/float(nmb_outer_iter)))
            print('restarting the model training... ')

            self.scanner_opt_params = self.init_variables()
            self.init_optimizer()
           
            for inner_iter in range(nmb_inner_iter):
                self.new_batch()
                self.optimizer.step(self.weak_closure)
                
                _,reco,error = self.phi_FRP_model(self.scanner_opt_params, self.aux_params)
                
                if error < best_error:
                    print("recon error = %f" %error)
                    best_error = error
                    
                    state = {'optimizer': self.optimizer.state_dict()}
                    torch.save(state, "../../results/optimizer_state.tmp")
                    self.best_optimizer_state = 'saved'
                    
                    best_vars = []
                    for par in self.scanner_opt_params:
                        best_vars.append(par.detach().clone())
                    

                    self.print_status(do_vis_image,reco)
                        
        for pidx in range(len(self.scanner_opt_params)):
            self.scanner_opt_params[pidx] = best_vars[pidx]
            self.scanner_opt_params[pidx].requires_grad = True        # needed?
                
            
            
            
    def print_status(self, do_vis_image=False, reco=None):
        if do_vis_image:
            sz=self.spins.sz
            recoimg = tonumpy(reco).reshape([sz[0],sz[1],2])
                       
            ax1=plt.subplot(151)
            ax=plt.imshow(magimg(self.target), interpolation='none')
            #plt.clim(0,1)
            fig = plt.gcf()
            fig.colorbar(ax)
            plt.title('target')
            plt.ion()
            
            plt.subplot(152, sharex=ax1, sharey=ax1)
            ax=plt.imshow(magimg(recoimg), interpolation='none')
            plt.clim(np.min(self.target),np.max(self.target))
            fig = plt.gcf()
            fig.colorbar(ax)
            plt.title('reco')
            plt.ion()
               
            plt.subplot(153)
            ax=plt.imshow(tonumpy(self.scanner_opt_params[0].permute([1,0]))*180/np.pi,cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('FA [\N{DEGREE SIGN}]')
            plt.clim(-90,270)
            fig = plt.gcf()
            fig.colorbar(ax)
            fig.set_size_inches(18, 3)
            
            
            plt.subplot(154)
            ax=plt.imshow(tonumpy(torch.abs(self.scanner_opt_params[2])[:,:,0].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('TR [s]')
            fig = plt.gcf()
            fig.set_size_inches(18, 3)
            fig.colorbar(ax)
              
            
            ax1=plt.subplot(2, 5, 5)
            ax=plt.imshow(tonumpy(torch.abs(self.scanner_opt_params[1])[:,:,0].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('gradx')
            fig = plt.gcf()
            fig.set_size_inches(18, 3)
            fig.colorbar(ax)
               
            
            ax1=plt.subplot(2, 5, 10)
            ax=plt.imshow(tonumpy(torch.abs(self.scanner_opt_params[1])[:,:,1].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('grady')
            fig = plt.gcf()
            fig.set_size_inches(18, 3)
            fig.colorbar(ax)
            
        plt.show()                   
                                   
def magimg(x):


