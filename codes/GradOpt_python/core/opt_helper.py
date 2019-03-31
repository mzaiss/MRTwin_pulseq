import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fnn
from termcolor import colored
import matplotlib.pyplot as plt
from torch import optim
import os, sys
import scipy
import math
from torch.optim.optimizer import Optimizer

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import pickle

# torch to numpy
def tonumpy(x):
    return x.detach().cpu().numpy()
    
def get_cuda_mem_GB():
    return torch.cuda.get_device_properties(0).total_memory / 1024.0**3

def magimg(x):
    return np.sqrt(np.sum(np.abs(x)**2,2))

def phaseimg(x):
    return np.angle(1j*x[:,:,1]+x[:,:,0])

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
        
        self.last_reco = None
        self.last_error = None        
        
        self.param_reco_history = []
        
        self.target_seq_holder = None
        
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
        
    # evaluate loss and partial derivatives over parameters
    def weak_closure(self):
        self.optimizer.zero_grad()
        loss,last_reco,last_error = self.phi_FRP_model(self.scanner_opt_params, self.aux_params)
        self.last_reco = last_reco
        self.last_error = last_error
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
                optimizable_params.append({'params':self.scanner_opt_params[self.opt_param_idx[i]], 'lr': self.custom_learning_rate[self.opt_param_idx[i]]} )
            
        # optimize only sequence parameters
        if self.opti_mode == 'seq':
            if self.optimzer_type == 'Adam':
                self.optimizer = Bdam(optimizable_params, lr=self.learning_rate, weight_decay=WEIGHT_DECAY)
            else:
                self.optimizer = optim.LBFGS(optimizable_params, lr=self.learning_rate)
                
        # optimize only NN reconstruction module parameters
        elif self.opti_mode == 'nn':
            if self.optimzer_type == 'Adam':
                self.optimizer = Bdam(list(self.NN.parameters()), lr=self.learning_rate, weight_decay=WEIGHT_DECAY)
            else:
                self.optimizer = optim.LBFGS(list(self.NN.parameters()), lr=self.learning_rate)
            
        # optimize both sequence and NN reconstruction module parameters
        elif self.opti_mode == 'seqnn':
            optimizable_params.append({'params':self.NN.parameters(), 'lr': self.learning_rate} )
            
            if self.optimzer_type == 'Adam':
                self.optimizer = Bdam(optimizable_params, lr=self.learning_rate, weight_decay=WEIGHT_DECAY)
            else:
                self.optimizer = optim.LBFGS(optimizable_params, lr=self.learning_rate)
            
        
    # main training function
    def train_model(self, training_iter = 100, show_par=False, do_vis_image=False, save_intermediary_results=False):

        self.aux_params = [self.use_periodic_grad_moms_cap, self.opti_mode]
        self.init_optimizer()
        
        # continue optimization if optimizer state is saved
        if self.best_optimizer_state is not None:
            checkpoint = torch.load("results/optimizer_state.tmp")
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            print('Loading saved optimizer state....')
            
        # main optimization loop
        for inner_iter in range(training_iter):
            
            # evaluate initial image state before doing optimizer step
            if inner_iter == 0:
                _,self.last_reco,self.last_error = self.phi_FRP_model(self.scanner_opt_params, self.aux_params)
            print(colored("\033[93m iter %d, recon error = %f \033[0m" % (inner_iter,self.last_error), 'green'))
            
            
            # save entire history of optimized params/reco images
            if save_intermediary_results:
                    
                saved_state = dict()
                if 0 in self.opt_param_idx:
                    saved_state['adc_mask'] = tonumpy(self.scanner_opt_params[0])
                else:
                    saved_state['adc_mask'] = tonumpy(self.target_seq_holder.adc_mask)
                    
                if 1 in self.opt_param_idx:
                    saved_state['flips_angles'] = tonumpy(self.scanner_opt_params[1])
                else:
                    saved_state['flips_angles'] = tonumpy(self.target_seq_holder.flips_angles)
                    
                if 2 in self.opt_param_idx:
                    saved_state['event_times'] = tonumpy(self.scanner_opt_params[2])
                else:
                    saved_state['event_times'] = tonumpy(self.target_seq_holder.event_time)
                    
                if 3 in self.opt_param_idx:
                    saved_state['grad_moms'] = tonumpy(self.scanner_opt_params[3])
                else:
                    saved_state['grad_moms'] = tonumpy(self.target_seq_holder.grad_moms)
                    

                legs=['x','y','z']
                for i in range(3):
                    M_roi = tonumpy(self.scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(self.scanner.T+1)*self.scanner.NRep])
                    saved_state['ROI_def %d, %s'  % (self.scanner.ROI_def,legs[i])]  = M_roi

                saved_state['reco_image'] = tonumpy(self.last_reco)
                saved_state['error'] = self.last_error
                
                self.param_reco_history.append(saved_state)

            self.print_status(do_vis_image,self.last_reco)

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
                    torch.save(state, "results/optimizer_state.tmp")
                    self.best_optimizer_state = 'saved'
                    
                    best_vars = []
                    for par in self.scanner_opt_params:
                        best_vars.append(par.detach().clone())
                    

                    self.print_status(do_vis_image,reco)
                        
        for pidx in range(len(self.scanner_opt_params)):
            self.scanner_opt_params[pidx] = best_vars[pidx]
            self.scanner_opt_params[pidx].requires_grad = True        # needed?
                
            
    def set_target(self,target):
        self.target = target
            
    def print_status(self, do_vis_image=False, reco=None):
        if do_vis_image:
            sz=self.spins.sz
            recoimg = tonumpy(reco).reshape([sz[0],sz[1],2])

            # clear previous figure stack            
            plt.clf()            
            
            ax1=plt.subplot(251)
            ax=plt.imshow(magimg(self.target), interpolation='none')
            plt.clim(np.min(np.abs(self.target)),np.max(np.abs(self.target)))
            #plt.clim(0,1)
            fig = plt.gcf()
            fig.colorbar(ax)
            plt.title('target')
            plt.ion()
            
            ax1=plt.subplot(256)
            ax=plt.imshow(tonumpy(self.spins.PD0_mask)*phaseimg(self.target), interpolation='none')
            plt.clim(-np.pi,np.pi) 
            #plt.clim(0,1)
            fig = plt.gcf()
            fig.colorbar(ax)
            plt.title('target phase')
            plt.ion()
            
            plt.subplot(252, sharex=ax1, sharey=ax1)
            ax=plt.imshow(magimg(recoimg), interpolation='none')
            plt.clim(np.min(np.abs(self.target)),np.max(np.abs(self.target)))
            fig = plt.gcf()
            fig.colorbar(ax)
            plt.title('reco')
            plt.ion()
            
            plt.subplot(257, sharex=ax1, sharey=ax1)
            ax=plt.imshow(tonumpy(self.spins.PD0_mask)*phaseimg(recoimg), interpolation='none')
            plt.clim(-np.pi,np.pi) 
            fig = plt.gcf()
            fig.colorbar(ax)
            plt.title('reco phase')
            plt.ion()
               
            
            if self.scanner_opt_params[1].dim() == 3:
                FA=self.scanner_opt_params[1][:,:,0]
                phi=self.scanner_opt_params[1][:,:,1]
            else:
                FA=self.scanner_opt_params[1]
                phi=self.scanner_opt_params[1][:,:,1]
            plt.subplot(253)
            ax=plt.imshow(tonumpy(FA.permute([1,0]))*180/np.pi,cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('FA [\N{DEGREE SIGN}]')
            plt.clim(-180,270)
            fig = plt.gcf()
            fig.colorbar(ax)
            fig.set_size_inches(18, 3)
            
            plt.subplot(258)
            ax=plt.imshow(tonumpy(phi.permute([1,0]))*180/np.pi,cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('phase [\N{DEGREE SIGN}]')
            plt.clim(-180,270)
            fig = plt.gcf()
            fig.colorbar(ax)
            fig.set_size_inches(18, 3)
            
            
            plt.subplot(154)
            ax=plt.imshow(tonumpy(torch.abs(self.scanner_opt_params[2]).permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('TR [s]')
            fig = plt.gcf()
            fig.set_size_inches(18, 3)
            fig.colorbar(ax)
              
            
            ax1=plt.subplot(2, 5, 5)
            ax=plt.imshow(tonumpy(self.scanner_opt_params[3][:,:,0].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('gradx')
            fig = plt.gcf()
            fig.set_size_inches(18, 3)
            fig.colorbar(ax)
               
            
            ax1=plt.subplot(2, 5, 10)
            ax=plt.imshow(tonumpy(self.scanner_opt_params[3][:,:,1].permute([1,0])),cmap=plt.get_cmap('nipy_spectral'))
            plt.ion()
            plt.title('grady')
            fig = plt.gcf()
            fig.set_size_inches(18, 3)
            fig.colorbar(ax)
            
            plt.show()
            plt.pause(0.02)
            
            
            legs=['x','y','z']
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.plot(tonumpy(self.scanner.ROI_signal[:,:,1+i]).transpose([1,0]).reshape([(self.scanner.T+1)*self.scanner.NRep]) )
                if (i==0) and (self.target_seq_holder is not None):
                    plt.plot(tonumpy(self.target_seq_holder.ROI_signal[:,:,1]).transpose([1,0]).reshape([(self.scanner.T+1)*self.scanner.NRep]) ) 
                if (i==2):
                    plt.plot(tonumpy(self.scanner.ROI_signal[:,:,4]).transpose([1,0]).reshape([(self.scanner.T+1)*self.scanner.NRep]),'--') 
                plt.title("ROI_def %d, %s" % (self.scanner.ROI_def,legs[i]))
                fig = plt.gcf()
                fig.set_size_inches(16, 3)
            plt.show()
            plt.pause(0.02)
            
    # save current optimized parameter state to matlab array
    def export_to_matlab(self, experiment_id):
        _,reco,error = self.phi_FRP_model(self.scanner_opt_params, self.aux_params)        
        
        scanner_dict = dict()
        scanner_dict['adc_mask'] = tonumpy(self.scanner.adc_mask)
        scanner_dict['B1'] = tonumpy(self.scanner.B1)
        scanner_dict['flips'] = tonumpy(self.scanner_opt_params[1])
        scanner_dict['event_times'] = np.abs(tonumpy(self.scanner_opt_params[2]))
        scanner_dict['grad_moms'] = tonumpy(self.scanner_opt_params[3])
        scanner_dict['reco'] = tonumpy(reco).reshape([self.scanner.sz[0],self.scanner.sz[1],2])
        scanner_dict['ROI'] = tonumpy(self.scanner.ROI_signal)
        scanner_dict['sz'] = self.scanner.sz
        scanner_dict['adjoint_mtx'] = tonumpy(self.scanner.G_adj.permute([2,3,0,1,4]))
        scanner_dict['signal'] = tonumpy(self.scanner.signal)

        path=os.path.join('./out/',experiment_id)
        try:
            os.mkdir(path)
        except:
            print('export_to_matlab: directory already exists')
        scipy.io.savemat(os.path.join(path,"scanner_dict.mat"), scanner_dict)
        
    # save entire history of the optimized parameters
    def save_param_reco_history(self, experiment_id):
        path=os.path.join('./out/',experiment_id)
        try:
            os.mkdir(path)
        except:
            print('save_param_reco_history: directory already exists')
            
        param_reco_history = self.param_reco_history
        
        aux_info = dict()
        aux_info['sz'] = self.scanner.sz
        aux_info['NRep'] = self.scanner.NRep
        aux_info['T'] = self.scanner.T
        aux_info['target'] = self.target
        aux_info['ROI_def'] = self.scanner.ROI_def
        
        f = open(os.path.join(path,"param_reco_history.pdb"), "wb")
        pickle.dump((param_reco_history, aux_info), f)
        f.close()
                                   

# Adam variation to allow for blocking gradient steps on individual entries of parameter vector
class Bdam(Optimizer):
    r"""Implements Adam algorithm (with tricks).

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Bdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Bdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # set to zero masked-out entries
                if hasattr(p,'zero_grad_mask'):
                    exp_avg *= p.zero_grad_mask
                
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
    
    