import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fnn

# basic mumltilayer CNN
class RecoConvNet_basic(nn.Module):   
    # nmb_conv_neurons_list (list) - number of elements in the list - number of layers, each element of the list - number of conv neurons
    # kernel_size -- convolution kernel size
    
    def __init__(self,sz,nmb_conv_neurons_list,kernel_size=3,use_gpu=True,gpu_device=0):
        super(RecoConvNet_basic, self).__init__()
        
        self.sz = sz
        self.layer_spec = nmb_conv_neurons_list
        
        self.conv_layers = []                              # convolution layers
        self.bn_layers = []                        # batch normalization layers
        
        paramlist = []                                # paramlist for optimizer
        
        padding = (kernel_size - 1)//2
        
        for l_idx in range(len(self.layer_spec)-1):
            
            # set convolution layers
            conv_layer = nn.Conv2d(self.layer_spec[l_idx], self.layer_spec[l_idx+1], kernel_size, padding=padding)
            self.conv_layers.append(conv_layer)
            
            # set batch normalization layers
            bnlayer = nn.BatchNorm2d(self.layer_spec[l_idx+1])
            self.bn_layers.append(bnlayer)
            
            if use_gpu:
                conv_layer = conv_layer.cuda(gpu_device)
                bnlayer = bnlayer.cuda(gpu_device)
            
            for par in conv_layer.parameters():
                paramlist.append(par)            
            for par in bnlayer.parameters():
                paramlist.append(par)   

        self.paramlist = nn.ParameterList(paramlist)
            
    # define forward pass graph
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.permute([0,2,1])
        x = x.view([batch_size,self.layer_spec[0],self.sz[0],self.sz[1]])
        
        for l_idx in range(len(self.layer_spec)-1):
            x = self.conv_layers[l_idx](x)
            
            if l_idx < len(self.layer_spec) - 2:
                x = self.bn_layers[l_idx](x)
                x = torch.relu(x)
                
        x = x.view([batch_size,self.layer_spec[-1],self.sz[0]*self.sz[1]])
        x = x.permute([0,2,1])
        
        return x

# basic mumltilayer CNN, with identity (RESNET-like) connection between input/output
class RecoConvNet_residual(nn.Module):   
    # nmb_conv_neurons_list (list) - number of elements in the list - number of layers, each element of the list - number of conv neurons
    # kernel_size -- convolution kernel size
    
    def __init__(self,sz,nmb_conv_neurons_list,kernel_size=3,use_gpu=True,gpu_device=0):
        super(RecoConvNet_residual, self).__init__()
        
        self.sz = sz
        self.layer_spec = nmb_conv_neurons_list
        
        self.conv_layers = []                              # convolution layers
        self.bn_layers = []                        # batch normalization layers
        
        paramlist = []                                # paramlist for optimizer
        
        padding = (kernel_size - 1)//2
        
        for l_idx in range(len(self.layer_spec)-1):
            
            # set convolution layers
            conv_layer = nn.Conv2d(self.layer_spec[l_idx], self.layer_spec[l_idx+1], kernel_size, padding=padding)
            self.conv_layers.append(conv_layer)
            
            # set batch normalization layers
            bnlayer = nn.BatchNorm2d(self.layer_spec[l_idx+1])
            self.bn_layers.append(bnlayer)
            
            # since we use resnet, initialize all conv kern weight to small value
            #conv_layer.weight.data *= 1e-3
            
            if use_gpu:
                conv_layer = conv_layer.cuda(gpu_device)
                bnlayer = bnlayer.cuda(gpu_device)
            
            for par in conv_layer.parameters():
                paramlist.append(par)            
            for par in bnlayer.parameters():
                paramlist.append(par)            

        self.paramlist = nn.ParameterList(paramlist)
            
    # define forward pass graph
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.permute([0,2,1])
        x = x.view([batch_size,self.layer_spec[0],self.sz[0],self.sz[1]])
        
        identity = x
        
        for l_idx in range(len(self.layer_spec)-1):
            x = self.conv_layers[l_idx](x)
            
            if l_idx < len(self.layer_spec) - 2:
                x = self.bn_layers[l_idx](x)
                x = torch.relu(x)
                
        x += identity
        
        x = x.view([batch_size,self.layer_spec[-1],self.sz[0]*self.sz[1]])
        x = x.permute([0,2,1])
        
        return x
        

class VoxelwiseNet(nn.Module):   
    # nmb_conv_neurons_list (list) - number of elements in the list - number of layers, each element of the list - number of conv neurons
    # kernel_size -- convolution kernel size
    
    def __init__(self,sz,nmb_hidden_neurons,use_gpu=True,gpu_device=0):
        super(VoxelwiseNet, self).__init__()
        
        self.sz = sz
        self.layer_spec = nmb_hidden_neurons
        self.dense_layers = []
        
        paramlist = []                                # paramlist for optimizer
        
        for l_idx in range(len(self.layer_spec)-1):
            # set convolution layers
            dense_layer = nn.Linear(self.layer_spec[l_idx], self.layer_spec[l_idx+1], bias=True)
            self.dense_layers.append(dense_layer)
            
            if use_gpu:
                dense_layer = dense_layer.cuda(gpu_device)
             
            for par in dense_layer.parameters():
                paramlist.append(par)

        self.paramlist = nn.ParameterList(paramlist)
            
    # define forward pass graph
    def forward(self, x):
        for l_idx in range(len(self.layer_spec)-1):
            x = self.dense_layers[l_idx](x)
            
            if l_idx < len(self.layer_spec) - 2:
				#x = torch.tanh(x)
                x = torch.relu(x)
                
        return x