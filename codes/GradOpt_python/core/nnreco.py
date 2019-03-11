import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fnn


# HOW we reconstruct
class RecoConvNet_basic(nn.Module):
    def __init__(self,sz,nmb_conv_neurons_list):
        super(RecoConvNet_basic, self).__init__()
        
        self.sz = sz
        self.lspec = nmb_conv_neurons_list
        
        self.conv_layers = []
        self.bn_layers = []
        
        paramlist = []
        
        ksz = 3
        
        for l_idx in range(len(self.lspec)-1):
            conv_layer = nn.Conv2d(self.lspec[l_idx], self.lspec[l_idx+1], ksz, padding=1)
            self.conv_layers.append(conv_layer)
            
            conv_layer = conv_layer.cuda()
            
            bnlayer = nn.BatchNorm2d(self.lspec[l_idx+1])
            self.bn_layers.append(bnlayer)
            
            bnlayer = bnlayer.cuda()
            
            paramlist.append(conv_layer.weight)
            paramlist.append(bnlayer.weight)

            
        self.paramlist = nn.ParameterList(paramlist)
            

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.permute([0,2,1])
        x = x.view([batch_size,self.lspec[0],self.sz[0],self.sz[1]])
        
        for l_idx in range(len(self.lspec)-1):
            x = self.conv_layers[l_idx](x)
            
            if l_idx < len(self.lspec) - 2:
                x = self.bn_layers[l_idx](x)
                x = torch.relu(x)
        
        x = x.view([batch_size,self.lspec[-1],self.sz[0]*self.sz[1]])
        x = x.permute([0,2,1])
        
        return x
        
