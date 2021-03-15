#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:32:35 2020

@author: yan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self, encoder_arch, decoder_arch):
        super(Net, self).__init__()        
        self.encoder = self.Encoder(encoder_arch)
        self.decoder = self.Decoder(decoder_arch)
        
    def Encoder(self, encoder_arch):
        modules = []
        for i in range(len(encoder_arch)-1):
            modules.append(torch.nn.Linear(encoder_arch[i][0],encoder_arch[i][1]))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(encoder_arch[-1][0],encoder_arch[-1][1]))
        modules.append(torch.nn.Tanh())
        encoder = nn.Sequential(*modules)        
        return encoder
        
    def Decoder(self, decoder_arch):
        modules = []
        for i in range(len(decoder_arch)-1):
            modules.append(torch.nn.Linear(decoder_arch[i][0],decoder_arch[i][1]))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(decoder_arch[-1][0],decoder_arch[-1][1]))
        modules.append(torch.nn.Tanh())
        decoder = nn.Sequential(*modules) 
        return decoder
    
    def forward(self,x,s):
        #concatenate slave and target
        tmp = torch.cat((s,x),1)
        y = self.encoder.forward(tmp)
        #concatenate slave, target, constraint sequentially
        tmp = torch.cat((tmp,y),1)
        #detach grad from encoder to decoder
        z = self.decoder.forward(tmp.detach())
        #z = self.decoder.forward(tmp)
        #return constraint and master
        return y,z
    
    def forward_s(self,x,c):
        y = self.encoder.forward(x)
        tmp = torch.cat((x,c),0)
        z = self.decoder.forward(tmp)        
        return y,z
    
class ExoskeletonAENet(nn.Module):    
    def __init__(self, encoder_arch):
        super().__init__()  
        modules = []        
        for i in range(len(encoder_arch)-1):
            modules.append(torch.nn.Linear(encoder_arch[i][0],encoder_arch[i][1]))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(encoder_arch[-1][0],encoder_arch[-1][1]))
        modules.append(torch.nn.Tanh())
        self.net = nn.Sequential(*modules)     
        
    def forward(self, x):
        return self.net.forward(x)  

        
