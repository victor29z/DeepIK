#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 13:47:59 2020

@author: yan
"""
from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

class ExoskeletonDataset(Dataset):
    """ Exoskeleton Dataset."""
   
    def __init__(self, file, root_dir, transform = None):
        """
        file (string): Exoskeleton data file name.
        root_dir(string): Directory with all the data.
        transform (callable, optional): Optional transform to be applied
                on a sample.
        """        
        self.data_file = file
        self.root_dir = root_dir
        self.transform = transform
        self.read()
        self.train = None#[]
        self.val = None#[]
        self.test = None#[]#np.array([])

    def __len__(self):
        return len(self.exo_data)    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        master_pose = self.exo_data[idx]['master']
        slave_pose = self.exo_data[idx]['slave']
        constrain_pose = self.exo_data[idx]['constrains']
        target_pose = self.exo_data[idx]['target']
        sample = {'master': master_pose, 'slave': slave_pose ,
                  'constrians': constrain_pose, 'target': target_pose}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def read(self):
        file = self.data_file
        self.exo_data = [];
        with open(file) as f:
            lines = f.readlines() # list containing lines of file
            
            columns = ['index', 'target','master', 'constrains','slave']
            for line in lines:
                line = line.strip() # remove leading/trailing white spaces
                if line:
                    d = {} # dictionary to store file data (each line)
                    data = [item.strip() for item in line.split(',')]
                    for index, elem in enumerate(data):
                        #d[columns[index]] = data[index]
                        #tmp = " ".join(elem.split())
                        #tmp = [item.strip() for item in tmp.split(' ')]
                        tmp = elem.split()
                        tmp = [float(x) for x in tmp]
#                        tmp = [np.sin(float(x)) for x in tmp]
                        d[columns[index]] = tmp#/math.pi
#                        d[columns[index]] = tmp            
                    self.exo_data.append(d) # append dictionary to list

    def GetDataset(self):
        l = list(range(len(self.exo_data)))
        print(len(self.exo_data))
        sz = len(l)
        cut_t = int(0.6 * sz) #70% of the list
        cut_v = cut_t + int(0.2 * sz)
        random.shuffle(l)
        ltrain = l[:cut_t] # first 80% of shuffled list
        lval = l[cut_t:cut_v]
        ltest = l[cut_v:] # last 20% of shuffled list
        
        self.train = np.array(self.exo_data)[np.array(ltrain)]
        self.val = np.array(self.exo_data)[np.array(lval)]
        self.test = np.array(self.exo_data)[np.array(ltest)]

        return self.train, self.val, self.test


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        master = sample['master']
        constrains = sample['constrains']
        target = sample['target']
        slave = sample['slave']
        return {'master': torch.Tensor(np.array(master)),
                'constrains': torch.Tensor(np.array(constrains)),
                'target': torch.Tensor(np.array(target)),
                'slave': torch.Tensor(np.array(slave))}
            