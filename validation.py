#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:32:16 2020

@author: yan
"""

import exoskeleton_dataset
import network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from queue import Queue

import os
import socket
import threading,time
import numpy as np


# encoder_arch = [[6,14],[14,28],[28,28],[28,6]]
# decoder_arch = [[12,14],[14,28],[28,28],[28,7]]
# net = network.Net(encoder_arch,decoder_arch)
# dataset = exoskeleton_dataset.ExoskeletonDataset(file="data/exoskeleton_data",root_dir="./")
# sample_d = dataset[0]
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# loss = nn.MSELoss()
columns = ['index', 'target','master', 'constrains','slave']


def exoskeleton_ae_network():
    encoder_arch = [[7, 14], [14, 28], [28, 56], [56, 4]]
    decoder_arch = [[18, 36], [36, 72], [72, 72], [72, 7]]

    encoder = network.ExoskeletonAENet(encoder_arch)
    decoder = network.ExoskeletonAENet(decoder_arch)
    return encoder, decoder

def predict(encoder, target):
    return encoder.forward(target)

def predict_w(model, target):
    y = model.encoder.forward(target)
    return y

def data_to_tensor(sample):
    master = sample['master']
    constrains = sample['constrains']
    target = sample['target']
    slave = sample['slave']
    return {'master': torch.Tensor(np.array(master)),
            'constrains': torch.Tensor(np.array(constrains)),
            'target': torch.Tensor(np.array(target)),
            'slave': torch.Tensor(np.array(slave))}

def string_to_tensor(string):
    d = {}  # dictionary to store file data (each line)

    data = [item.strip() for item in string.decode(encoding="utf-8").split(',')]

    for index, elem in enumerate(data):
        tmp = elem.split()

        tmp = [float(x) for x in tmp]
        d[columns[index]] = tmp  # /math.pi

    tensor_data = data_to_tensor(d)
    return tensor_data

def string_to_tensor_1(string):
    d = {}  # dictionary to store file data (each line)

    data = [item.strip() for item in string.split(',')]

    for index, elem in enumerate(data):
        tmp = elem.split()

        tmp = [float(x) for x in tmp]
        d[columns[index]] = tmp  # /math.pi

    tensor_data = data_to_tensor(d)
    return tensor_data

def test_data(data,rec_file):
    encoder, decoder = exoskeleton_ae_network()

    #encoder.net.load_state_dict(torch.load("encoder.model"))

    encoder.net.load_state_dict(torch.load("trained_models/0309/encoder.model(ag2_18kep)"))

    predict_error = nn.MSELoss()
    tensor_data = data_to_tensor(data)
    target = tensor_data["target"].unsqueeze(dim=1)

    constrains = tensor_data["constrains"].unsqueeze(dim=1)

    master = tensor_data["master"].unsqueeze(dim=1)

    slave = tensor_data["slave"].unsqueeze(dim=1)

    tmp = torch.cat((slave, target), 0)
    #y = model.encoder.forward(torch.transpose(tmp, 0, 1))


    x = encoder.forward(torch.transpose(target, 0, 1))

    er = predict_error.forward(x.detach(),torch.transpose(constrains,0,1))
    print("predicted constraint:    {constraint_y}".format(constraint_y = x.detach()))
    print("calculated constraint:   {constraint_c}".format(constraint_c=torch.transpose(constrains,0,1).detach()))
    print("MSE:", er)
    rec_file.write("%f\n" % (er))




def main():

    dataset = exoskeleton_dataset.ExoskeletonDataset(
        file="data/validationset/va_set1",root_dir="/")
    train_dataset, val_dataset, test_dataset = dataset.GetDataset()

    f = open("mse_va1.txt", "w+")

    realtime_loss = nn.MSELoss()

    dataset_size = train_dataset.size
    for k in range(dataset_size):
        test_data(train_dataset[k],f)

    dataset = []
    dataset = exoskeleton_dataset.ExoskeletonDataset(
        file="data/validationset/va_set2", root_dir="/")
    train_dataset, val_dataset, test_dataset = dataset.GetDataset()
    f = open("mse_va2.txt", "w+")
    dataset_size = train_dataset.size
    for k in range(dataset_size):
        test_data(train_dataset[k],f)

    dataset = []
    dataset = exoskeleton_dataset.ExoskeletonDataset(
        file="data/validationset/va_set3", root_dir="/")
    train_dataset, val_dataset, test_dataset = dataset.GetDataset()
    f = open("mse_va3.txt", "w+")
    dataset_size = train_dataset.size
    for k in range(dataset_size):
        test_data(train_dataset[k], f)

    dataset = []
    dataset = exoskeleton_dataset.ExoskeletonDataset(
        file="data/validationset/va_set4", root_dir="/")
    train_dataset, val_dataset, test_dataset = dataset.GetDataset()
    f = open("mse_va4.txt", "w+")
    dataset_size = train_dataset.size
    for k in range(dataset_size):
        test_data(train_dataset[k], f)

if __name__ == "__main__":
    main()
