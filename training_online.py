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
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
bind_addr = ('127.0.0.1',9180)
send_addr = ('127.0.0.1',9120)
s.bind(bind_addr)


recv_cmd = {}

# encoder_arch = [[6,14],[14,28],[28,28],[28,6]]
# decoder_arch = [[12,14],[14,28],[28,28],[28,7]]
# net = network.Net(encoder_arch,decoder_arch)
# dataset = exoskeleton_dataset.ExoskeletonDataset(file="data/exoskeleton_data",root_dir="./")
# sample_d = dataset[0]
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# loss = nn.MSELoss()
columns = ['index', 'target','master', 'constrains','slave']


def udprecv(q):
    while True:
        global recv_cmd
        line = s.recv(256)
        q.put(line)

        #print("target:{}\n".format(target_cmd))




def exoskeleton_ae_network():
    encoder_arch = [[14,28],[28,56],[56,56],[56,4]]
    decoder_arch = [[18,36],[36,72],[72,72],[72,7]]

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

        tmp = [float(x.strip('\0')) for x in tmp]
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
def test_data(model, data):
    realtime_loss = nn.MSELoss()
    tensor_data = data_to_tensor(data)
    target = tensor_data["target"].unsqueeze(dim=1)

    constrains = tensor_data["constrains"].unsqueeze(dim=1)

    master = tensor_data["master"].unsqueeze(dim=1)

    slave = tensor_data["slave"].unsqueeze(dim=1)
    y, z = model.forward(torch.transpose(target, 0, 1), torch.transpose(slave, 0, 1))
    rl = realtime_loss.forward(constrains, y)
    print(y)
    print(torch.transpose(constrains,0,1))
    print(z)
    print(torch.transpose(master,0,1))
    print("loss:",rl)

def main():
    q = Queue()
    dataset = exoskeleton_dataset.ExoskeletonDataset(
        file="data/exo_data_3", root_dir="/")
    train_dataset, val_dataset, test_dataset = dataset.GetDataset()

    t_udp = threading.Thread(target=udprecv, args=(q,))
    t_udp.start()
    encoder_arch = [[14,28],[28,56],[56,56],[56,4]]
    decoder_arch = [[18,36],[36,72],[72,72],[72,7]]
    model = network.Net(encoder_arch, decoder_arch)


    #model.load_state_dict(torch.load('./model'))
    model.decoder.load_state_dict(torch.load("decoder.model"))
    model.encoder.load_state_dict(torch.load("encoder.model"))
    realtime_loss = nn.MSELoss()


    while True:

        string = q.get()
        tensor_data = string_to_tensor(string)
        target = torch.Tensor()
        constrains = torch.Tensor()
        master = torch.Tensor()
        slave = torch.Tensor()



        target = tensor_data["target"].unsqueeze(dim=1)

        constrains = tensor_data["constrains"].unsqueeze(dim=1)

        master = tensor_data["master"].unsqueeze(dim=1)

        slave = tensor_data["slave"].unsqueeze(dim=1)
        if target.__len__() != 7:
            continue
        if constrains.__len__() != 4:
            continue
        if slave.__len__() != 7:
            continue

        tmp = torch.cat((slave, target), 0)
        y = model.encoder.forward(torch.transpose(tmp, 0, 1))

        rl = realtime_loss.forward(constrains,y)
        jc = np.array(y.detach().numpy())
        jc_c = np.array(constrains.detach().numpy())


        data_str = "%f,%f,%f,%f" % (jc_c[0], jc_c[1], jc_c[2], jc_c[3])


        print("calculated:{}".format(data_str))

        data_str = "%f,%f,%f,%f" % (jc[0][0], jc[0][1], jc[0][2], jc[0][3])

        print("predicted:{}".format(data_str))

        print(rl.detach())
        s.sendto(data_str.encode('utf-8'),send_addr)


if __name__ == "__main__":
    main()
