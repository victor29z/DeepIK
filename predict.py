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

import os
import socket
import threading,time
import numpy as np
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
bind_addr = ('127.0.0.1',9180)
send_addr = ('127.0.0.1',9120)
s.bind(bind_addr)
target_cmd = np.zeros(7)



# encoder_arch = [[6,14],[14,28],[28,28],[28,6]]
# decoder_arch = [[12,14],[14,28],[28,28],[28,7]]
# net = network.Net(encoder_arch,decoder_arch)
# dataset = exoskeleton_dataset.ExoskeletonDataset(file="data/exoskeleton_data",root_dir="./")
# sample_d = dataset[0]
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# loss = nn.MSELoss()


def udprecv():
    while True:
        global target_cmd
        data = s.recv(256)
        tmp = np.fromstring(data,dtype=np.float32, sep=' ')

        target_cmd = tmp

        #print("target:{}\n".format(target_cmd))


def exoskeleton_ae_network():
    encoder_arch = [[7, 14], [14, 28], [28, 56], [56, 4]]
    decoder_arch = [[18, 36], [36, 72], [72, 72], [72, 7]]

    encoder = network.ExoskeletonAENet(encoder_arch)
    decoder = network.ExoskeletonAENet(decoder_arch)
    return encoder, decoder





def main():
    t_udp = threading.Thread(target=udprecv)
    t_udp.start()


    encoder, decoder = exoskeleton_ae_network()

    #model.load_state_dict(torch.load('./model'))
    #model.decoder.load_state_dict(torch.load("decoder.model"))
    #model.encoder.load_state_dict(torch.load("encoder.model"))

    encoder.net.load_state_dict(torch.load("encoder.model"))
    while True:
        target_cmd.dtype = 'float32'
        if target_cmd.__len__() != 7:
            continue
        #convert received data to torch tensor
        target_tensor = torch.from_numpy(target_cmd)


        y = encoder.forward(target_tensor)

        jc = np.array(y.detach().numpy())

        data_str = "%f,%f,%f,%f" % (jc[0],jc[1],jc[2],jc[3])

        print("str:{}".format(data_str))
        s.sendto(data_str.encode('utf-8'),send_addr)


if __name__ == "__main__":
    main()
