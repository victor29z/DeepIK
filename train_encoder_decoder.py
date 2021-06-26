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


def exoskeleton_ae_network():
    encoder_arch = [[7, 14], [14, 28], [28, 56], [56, 4]]
    decoder_arch = [[18, 36], [36, 72], [72, 72], [72, 7]]
    
    encoder = network.ExoskeletonAENet(encoder_arch)
    decoder = network.ExoskeletonAENet(decoder_arch)    
    return encoder, decoder


def decoder_training(decoder, loss_de, optimizer_de,
            train_dataset):
    target = torch.Tensor()
    constrains = torch.Tensor()
    master = torch.Tensor()
    slave = torch.Tensor()

    tt = exoskeleton_dataset.ToTensor()

    for item in train_dataset:
        tensor_data = tt(item)
        tmp = tensor_data["target"].unsqueeze(dim=1)
        target = torch.cat((target, tmp), 1)
        tmp = tensor_data["constrains"].unsqueeze(dim=1)
        constrains = torch.cat((constrains, tmp), 1)
        tmp = tensor_data["master"].unsqueeze(dim=1)
        master = torch.cat((master, tmp), 1)
        tmp = tensor_data["slave"].unsqueeze(dim=1)
        slave = torch.cat((slave, tmp), 1)

    optimizer_de.zero_grad()
    # optimizer_de.zero_grad()
    # concat slave, target, constraint sequentially
    tmp = torch.cat((slave, target), 0)
    tmp = torch.cat((tmp,constrains),0)
    y = decoder.forward(torch.transpose(tmp, 0, 1))
    ld = loss_de.forward(torch.transpose(y, 0, 1), master)


    ld.backward()

    optimizer_de.step()
    return ld.item()

def train_s(encoder, loss_en, optimizer_en,
         train_dataset):
    target = torch.Tensor()
    constrains = torch.Tensor()
    master = torch.Tensor()
    slave = torch.Tensor()
        
    tt = exoskeleton_dataset.ToTensor()
    
    for item in train_dataset:        
        tensor_data = tt(item)
        tmp = tensor_data["target"].unsqueeze(dim=1)
        target = torch.cat((target, tmp), 1)
        tmp = tensor_data["constrains"].unsqueeze(dim=1)
        constrains = torch.cat((constrains, tmp), 1)
        tmp = tensor_data["master"].unsqueeze(dim=1)
        master = torch.cat((master, tmp), 1)
        tmp = tensor_data["slave"].unsqueeze(dim=1)
        slave = torch.cat((slave, tmp), 1)
    
    optimizer_en.zero_grad()
    #optimizer_de.zero_grad()

    tmp = torch.cat((slave, target), 0)
    x = encoder.forward(torch.transpose(target,0,1))
    #tmp = torch.cat((tmp,constrains),0)
    #y = decoder.forward(torch.transpose(tmp,0,1))
    
    le = loss_en.forward(torch.transpose(x,0,1), constrains)
    #ld = loss_de.forward(torch.transpose(y,0,1),master)
    
    le.backward()
    #ld.backward()
    
    optimizer_en.step()
    #optimizer_de.step()
    #return le.item(), ld.item()
    return le.item()

def train(model, loss, optimizer, train_dataset):   
    target = torch.Tensor()
    constrains = torch.Tensor()
    master = torch.Tensor()
    slave = torch.Tensor()
    tt = exoskeleton_dataset.ToTensor()
    
    for item in train_dataset:        
        tensor_data = tt(item)
        tmp = tensor_data["target"].unsqueeze(dim=1)
        target = torch.cat((target, tmp), 1)
        tmp = tensor_data["constrains"].unsqueeze(dim=1)
        constrains = torch.cat((constrains, tmp), 1)
        tmp = tensor_data["master"].unsqueeze(dim=1)
        master = torch.cat((master, tmp), 1)
        tmp = tensor_data["slave"].unsqueeze(dim=1)
        slave = torch.cat((slave, tmp), 1)
    
    optimizer.zero_grad()
    y,z = model.forward(torch.transpose(target,0,1),torch.transpose(slave,0,1))
    l = loss.forward(z,torch.transpose(master,0,1))
    
    l.backward()
    optimizer.step()
    return l.item()


def predict(encoder, target):    
    return encoder.forward(target)

def train_separately(train_dataset, val_dataset, test_dataset):    
    loss_en = nn.MSELoss()

    loss_de = nn.MSELoss()

    n_examples = len(train_dataset)  
    encoder, decoder = exoskeleton_ae_network()
    
    optimizer_en = optim.SGD(encoder.parameters(), lr=0.01, momentum=0.9)
    optimizer_de = optim.SGD(decoder.parameters(), lr=0.01, momentum=0.9)


    decoder.net.load_state_dict(torch.load("decoder.model"))

    encoder.net.load_state_dict(torch.load("encoder.model"))
    batch_size = 500
    num_batches = n_examples // batch_size
    f = open("loss.txt","w+")
    i = 0
    for i in range(8000):
        cost_le = 0.
        cost_ld = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            '''
            le, ld = train_s(encoder, decoder, loss_en, loss_de, optimizer_en, 
                          optimizer_de, train_dataset[start: end])
            '''
            le = train_s(encoder, loss_en, optimizer_en,
                              train_dataset[start: end])
            cost_le += le

            
        print("Epoch = {epoch}, cost_le = {le}".format(
        epoch = i+1, le = cost_le / num_batches))


        f.write("%f\n"%(cost_le))

        if cost_le < 0.005:
            break

#        print ("Epoch %d, cost_le = %f, cost_ld = %f", i + 1, 
#               cost_le / num_batches, cost_ld / num_batches,)
#        predY = predict(model, val_dataset)
#        print("Epoch %d, cost = %f, acc = %.2f%%"
#              % (i + 1, cost / num_batches, 100. * np.mean(predY == teY)))

    torch.save(encoder.net.state_dict(), "encoder.model")
    #torch.save(decoder.net.state_dict(), "decoder.model")


def train_decoder(train_dataset, val_dataset, test_dataset):
    loss_en = nn.MSELoss()

    loss_de = nn.MSELoss()

    n_examples = len(train_dataset)
    encoder, decoder = exoskeleton_ae_network()

    optimizer_en = optim.SGD(encoder.parameters(), lr=0.01, momentum=0.9)
    optimizer_de = optim.SGD(decoder.parameters(), lr=0.01, momentum=0.9)

    #decoder.net.load_state_dict(torch.load("decoder.model"))

    #encoder.net.load_state_dict(torch.load("encoder.model"))
    batch_size = 500
    num_batches = n_examples // batch_size
    f = open("loss.txt", "w+")
    i = 0
    for i in range(8000):
        cost_le = 0.
        cost_ld = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            '''
            le, ld = train_s(encoder, decoder, loss_en, loss_de, optimizer_en, 
                          optimizer_de, train_dataset[start: end])
            '''
            ld = decoder_training(decoder, loss_de, optimizer_de,
                         train_dataset[start: end])
            cost_ld += ld

        print("Epoch = {epoch}, cost_ld = {ld}".format(
            epoch=i + 1, le=cost_ld / num_batches))

        f.write("%f\n" % (cost_ld))
        f.flush()
        if cost_ld < 0.005:
            break

    #        print ("Epoch %d, cost_le = %f, cost_ld = %f", i + 1,
    #               cost_le / num_batches, cost_ld / num_batches,)
    #        predY = predict(model, val_dataset)
    #        print("Epoch %d, cost = %f, acc = %.2f%%"
    #              % (i + 1, cost / num_batches, 100. * np.mean(predY == teY)))

    torch.save(encoder.net.state_dict(), "decoder.model")
    # torch.save(decoder.net.state_dict(), "decoder.model")


def main():
    dataset = exoskeleton_dataset.ExoskeletonDataset(
            file="data/nc_agger2/constraint_regression",root_dir="/")
    train_dataset, val_dataset, test_dataset = dataset.GetDataset()
    

    #train_separately(train_dataset, val_dataset, test_dataset)
    train_decoder(train_dataset, val_dataset, test_dataset)
    #print("Separate network train done! Begin whole network training...")
    
    #train_wholenet(train_dataset, val_dataset, test_dataset)


if __name__ == "__main__":
    main()
