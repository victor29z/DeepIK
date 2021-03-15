# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:31:44 2020

@author: yan
"""

import math
import numpy as np
import matplotlib.pyplot as plt

file = 'data/exoskeleton_data_new'
exo_data = [];
with open(file) as f:
    lines = f.readlines() # list containing lines of file
    
    columns = ['index', 'target','constrains', 'master','slave']
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
            exo_data.append(d) # append dictionary to list
            
            
#plt.plot(exo_data)

master = []

constrains = []
target = []
            
for item in exo_data:
    master.append(item["master"])
    constrains.append(item["constrains"])
    target.append(item["target"])
    
master_data = np.array(master)
constrains_data = np.array(constrains)
target_data = np.array(target)
plt.plot(master_data)
plt.figure()
plt.plot(constrains_data)
plt.figure()
plt.plot(target_data)
