# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:05:28 2020

@author: yan
"""
my_list = [];

with open('dest_log_3') as f:
    lines = f.readlines() # list containing lines of file
    
    columns = ['index', 'end-effector-pose','constrained-pose',
               'master-joint-pose','slave-joint-pose'] 
#    columns = ['index', 'ee-x','ee-y','ee-z','ee-ex','ee-ey','ee-ez','c-x','c-y','c-z','c-ex','c-ey','c-ez',
#               'm-j1','m-j2','m-j3','m-j4','m-j5','m-j6','m-j7','s-j1','s-j2','s-j3','s-j4','s-j5','s-j6','s-j7'] # To store column names

#    i = 1
    for line in lines:
        line = line.strip() # remove leading/trailing white spaces
        if line:
#            if i == 1:
#                columns = [item.strip() for item in line.split(',')]
#                i = i + 1
#            else:
            d = {} # dictionary to store file data (each line)
            data = [item.strip() for item in line.split(',')]
            for index, elem in enumerate(data):
                d[columns[index]] = data[index]
    
            my_list.append(d) # append dictionary to list