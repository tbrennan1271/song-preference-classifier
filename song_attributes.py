#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:38:22 2020

@author: tylerbrennan
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import pandas as pd

PLAYLIST = 'Music.csv'
CURRENT_TOP = 'Top_Playlist.csv' 
LABEL = 'LIKE'
THRESHOLD_LIM = 5       # The number of sections the attributes will be divided into for the information gain
TOP_N = 3               # Top n information gains to use in decision tree

# In: name: file name  &  label: True if the data contains labels for data, and False if regular data
# Out: data: array that contains all the data
def get_data(name):
    global DIM
    attribute = []
    data = {}
    with open(name, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for entry in reader:
            DIM = len(entry)
            if len(attribute) != 0:
                for i in range(DIM - 1):
                    i += 1                          # This ignores the '#' category, which only counts songs in the playlist
                    entry[i] = entry[i].strip('-')
                    if entry[i].isdigit():
                        data[attribute[i - 1]].append(int(entry[i]))
                    else:
                        data[attribute[i - 1]].append(entry[i])
            else:
                for i in range(DIM - 1):
                    attribute.append(entry[i + 1])
                    data[attribute[i]] = []
                
    return data, attribute

# Just creates an array of attributes that doesn't contain string values or the labels
def att_without_str(data, attribute):
    att_no_str = []
    for att in attributes:
        if att != LABEL and (type(data[att][0]) == float or type(data[att][0]) == int):
                att_no_str.append(att)
    return att_no_str

# Puts all the points within the range of 0 and 1
def standardize(data, attributes):
    for att in attributes:
        minimum = float(min(data[att]))
        maximum = float(max(data[att]) - minimum)
        for i in range(len(data[att])):
            data[att][i] = float(data[att][i] - minimum) / maximum
    return data

# Visualizes the distribution with histograms
def create_histograms(data, data_pop, attributes):
    for att in attributes:
        temp = []
        temp_like = []
        temp_not_like = []
        for feature in data[att]:
            if data[LABEL][data[att].index(feature)] == 1:
                temp_like.append(feature)
            else:
                temp_not_like.append(feature)
        plt.hist(temp_like, alpha = .5, label = 'My music')
        plt.hist(temp_not_like, alpha = .5, label = "Music I don't like")
        for feature in data_pop[att]:
            temp.append(feature)
        plt.hist(temp, alpha = .5, label = 'Top 50')
        plt.legend(loc='upper right')
        plt.title(att)
        plt.show()
        print('My avg', att, ':', np.average(temp))
        print('Top avg', att, ':', np.average(temp))
            
def create_groups(data, attributes):
    label_groups = {}
    group = {}
    threshold = 1/THRESHOLD_LIM
    for att in attributes:
        label_groups[att] = {}
        group[att] = {}
        for i in range(THRESHOLD_LIM):
            val = (i + 1) * threshold
            label_groups[att][val] = []
            group[att][val] = [0, 0]        # [deximal value of liked points,  overall weight decimal value]
        for i in range(len(data[att])):
            pt = data[att][i]
            pt_label = data[LABEL][i]
            if pt == 0:
                if pt_label == 1:
                    group[att][threshold][0] += 1
                group[att][threshold][1] += 1
            for val in label_groups[att]:
                if pt <= val and pt > (val - threshold):
                    group[att][val][1] += 1
                    if pt_label == 1:
                        group[att][val][0] += 1
        for val in label_groups[att]:
            group[att][val][1] /= len(data[LABEL])
            group[att][val][0] /= len(data[att])
    return group

def information_gain(groups, attributes):
    ig = {}
    for att in attributes:
        final = 0
        for pt in groups[att]:
            entropy = 0
            val = groups[att][pt][0]
            weight = groups[att][pt][1]
            if val != 0:
                entropy -= (val * np.log2(val))
                entropy -= ((1 - val) * np.log2(1 - val))
            final += entropy * weight
        ig[att] = final
    return ig


data_pop, attributes = get_data(CURRENT_TOP)
data, attributes = get_data(PLAYLIST)
int_att = att_without_str(data, attributes)
data = standardize(data, int_att)
data_pop = standardize(data_pop, int_att)
create_histograms(data, data_pop, int_att)
groups = create_groups(data, int_att)
info_gain = information_gain(groups, int_att)
print(info_gain)



