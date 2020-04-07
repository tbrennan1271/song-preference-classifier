#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:38:22 2020

@author: tylerbrennan
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

PLAYLIST = 'Music.csv'
CURRENT_TOP = 'Top_Playlist.csv' 
LABEL = 'LIKE'

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
    print(att_no_str)
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


data_pop, attributes = get_data(CURRENT_TOP)
data, attributes = get_data(PLAYLIST)
int_att = att_without_str(data, attributes)
data = standardize(data, int_att)
data_pop = standardize(data_pop, int_att)
create_histograms(data, data_pop, int_att)