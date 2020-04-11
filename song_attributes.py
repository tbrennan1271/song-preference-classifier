#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:38:22 2020

@author: tylerbrennan
"""

import csv
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree


''' ---------- SETTINGS ---------- '''

PLAYLIST = 'Music.csv'
CURRENT_TOP = 'Top_Playlist.csv' 
LABEL = 'LIKE'
THRESHOLD_LIM = 5       # The number of sections the attributes will be divided into for the information gain
TOP_N = 3               # Top n information gains to use in decision tree
TEST_PERCENT = .2


''' ---------- METHODS ---------- '''

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

# Splits data based on TEST_PERCENT
def data_split(input_data):
    training_data = copy.copy(input_data)
    test_data = {}
    for key in input_data:
        test_data[key] = []
    for i in range(int(len(input_data[LABEL]) * TEST_PERCENT)):
        index = random.randrange(len(training_data[LABEL]))
        for key in input_data:
            test_data[key].append(training_data[key].pop(index)) 
    return training_data, test_data

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
        for i in range(len(data[att])):
            feature = data[att][i]
            if data[LABEL][i] == 1:
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
        print('My avg', att, ':', np.average(temp_like))
        print('Not like avg', att, ':', np.average(temp_not_like))
        print('Top avg', att, ':', np.average(temp))
            
# Creates groups to be used to calculate information gain/fit the decision tree
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

# Calculates the information gain for each attribute
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

# Determines the top n information gains and fits a decision tree based on that
def decision_tree(info_gain, test, train_label):
    top_n = {}
    top_att = []
    dt_input = []
    for i in range(TOP_N):
        max_ig = 0
        max_att = ''
        for att in info_gain:
            ig = info_gain[att]
            if ig > max_ig:
                max_ig = ig
                max_att = att
        top_n[max_att] = max_ig
        info_gain.pop(max_att)
        top_att.append(max_att)
        dt_input.append(test[max_att])
    arr = []
    for i in range(len(dt_input[0])):   # Done to fit the format of the decision tree where each point is its own array
        output = [0] * TOP_N
        for j in range(TOP_N):
            output[j] = dt_input[j][i]
        arr.append(output)
    print(top_att)
    clf = tree.DecisionTreeClassifier(criterion = "entropy", splitter = "best", max_depth = 3, max_features = TOP_N)
    clf = clf.fit(arr, train_label)
    return clf


''' ---------- MAIN ---------- '''

data_pop, attributes = get_data(CURRENT_TOP)
data, attributes = get_data(PLAYLIST)
int_att = att_without_str(data, attributes)
create_histograms(data, data_pop, int_att)
train, test = data_split(data)
train = standardize(train, int_att)
groups = create_groups(train, int_att)
info_gain = information_gain(groups, int_att)
    
clf = decision_tree(info_gain, train, train[LABEL])
tree.plot_tree(clf) 
