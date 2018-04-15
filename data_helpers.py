#-*- coding: utf-8 -*-
import numpy as np
import re
import itertools
from collections import Counter
get_one_hot = lambda x: [[[0]]] * x + [[[1]]] + [[[0]]] * (3 - x)


def load_data(train = 'data/train.txt',test = 'data/test.txt'):
    f = open(train)
    data = []
    type_list = []
    labels = []
    for line in f:
        _ = line.strip().split('\t')
        if _[0] == '1':
            label = [0, 1]
        elif _[0] == '0':
            label = [1, 0]
        else:
            print _[0]
        type = get_one_hot(int(_[1]))
        text = _[2]
        data.append(text)
        type_list.append(type)
        labels.append(label)
    f = open(test)
    for line in f:
        _ = line.strip().split('\t')
        if _[0] == '1':
            label = [0, 1]
        elif _[0] == '0':
            label = [1, 0]
        else:
            print _[0]
        type = get_one_hot(int(_[1]))
        text = _[2]
        data.append(text)
        type_list.append(type)
        labels.append(label)
    return [np.array(data),type_list,np.array(labels)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
