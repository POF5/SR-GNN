#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/9/23 2:52
# @Author : {ZM7}
# @File : utils.py
# @Software: PyCharm

import networkx as nx
import numpy as np


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):

    us_lens = [len(upois) for upois in all_usr_pois]        #每个序列的长度
    len_max = max(us_lens)  #最大长度
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)] #将每个序列补0补充到len_max
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens] #序列mask定义为有位填1，否则为0
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, sub_graph=False, method='ggnn', sparse=False, shuffle=False):

        inputs = data[0]        #获取序列
        inputs, mask, len_max = data_masks(inputs, [0]) #
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.sub_graph = sub_graph
        self.sparse = sparse
        self.method = method

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        print("nbatch: " , n_batch,"???????")
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        if 1:
            items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
            for u_input in self.inputs[index]: #取该index batch对应序列
                n_node.append(len(np.unique(u_input))) #每个序列有多少个非重复元素（建图节点个数）
            max_n_node = np.max(n_node)
            if self.method == 'ggnn':
                for u_input in self.inputs[index]:
                    node = np.unique(u_input)
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])    #item为序列去重后补充0到最大长度
                    u_A = np.zeros((max_n_node, max_n_node))    #全0 方阵
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1 #建图，序号为去重后序号
                    u_sum_in = np.sum(u_A, 0)   #按列相加得到入度向量
                    u_sum_in[np.where(u_sum_in == 0)] = 1
                    u_A_in = np.divide(u_A, u_sum_in)   #入度矩阵定义为边数除以入度
                    u_sum_out = np.sum(u_A, 1)  #按行相加得到出度向量
                    u_sum_out[np.where(u_sum_out == 0)] = 1
                    u_A_out = np.divide(u_A.transpose(), u_sum_out)

                    A_in.append(u_A_in)
                    A_out.append(u_A_out)
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) #序列去重后序号重定义
                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]
            elif self.method == 'gat':
                A_in = []
                A_out = []
                for u_input in self.inputs[index]:
                    node = np.unique(u_input)
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])
                    u_A = np.eye(max_n_node)
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1
                    A_in.append(-1e9 * (1 - u_A))
                    A_out.append(-1e9 * (1 - u_A.transpose()))
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]

        else:
            return self.inputs[index], self.mask[index], self.targets[index]