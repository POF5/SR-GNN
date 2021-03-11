#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/17 5:40
# @Author : {ZM7}
# @File : main.py
# @Software: PyCharm

from __future__ import division
from tensorflow_code.model import *
from tensorflow_code.utils import build_graph, Data, split_validation
import numpy as np
import pickle
import argparse
import datetime
import tensorflow.compat.v1 as tf

parser = argparse.ArgumentParser()      #读取命令行参数
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--method', type=str, default='ggnn', help='ggnn/gat/gcn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
opt = parser.parse_args()

train_data = pickle.load(open('./datasets/' + opt.dataset + '/train2.txt', 'rb'))    #读取数据文件反序列化为对象
test_data = pickle.load(open('./datasets/' + opt.dataset + '/test2.txt', 'rb'))
# all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
if opt.dataset == 'diginetica':
    n_node = 43098
elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    n_node = 37484
else:
    #TODO :先获取数据集节点总个数
    n_node = 65349 #节点总个数
# g = build_graph(all_train_seq)
train_data = Data(train_data, sub_graph=True, method=opt.method, shuffle=True)
test_data = Data(test_data, sub_graph=True, method=opt.method, shuffle=False)
model = GGNN(hidden_size=opt.hiddenSize, out_size=opt.hiddenSize, batch_size=opt.batchSize, n_node=n_node,
             lr=opt.lr, l2=opt.l2, step=opt.step, decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize,
             lr_dc=opt.lr_dc,
             nonhybrid=opt.nonhybrid)
model.reload()
saver = tf.train.Saver()
print(model.sess.run(model.embedding))
best_result = [0, 0]
best_epoch = [0, 0]


for epoch in range(opt.epoch):
    print('epoch: ', epoch, '===========================================')
    # slices = train_data.generate_batch(model.batch_size)    #返回Batch序列[[0,1,2],[3,4,5]]这样，可能打乱原序列顺序
    # fetches = [model.opt, model.loss_train, model.global_step]  # opt为训练得到的最优参数，loss_train损失函数值，global_step最终迭代轮数
    # print('start training: ', datetime.datetime.now())
    # loss_ = []
    # for i, j in zip(slices, np.arange(len(slices))):
    #     adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
    #     _, loss, _ = model.run(fetches, targets, item, adj_in, adj_out, alias, mask)
    #     loss_.append(loss)
    slices = test_data.generate_batch(model.batch_size)
    print('start predicting: ', datetime.datetime.now())
    hit, mrr, test_loss_ = [], [], []
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i)
        scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, adj_in, adj_out, alias, mask) # 预测时更新计算分数值和损失函数值
        test_loss_.append(test_loss)
        index = np.argsort(scores, 1)[:, -20:] #取每行后20个
        # print(len(index[0]),"??????\n\n\n\n\n\n\n\n")
        for score, target in zip(index, targets):
            hit.append(np.isin(target - 1, score)) #命中
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (20 - np.where(score == target - 1)[0][0]))  #越后面分越高
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    print(hit,mrr, "???????\n\n\n\n\n\n")
    test_loss = np.mean(test_loss_)
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1] = epoch
    # print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' %
    #       (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
# saver = tf.train.Saver()
# saver.save(model.sess, "Model/model.ckpt")
# print("save model...")
print(model.sess.run(model.nasr_b))