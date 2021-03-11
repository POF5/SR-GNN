#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/16 4:36
# @Author : {ZM7}
# @File : model.py
# @Software: PyCharm
# import tensorflow as tf
import math
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, nonhybrid=True):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.mask = tf.placeholder(dtype=tf.float32)  # 变量
        self.alias = tf.placeholder(dtype=tf.int32)  # 给给每个输入重新
        self.item = tf.placeholder(dtype=tf.int32)  # 重新编号的序列构成的矩阵
        self.tar = tf.placeholder(dtype=tf.int32)
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.nasr_w1 = tf.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))  # 需要训练的变量
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())

    def forward(self, re_embedding, train=True): # re_embedding训练后的嵌入向量

        rm = tf.reduce_sum(self.mask, 1)  # mask每一行分别求和
        last_id = tf.gather_nd(self.alias,
                               tf.stack([tf.range(self.batch_size), tf.to_int32(rm) - 1], axis=1))  # 取序列最后一个的id
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1)) # 取最后一个节点的嵌入向量 batch_size*hidden_size
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(self.batch_size)],
                         axis=0)  # batch_size*T*d  //batch_size*maxlen*hidden_size
        last = tf.matmul(last_h, self.nasr_w1)   # w1*vn , batch_size*hidden_size
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)   # w2*vi , (batch_size*maxlen)*hidden_size
        last = tf.reshape(last, [self.batch_size, 1, -1]) # batch_size * 1 * hidden_size
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b) # 取激活函数sigmoid , batch_size*maxlen*out_size
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(   # sg(未求和) , (batch_size*maxlen)*out_size
            self.mask, [-1, 1]) # coef: (batch_size*maxlen)*1
        b = self.embedding[1:] # n_node*out_size
        if not self.nonhybrid:  # 拼接si和sg
            ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1), #batch_size*out_size  (为什么要乘一个seq_h再求和?
                            tf.reshape(last, [-1, self.out_size])], -1) # ma: batch_size*(2*out_size) ==论文中的sh
            self.B = tf.get_variable('B', [2 * self.out_size, self.out_size],
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(ma, self.B)  # batch_size*out_size
            logits = tf.matmul(y1, b, transpose_b=True) # batch_size*n_node
            # print(y1, "logits?????\n\n\n\n\n")
        else:   # 仅用sg
            ma = tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1)
            logits = tf.matmul(ma, b, transpose_b=True)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.trainable_variables()
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2 #正则化
            loss = loss + lossL2
        return loss, logits

    def run(self, fetches, tar, item, adj_in, adj_out, alias, mask):
        Sess = self.sess.run(fetches, feed_dict={self.tar: tar, self.item: item, self.adj_in: adj_in, self.adj_out: adj_out, self.alias: alias, self.mask: mask})
        return Sess



class GGNN(Model):
    def __init__(self, hidden_size=100, out_size=100, batch_size=300, n_node=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1, nonhybrid=False):
        super(GGNN, self).__init__(hidden_size, out_size, batch_size, nonhybrid)
        self.embedding = tf.get_variable(shape=[n_node, hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))  # 每个节点的嵌入向量
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.n_node = n_node
        self.L2 = l2
        self.step = step
        self.nonhybrid = nonhybrid
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        with tf.variable_scope('ggnn_model', reuse=None):
            self.loss_train, _ = self.forward(self.ggnn())
        with tf.variable_scope('ggnn_model', reuse=True):
            self.loss_test, self.score_test = self.forward(self.ggnn(), train=False)
        self.global_step = tf.Variable(0)

        self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)   # 动态衰减的学习率
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step) # 学习到的参数
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        logdir='D:\\tf_dir\\tensorboard_study'
        if tf.gfile.Exists(logdir):
            tf.gfile.DeleteRecursively(logdir)
        writer = tf.summary.FileWriter(logdir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())  #初始化变量
        # saver = tf.train.import_meta_graph("Model/model.ckpt.meta")
        # saver.restore(self.sess, "./Model/model.ckpt")  # 注意路径写法
        writer.close()

    def ggnn(self):
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item)  # 获取每个序列中每个节点的嵌入表示？
        cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size]) # batch_size*maxlen*hidden_size
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]), # 这个reshape相当于不分组
                                                    self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])# 矩阵乘法结果还是 batch_size*maxlen*hidden_size
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [self.batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(self.adj_in, fin_state_in),   #batch_size*maxlen*(2*hiden_size)
                                tf.matmul(self.adj_out, fin_state_out)], axis=-1)   #结果为batch_size*maxlen*hidden_size最内层连接[[[a1,a2,a3,a4,b1,b2,b3,b4]]]
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2 * self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
    def reload(self, path = '../Model/model.ckpt.meta'):
        saver = tf.train.Saver()
        saver.restore(self.sess, "Model/model.ckpt")
        # saver = tf.train.import_meta_graph("Model/model.ckpt.meta")
        # saver.restore(self.sess, "/Model/model.ckpt")  # 注意路径写法
        # print(self.nasr_b.eval())
        # print(self.sess.run(self.nasr_w1))
        # saver = tf.train.import_meta_graph('Model/model.ckpt.meta')
        # saver.restore(self.sess, tf.train.latest_checkpoint("./Model.model.ckpt"))
