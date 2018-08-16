"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)
import time


# Deep Q Network off-policy
class DeepQNetwork:
    #图像参数
    width = 250
    height = 160
    deep = 3

    def __init__(
            self,
            n_actions,  # 几个可选择的动作
            n_features,  # cnn,输出维度,状态维度
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,  # 记忆库大小（行数)
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.width*self.height * self.deep*2 + 2))

        # consist of [target_net, evaluate_net]

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
        #self.saver.restore(self.sess, "my_net/save_net.ckpt")

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)


        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, 250, 160, 3],name='s') / 255.
        # ------------------ build evaluate_net ------------------
        #self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 100, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('cnn'):
                cnnValue = self._cnn(self.s)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(cnnValue, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, 250, 160, 3],name='s_') / 255.  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('cnn'):

                cnnValue = self._cnn(self.s_)
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(cnnValue, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2
    #卷积层
    def _cnn(self,tf_x):

        # CNN
        conv1 = tf.layers.conv2d(  # shape (250, 160, 3)
            inputs=tf_x,
            filters=24,
            kernel_size=(8,6),
            strides=(3,3),
            padding='valid',
            activation=tf.nn.relu
        )  # -> (81, 52, 24)
        print(conv1)
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=(3,2),
            strides=(3,2),
        )  # -> (27, 26, 24)
        print(pool1)

        conv2 = tf.layers.conv2d(pool1, 48, (3,2), 1, 'valid', activation=tf.nn.relu)  # -> (25, 25, 48)
        print(conv2)

        pool2 = tf.layers.max_pooling2d(conv2, 5, 5)  # -> (9, 13, 48)
        print(pool2)

        flat = tf.reshape(pool2, [-1,5*5* 48])  # -> (7*7*32, )
        print(flat)
        self.output = tf.layers.dense(flat,self.n_features)  # output layer
        return self.output

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s.reshape(-1), [a, r], s_.reshape(-1)))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # (20000,6) 开始时全为0,单次刷入记忆

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})  # (1,10)
            print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 随机抽取记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        # 一条记忆中，前两列是动作前状态，后两列是动作后状态,
        # 取得新神经网络的Q值(动作前状态s)和旧神经网络的Q值(动作后状态s_估算)
        # print(batch_memory.shape)#(32, 240002)
        #
        # print(batch_memory[:, -self.width*self.height*self.deep:].shape)   #(32, 120000)
        # print(batch_memory[:, -self.width*self.height*self.deep:].reshape(self.batch_size,self.width, self.height, self.deep))
        #

        s_f = batch_memory[:, -self.width*self.height*self.deep:].reshape(self.batch_size,self.width,self.height,self.deep)
        sf = batch_memory[:, :self.width*self.height*self.deep].reshape(self.batch_size,self.width,self.height,self.deep)

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_:s_f,  # fixed params
                self.s: sf,  # newest params
            })

        # change q_target w.r.t q_eval's action
        # 得到这一批状态下每个动作的Q值
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)  # 0-31

        # 在batch_memory,前n_features列都是状态,所以第n_features列刚好就是动作
        # 这里就可以根据索引选出动作

        eval_act_index = batch_memory[:, self.width*self.height*self.deep].astype(int)#取得动作值

        # 动作奖励
        reward = batch_memory[:, self.width*self.height*self.deep + 1]

        # 以下为q距阵(32,4),更新eval_act_index动作的Q值

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)


        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.width*self.height*self.deep].reshape(self.batch_size,self.width,self.height,self.deep),
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        self.saver.save(self.sess, "my_net/save_net.ckpt")
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
