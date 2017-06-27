#!/usr/bin/env python
#-*- coding:utf-8 -*-
# what's the meaning of this function?
from __future__ import print_function

import cv2
import sys
sys.path.append("game/")
import dummy_game
import tetris_fun as game
import numpy as np
import tensorflow as tf
import random
from collections import deque

GAME = "Tetris" # name of the game being played
ACTIONS = 5 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to onserve before training
EXPLORE = 2000000. # frames over which to anneal epsilon
final_epsilon = 0.0001 # final value of epsilon
initial_epsilon = 0.1 # starting value of epsilon
replay_memory = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
frame_per_action = 1

'''
#执行动作，与模拟器交互获得奖励和下一帧图像以及游戏是否终止
img_state, reward, terminal = game_state.frame_step(action)

#游戏分数
game_state.score
'''

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = 
        [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # input layer
    s = tf.placeholder(tf.float32, [None, ..., ..., ...])

    # 1st convolutional layer and apply the Relu function
    W_conv1 = weight_variable([..., ..., ..., ...])
    b_conv1 = bias_variable([...])
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, ...) + b_conv1)
    # apply finally max pool
    h_pool1 = max_pool_2x2(h_conv1)


    # 2nd convolutional layer and apply the Relu function
    W_conv2 = weight_variable([..., ..., ..., ...])
    b_conv2 = bias_variable([...])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, ...) + b_conv2)
    # apply finally max pool
    # h_pool2 = max_pool_2x2(h_conv2)

    # 3rd convolutional layer and apply the Relu function
    W_conv3 = weight_variable([..., ..., ..., ...])
    b_conv3 = bias_variable([...])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, ...) + b_conv3)
    # apply finally max pool
    # h_pool3 = max_pool_2x2(h_conv3)

    # 1st fully-connected layer and apply the Relu function
    W_fc1 = weight_variable([..., ...])
    b_fc1 = bias_variable([...])

    h_conv3_flat = tf.reshape(h_conv3, [..., ...])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # 2nd fully-connected layer
    W_fc2 = weight_variable([..., ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    
    # output layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function and train with Adam optimizer
    a = tf.placeholder(tf.float32, [None, ACTIONS])
    y = tf.placeholder(tf.float32, [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()
    # store the previous obervations in replay memory
    D = deque()

    # get the 1st state by doing nothing 
    # preprocess the image to ideal size
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (..., ...)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights!")

    # start training
    epsilon = initial_epsilon
    t = 0
    while True:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict = {s:[s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % frame_per_action == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > final_epsilon and t > OBSERVE:
            epsilon -= (initial_epsilon - final_epsilon) / EXPLORE

            # run the selected action and oberve nest state & reward
        x_t1_colored, r_t, termianl = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (..., ...)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (..., ..., ...))
        #s_t1 = np.append(x_t1, s_t[:, :, :3], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis = 2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > replay_memory:
            D.popleft()

        # only train is done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minnibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict= {
                y : y_batch
                a : a_batch
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        #save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/'+ GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))

def playGame():
    sess = tf.InteravtiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ = "__main__":
    main()
