# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tmp import *

class clstm_clf(object):
    """
    A C-LSTM classifier for text classification
    Reference: A C-LSTM Neural Network for Text Classification
    """
    def __init__(self, max_length, num_classes, filter_sizes, num_filters, num_layer, l2_reg_lambda=0.0):
       
        # Placeholders
        # self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, max_length, None], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='input_y')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='sequence_length')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        # L2 loss
        self.l2_loss = tf.constant(0.0)
        self.hidden_size = num_filters

        liist = []
        
        for i in range(1):
           liist.append(self.sequence_length[i][0])


            
        # Word embedding
        inputs = tf.expand_dims(self.input_x, -1)

        # Input dropout
        # inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        conv_outputs = []
        max_feature_length = max_length - max(filter_sizes) + 1
        # Convolutional layer with different lengths of filters in parallel
        # No max-pooling
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('conv-%s' % filter_size):
                # [filter size, embedding size, channels, number of filters]
                filter_shape = [filter_size, 7, 1, num_filters]
                W = tf.get_variable('weights', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(0.0))

                # Convolution
                conv = tf.nn.conv2d(inputs,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                # Activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                #conv_outputs.append(pooled)

                # Remove channel dimension
                h_reshape = tf.squeeze(h, [2])
                # Cut the feature sequence at the end based on the maximum filter length
                h_reshape = h_reshape[:, :max_feature_length, :]

                conv_outputs.append(h_reshape)

    # Concatenate the outputs from different filters
        # if len(filter_sizes) > 1:
        #     rnn_inputs = tf.concat(conv_outputs, -1)
        # # else:
        rnn_inputs = h_reshape

        # LSTM cell
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       forget_bias=1.0,
                                       state_is_tuple=True,
                                       reuse=tf.get_variable_scope().reuse)
        # Add dropout to LSTM cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layer, state_is_tuple=True)

        self._initial_state = cell.zero_state(1, dtype=tf.float32)

        # Feed the CNN outputs to LSTM network
        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               rnn_inputs,
                                               initial_state=self._initial_state,
                                               sequence_length=liist)
            self.final_state = state

        # Softmax output layer
        with tf.name_scope('softmax'):
            W = tf.get_variable('W', shape=[self.hidden_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

            self.logits = tf.nn.xw_plus_b(self.final_state[num_layer - 1].h, W, b, name='scores')
            self.hypothesis = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.logits, 1, name='predictions')


        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
           correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
           self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

            