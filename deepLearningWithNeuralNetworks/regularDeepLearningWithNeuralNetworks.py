# -*- coding: utf-8 -*-
"""Deep Learning with Neural Networks and TensorFlow.

Deep learning is part of a broader family of machine learning methods based on
learning data representations, as opposed to task-specific algorithms. Learning
can be supervised, partially supervised or unsupervised.

A deep neural network (DNN) is an artificial neural network with multiple
hidden layers between the input and output layers.

We will use the library Tensorflow to do number crunching. A package like
TensorFlow allows us to perform specific machine learning number-crunching
operations tensors with large efficiency. We can also easily distribute this
processing across our CPU cores, GPU cores, or even multiple devices like
multiple GPUs. But that's not all! We can even distribute computations across
a distributed network of computers with TensorFlow.

The data used is from the MNIST database of handwritten digits:
http://yann.lecun.com/exdb/mnist/ but is imported by tensorflow library

Example:

        $ python regularDeepLearningWithNeuralNetworks.py

Todo:
    *
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# following two lines is to avoid warnings.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Computation graph and modeling the deep neural network
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100
X = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    """Function to create neural network model.

    The model we have for each layer has the form:
    (input_data * weights) + biases
    """
    # create dynamic layer weights and biases
    hl1_weights = tf.Variable(tf.random_normal([784, n_nodes_hl1]))
    hl1_biases = tf.Variable(tf.random_normal(n_nodes_hl1))
    hidden_1_layer = {'wgt': hl1_weights, 'bias': hl1_biases}

    hl2_weights = tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]))
    hl2_biases = tf.Variable(tf.random_normal(n_nodes_hl2))
    hidden_2_layer = {'wgt': hl2_weights, 'bias': hl2_biases}

    hl3_weights = tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]))
    hl3_biases = tf.Variable(tf.random_normal(n_nodes_hl3))
    hidden_3_layer = {'wgt': hl3_weights, 'bias': hl3_biases}

    out_weights = tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]))
    out_biases = tf.Variable(tf.random_normal(n_classes))
    output_layer = {'wgt': out_weights, 'bias': out_biases}

    # create model {(input_data * weights) + biases} for each layer
    l1 = tf.add(tf.matmul(data, hidden_1_layer['wgt']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['wgt']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['wgt']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['wgt']), output_layer['bias'])

    return output
