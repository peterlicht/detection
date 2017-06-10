from __future__ import division, print_function, absolute_import
import time

import tensorflow as tf

import matplotlib.pyplot as plt


from functions import save_data, get_data, split_by_y, store_from_csv, get_all_summary_data, get_stochastic_batch

from classification_training import train_classes
from k_means import k_means

import numpy as np
import os

from sklearn.preprocessing import scale, MinMaxScaler

flags = tf.app.flags
FLAGS = flags.FLAGS

# All comment values are examples for stage 3 to give an idea.

# Init Parameters
side = 0
rng = 47
rng_str = str(rng)

# GITTEST

# Parameters
# common
flags.DEFINE_string('summaries_dir', '/home/patrick/resources/logs', 'Directory for storing logs')
flags.DEFINE_string('variables_dir', '/home/patrick/resources/variables', 'Directory for storing variable data')
# classification
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_integer('features',8, 'Amount of features per convolutional step')
# clustering
flags.DEFINE_integer('clusters',8, 'Number of clusters you want')    #doesn't do anything, no of clusters is defined by amount of different labels later

# Execution modes
# common
flags.DEFINE_integer('stage', 4, 'The resolution of data')
flags.DEFINE_string('readout', 'nope', 'Readout mode is -data, -maps, or -both. Anything else will read from summary files')
# classification
flags.DEFINE_string('train', 'nope', '-train the set, -evaluate a specific entry or -both')
flags.DEFINE_boolean('anew', False, 'Continue training variables or create a new one. Will create new, if no previous data is availabe')
# clustering
flags.DEFINE_string('clustering','nope', '-sklearn, -tensorflow')
flags.DEFINE_string('PCA','nope', '-sklearn')


stage = FLAGS.stage
stagex = 2 ** stage
stagey = 2 ** (stage + 1)
tiles = stagex*stagey
stage_str = str(stage)
counter = 0



if FLAGS.readout in ['data', 'maps', 'both']:
    store_from_csv(mode_str=FLAGS.readout,stage=stage,side=side)

coils, defect_tensor, labels, labels_matrix = get_all_summary_data(stage)

scaled_defect_tensor = MinMaxScaler().fit_transform(defect_tensor)

train_coils, test_coils, train_set, train_labels, test_set, test_labels = split_by_y(coils, defect_tensor, labels_matrix, ratio=0.75, random = rng, randomize = True)
train_coils_n, test_coils_n, train_set_n, train_labels_n, test_set_n, test_labels_n = split_by_y(coils, defect_tensor, labels, ratio=0.75, random= rng, randomize= True)

# print(batch_train_coils)
# print(batch_train_set)
# print(batch_train_labels)


#print(labels)
#print(train_set_n)
#print(train_labels_n)
#print(train_coils)
#print(len(train_set_n))
#print(len(train_labels_n))
#print(test_labels_n)

if FLAGS.train == 'train' or FLAGS.train == 'evaluate':
    train_classes(stage,train_coils,train_set,train_labels,test_coils,test_set,test_labels,train_labels_n,test_labels_n,FLAGS.features,FLAGS.learning_rate,FLAGS.max_steps,rng_str,FLAGS.dropout,
              FLAGS.summaries_dir,FLAGS.variables_dir,mode=FLAGS.train)

if FLAGS.clustering in ['tensorflow','sklearn']:
    k_means(defect_tensor, labels,stage, FLAGS.summaries_dir, FLAGS.clustering, FLAGS.PCA)


# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""


import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 30
batch_size = len(scaled_defect_tensor)
display_step = 5
examples_to_show = 10

# Network Parameters
n_hidden_1 = int(len(scaled_defect_tensor[:,0])/2) # 1st layer num features
print('n_hidden_1 of length', n_hidden_1)
n_hidden_2 = int(len(scaled_defect_tensor[:,0])/4) # 2nd layer num features
print('n_hidden_2 of length', n_hidden_2)
n_input = int(len(scaled_defect_tensor[0,:])) # MNIST data input (img shape: 28*28)

print('Autoencoder started..')
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
# encoder_op = tf.Print(encoder_op,[encoder_op], summarize = 40, message='encoder_op')
# encoder_op = tf.Print(encoder_op,[tf.shape(encoder_op)], summarize = 40, message='encoder_op_shape')
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# y_pred = tf.Print(y_pred,[y_pred], summarize = 16, message = 'y_pred')
# y_pred = tf.Print(y_pred,[tf.shape(y_pred)])
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
squares_1d = tf.scalar_summary('cost', cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    total_batch = int(len(scaled_defect_tensor[:,0]))
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = scaled_defect_tensor, labels
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: scaled_defect_tensor[0:examples_to_show,:]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(scaled_defect_tensor[i,:], (stagey, stagex)))
        a[1][i].imshow(np.reshape(encode_decode[i], (stagey, stagex)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
