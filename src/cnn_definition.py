# TF 2.1 MODEL BUILDING IN PROGRESS
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf

# CUSTOM CLASSES 
from decorder import Decorders as DecorderType
from DataLoader import FilePaths

def setupCNN(self, inputs):
    # Add the number of features initially which is 1
    cnnIn4d = tf.expand_dims(input = inputs, axis = 3)

    # First Layer : Conv(5x5) + Pool(2X2) :
    # Input Layer = 50 x 128x 32 x 1
    # Output size : 50 x 64 x 16 x 2
    with tf.name_scope('CONV_POOL_1'):
        kernel = tf.Variable(
            tf.truncated_normal([5, 5, 1, 2], stddev = 0.1)
        )
        conv = tf.nn.conv2d(
            cnnIn4d, kernel, padding = 'SAME', strides = [1,1,1,1]
        )
        lrelu = tf.nn.leaky_relu(conv, aplha = 0.01)
        pool = tf.nn.max_pool(lrelu, (1,2,2,1), (1,2,2,1), padding = "VALID")

    # Second Layer : Conv(5x5) + Pool(2 x 2)
    # INput Layer : 50 x 64 x 16 x 2
    # Output Size : 50 x 32 x 16 x 8
    with tf.name_scope('CONV_POOL_2'):
        kernel = tf.Variable(
            tf.truncated_normal([5,5,2,8], stddev = 0.1)
        )
        conv = tf.conv2d(
            pool, kernel, padding = 'SAME', strides = [1,1,1,1]
        )
        lrelu = tf.nn.leaky_relu(conv, alpha = 0.01)
        pool = tf.nn.max_pool(
            lrelu, (1,1,2,2), (1,1,2,2), padding = "VALID"
        )

    # THIRD LAYER 
    # INPUT SIZE = 50 X 32 X 16 X 8
    # OUTPUT SIZE = 50 X 32 X 16 X 16
    with tf.name_scope('CONV_3'):
        kernel = tf.Variable(
            tf.truncated_normal([3,3,8,16], stddev = 0.1)
        )
        conv = tf.conv2d(
            pool, kernel, padding = "SAME", strides = [1,1,1,1]
        )
        lrelu = tf.nn.leaky_relu(
            conv, alpha = 0.01
        )
        # Since no pooling is there
        pool = lrelu
    
    # FOURTH LAYER
    # INPUT LAYER  : 50 x 32 X 16 X 16
    # OUTPUT LAYER : 50 X 32 X 8 X 32
    with tf.name_scope('CONV_POOL_4'):
        kernel = tf.Variable(
            tf.truncated_normal([3, 3, 16, 32], stddev = 0.1)
        )
        conv = tf.conv2d(
            pool, kernel, padding = "SAME", strides = [1,1,1,1]
        )
        lrelu = tf.nn.leaky_relu(
            conv, alpha = 0.01
        )
        pool = tf.max_pool(
            lrelu, (1,1,2,1), (1,1,2,1), padding = "VALID"
        )
    
    # FIFTH LAYER 
    # INPUT LAYER  :50 X 32X 8 X32
    # OUTPUT LAYER : 50 X 32X 4 X 64
    with tf.name_scope('CON_POOL_BN_5'):
        kernel = tf.Variable(
            tf.truncated_normal([3,3,32,64], stddev = 0.1)
        )
        conv = tf.conv2d(
            pool, kernel, padding = "SAME", strides = [1,1,1,1]
        )
        # BATCH NORMALISATION 
        mean, variance = tf.nn.moments(conv, axes= [0])
        batch_norm = tf.nn.batch_normalisation(
            conv, mean = mean, variance = variance, offset = None, scale = None, variance_epsilon = 0.001
        )
        lrelu = tf.nn.leaky_relu(
            batch_norm, alpha = 0.01
        )
        pool = tf.max_pool(
            lrelu, (1,1,2,1), (1,1,2,1), padding = "VALID"
        )

    # SIXTH LAYER 
    # INPUT LAYER : 50 X 32 X 4 X 64
    # OUTPUT_LAYER : 50 X 32 X 2 X 128
    with tf.name_scope("CONV_POOL_6"):
        kernel = tf.Variable(
            tf.truncated_normal([3,3,64,128], stddev = 0.01)
        )
        conv = tf.nn.conv2d(
            pool, kernel, strides = [1,1,1,1], padding = "SAME" 
        )
        lrelu = tf.nn.leaky_relu(
            conv, alpha = 0.01
        )
        pool = tf.max_pool(
            lrelu, (1,1,2,1), (1,1,2,1), padding = "VALID"
        )

    # SEVENTH LAYER
    # INPUT LAYER : 50 X 32 X 2 X 128
    # OUTPUT LAYER : 50 X 32 X 1 X 256
    with tf.name_scope('CONV_LAYER_6'):
        kernel = tf.truncated_normal(
            [3,3,128,256], stddev = 0.01
        )
        conv = tf.nn.conv2d(
            pool, kernel, strides = [1,1,1,1], padding = "SAME"
        )
        lrelu = tf.nn.leaky_relu(
            conv, alpha = 0.01
        )
        pool = tf.max_pool(lrelu, (1,1,2,1), (1,1,2,1), padding = "VALID")

    outputs = pool

    # RETURN THE MAX POOLED LAYER  
    return outputs
