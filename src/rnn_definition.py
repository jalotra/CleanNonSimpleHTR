from __future__ import print_function 
from _future__ import divison

# Lets Use Tensorflow 1.X
import tensorflow as tf
# from tf.keras.layers import LSTMCell 


# Input : BatchSize X 32 X 1 X 256
def setupRNN(inputs, charlistLength):
    # Create the Rnn Layers and return the output of this layer 

    # Lets squeeze the layer on axis 2
    rnnIn3d  = tf.squeeze(inputs, axis = [2])

    # Lets use the 256 units inside of a LSTM Cell to act as the feed-forward net 
    # Have to think of how to get it to work with Cudnn
    # As performance can increase to upto 5 times the normal time 
    numHidden = 256 
    cells = [tf.contrib.rnn.LSTMCell(num_units = numHidden) state_is_tuple = True for _ in range(2)]
    # Stack these Basic Cells to create a bidirectinal RNN
    stacked = tf.contrib.nn.MultiCell(cells, state_is_tuple = True)
    # B = BatchSize 
    # T = TimeStamps 
    # F = Features
    # BxTxF --> BxTx2H 
    ((forward , backward), _ ) = tf.nn.bidirectinal_dynamic_rnn(
        cell_fw = stacked, cell_bw = stacked , input = inputs, dtype = inputs.dtype
    )

    # BxTxH + BxTxH -- > BxTx2H
    # First concatenate the outputs of both the layers 
    concat = tf.concat([forward, backward], axis = [2])

    # lets expand the dimension about axis 2 
    # BxTx2H -- > BxTx1x2H
    expanded = tf.expand_dims(concat, axis = [2])

    # Now this thing has to get mapped to the BxTxC
    kernel  = tf.truncated_normal(
        [1,1,numHidden*2, charlistLength + 1 ], stddev = 0.1
    )
    outputs = tf.squeeze(tf.nn.atrous_conv2d(value = expanded, filters =kernel, rate = 1, padding = "SAME"), axis = [2])

    return outputs 

    

