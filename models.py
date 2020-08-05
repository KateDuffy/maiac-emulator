import tensorflow as tf
import bdl
import numpy as np
# import tf_utils
import os, sys
from tensorflow import keras
from tensorflow.python.ops import math_ops




@tf.function
def loss(y, m, loc, logvar, probs, is_training, reg_losses, step, eps=1e-3):
    # if training then crop center of y, else, padding was applied
    slice_amt = 0  # (np.sum(self.filter_sizes) - len(self.filter_sizes)) / 2
    if slice_amt > 0:
        slice_y = y[:, slice_amt:-slice_amt, slice_amt:-slice_amt]
        slice_m = m[:, slice_amt:-slice_amt, slice_amt:-slice_amt]
    else:
        slice_y = y
        slice_m = m

    # start by slicing if training
    _y = tf.cond(is_training, lambda: slice_y, lambda: y)
    _m = tf.cond(is_training, lambda: slice_m, lambda: m)

    # get classification labels
    classes = tf.ones_like(_y) * _m               

    # select pixels with no mask
    indicies = tf.where(tf.equal(classes, 1))    
    _y_cond = tf.gather_nd(_y, indicies)
    _loc_cond = tf.gather_nd(loc, indicies)       
    _logvar_cond = tf.gather_nd(logvar, indicies) 
    _precision_cond = tf.exp(-_logvar_cond) + eps 

    with tf.device('/cpu:0'):
        step =  tf.cast(step, dtype=tf.int64)
        tf.compat.v2.summary.histogram('precision_cond', _precision_cond, step=step)
        tf.compat.v2.summary.histogram('logvar_cond', _logvar_cond, step=step)


    cond_logprob = _precision_cond * (_y_cond - _loc_cond) ** 2 + _logvar_cond
    cond_logprob *= -1

    # if no pixels contain precip, then cond_logprob is empty and reduce_mean fails
    cond_logprob = tf.cond(tf.equal(tf.size(cond_logprob), 0),
                           lambda: tf.constant(0.0), lambda: cond_logprob)

    # classification loss
    logprob_classifier = classes * tf.math.log(probs + eps)
    logprob_classifier += (1 - classes) * tf.math.log(1 - probs + eps)

    # total loss
    logprob = tf.reduce_mean(cond_logprob) + tf.reduce_mean(logprob_classifier)

    neglogloss = -tf.reduce_mean(logprob, name='neglogloss')

    with tf.device("/cpu:0"):
        tf.compat.v2.summary.scalar('loss/class_mean', tf.reduce_mean(classes), step=step)
        tf.compat.v2.summary.scalar('loss/logit_loss', -tf.reduce_mean(logprob_classifier), step=step)  # 
        tf.compat.v2.summary.scalar('loss/cond_logprob', -tf.reduce_mean(cond_logprob), step=step)
        tf.compat.v2.summary.scalar('loss/neglogloss', neglogloss, step=step)
        tf.compat.v2.summary.scalar('loss/regularizer', reg_losses, step=step)
        tf.compat.v2.summary.scalar('loss/logprobs', -tf.reduce_mean(tf.math.log(probs + eps)), step=step)  #
        tf.compat.v2.summary.scalar('loss/probs', tf.reduce_mean(probs), step=step)  #
        tf.compat.v2.summary.scalar('loss/logvar_cond', -tf.reduce_mean(_logvar_cond), step=step)
    return neglogloss + reg_losses



class DCBase(keras.Model):
    def __init__(self, output_bands, tau=1e-5, priorlengthscale=1e1, N=int(1e9)):
        super(DCBase, self).__init__()
        self.wd = priorlengthscale / (tau * N)
        self.dd = 2. / (tau * N)
        self.output_bands = output_bands
        

class DCCNN(DCBase):
    def __init__(self, filter_sizes, layer_sizes, *args,
                 **kwargs):
        super(DCCNN, self).__init__(*args, **kwargs)
        self.filter_sizes = filter_sizes
        self.layer_sizes = layer_sizes
        self.concrete_layers = []
        self.conv2d_layers = []
        for i, k in enumerate(self.filter_sizes):
            if i != 0:
                concrete_layer = bdl.ConcreteLayer(dropout_decay=self.dd)
                self.concrete_layers.append(concrete_layer)
            if i == len(filter_sizes) - 1:
                activation = None
            else:
                activation = tf.nn.relu 
            self.conv2d_layers.append(keras.layers.Conv2D(self.layer_sizes[i], k,
                                                          activation=activation,
                                                          padding='same'))

    @tf.function(
        input_signature=[tf.TensorSpec([None, None, None, 16], tf.float32)])
    def inference(self, inputs):
        for i in range(len(self.filter_sizes)):
            if i != 0:
                input_shape = tf.concat([tf.shape(inputs)[:-1],
                                         [self.layer_sizes[0]]],
                                        axis=0)
                inputs, reg_loss, dropout_prob = self.concrete_layers[i - 1](
                    inputs, training=False, input_shape=input_shape)
            inputs = self.conv2d_layers[i](inputs)
        logits = tf.expand_dims(inputs[:, :, :, 0], -1)
        probs = tf.sigmoid(logits)
        loc = inputs[:, :, :, 1:self.output_bands + 1]
        logvar = inputs[:, :, :, self.output_bands + 1:]
        return loc, logvar, probs

    #@tf.function
    def __call__(self, inputs, training=False):
        dropout_probs = []
        reg_losses = 0.
        for i in range(len(self.filter_sizes)):
            if i != 0:
                input_shape = tf.concat([tf.shape(inputs)[:-1],
                                         [self.layer_sizes[0]]],
                                        axis=0)
                inputs, reg_loss, dropout_prob = self.concrete_layers[i - 1](
                    inputs,training=training, input_shape=input_shape)
                reg_losses += reg_loss
                dropout_probs.append(dropout_prob)
                p = dropout_prob
            else:
                p = 0
            inputs = self.conv2d_layers[i](inputs)
        logits = tf.expand_dims(inputs[:, :, :, 0], -1)
        probs = tf.sigmoid(logits)
        loc = inputs[:, :, :, 1:self.output_bands + 1]
        logvar = inputs[:, :, :, self.output_bands + 1:]

        mask_discrete = tf.cast(tf.greater(probs, 0.8), tf.float32)
        prediction = mask_discrete * loc
        return loc, logvar, probs, prediction, reg_losses, dropout_probs



class ResidualBlock(keras.layers.Layer):
    def __init__(self, hidden_units, weight_decay=1e-4, dropout_decay=1e-4,
                 bayesian=True):
        super(ResidualBlock, self).__init__()
        self.bayesian = bayesian
        self.weight_decay = weight_decay

        if bayesian:
            self.concrete1 = bdl.ConcreteLayer(dropout_decay=dropout_decay)
        w_regularizer = keras.regularizers.l2(weight_decay)
        self.conv1 = keras.layers.Conv2D(hidden_units, 3, activation=tf.nn.leaky_relu,
                                         kernel_regularizer=w_regularizer,
                                         bias_regularizer=w_regularizer,
                                         padding="same")

        if bayesian:
            self.concrete2 = bdl.ConcreteLayer(dropout_decay=dropout_decay)
        w_regularizer = keras.regularizers.l2(weight_decay)
        self.conv2 = keras.layers.Conv2D(hidden_units, 3, activation=None,
                                         kernel_regularizer=w_regularizer,
                                         bias_regularizer=w_regularizer,
                                         padding="same")

        if bayesian:
            self.concrete3 = bdl.ConcreteLayer(dropout_decay=dropout_decay)
        w_regularizer = keras.regularizers.l2(weight_decay)
        self.conv3 = keras.layers.Conv2D(hidden_units, 3, activation=tf.nn.leaky_relu,
                                         kernel_regularizer=w_regularizer,
                                         bias_regularizer=w_regularizer,
                                         padding="same")

    def call(self, inputs, training=False):
        reg_losses = 0
        p = 0.
        original_inputs = inputs

        if self.bayesian:
            inputs, reg_loss, dropout_prob = self.concrete1(inputs, tf.shape(inputs), training)
            reg_losses += reg_loss
            p = dropout_prob
        if training:
            self.conv1.kernel_regularizer.l2 = self.weight_decay / (1 - p)
        inputs = self.conv1(inputs)

        if self.bayesian:
            inputs, reg_loss, dropout_prob = self.concrete2(inputs, tf.shape(inputs), training)
            reg_losses += reg_loss
            p = dropout_prob
        if training:
            self.conv2.kernel_regularizer.l2 = self.weight_decay / (1 - p)
        inputs = self.conv2(inputs)
        inputs += original_inputs

        if self.bayesian:
            inputs, reg_loss, dropout_prob = self.concrete3(inputs, tf.shape(inputs), training)
            reg_losses += reg_loss
            p = dropout_prob
        if training:
            self.conv3.kernel_regularizer.l2 = self.weight_decay / (1 - p)
        inputs = self.conv3(inputs)

        return inputs, reg_losses


class DCResNet(DCBase):
    def __init__(self, blocks=16, hidden_units=128, kernel_size=3, *args, **kwargs):
        super(DCResNet, self).__init__(*args, **kwargs)
        self.blocks = blocks
        self.hidden_units = hidden_units

        w_regularizer = keras.regularizers.l2(self.wd)
        self.conv1 = keras.layers.Conv2D(hidden_units, kernel_size,
                                         activation=tf.nn.relu,
                                         kernel_regularizer=w_regularizer,
                                         bias_regularizer=w_regularizer,
                                         padding="same")

        self.concrete1 = bdl.ConcreteLayer(dropout_decay=self.dd)

        self.residual_blocks = []
        for _ in range(self.blocks):
            self.residual_blocks.append(ResidualBlock(hidden_units,
                                                      weight_decay=self.wd,
                                                      dropout_decay=self.dd))
        self.concrete2 = bdl.ConcreteLayer(dropout_decay=self.dd)

        w_regularizer = keras.regularizers.l2(self.wd)
        self.conv2 = keras.layers.Conv2D(hidden_units, kernel_size,
                                         activation=tf.nn.relu,
                                         kernel_regularizer=w_regularizer,
                                         bias_regularizer=w_regularizer,
                                         padding="same")

        self.concrete3 = bdl.ConcreteLayer(dropout_decay=self.dd)
        w_regularizer = keras.regularizers.l2(self.wd)
        self.conv3 = keras.layers.Conv2D(self.output_bands * 2 + 1,
                                         kernel_size, activation=None,
                                         kernel_regularizer=w_regularizer,
                                         bias_regularizer=w_regularizer,
                                         padding="same")

    @tf.function(
        input_signature=[tf.TensorSpec([None, None, None, 16], tf.float32)],)
    def inference(self, inputs):
        inputs = self.conv1(inputs)
        middle_value = inputs
        inputs, reg_loss, dropout_prob = self.concrete1(inputs,
                                                        tf.shape(inputs),
                                                        False)
        for block in self.residual_blocks:
            inputs, reg_loss = block(inputs, training=False)
        inputs += middle_value

        inputs, reg_loss, dropout_prob = self.concrete2(inputs,
                                                        tf.shape(inputs),
                                                        False)
        inputs = self.conv2(inputs)

        inputs, reg_loss, dropout_prob = self.concrete3(inputs,
                                                        tf.shape(inputs),
                                                        False)
        inputs = self.conv3(inputs)
        
        logits = tf.expand_dims(inputs[:, :, :, 0], -1)
        probs = tf.sigmoid(logits)
        loc = tf.identity(inputs[:, :, :, 1:self.output_bands + 1], 'loc')
        logvar = tf.identity(inputs[:, :, :, self.output_bands + 1:], 'logvar')
        return loc, logvar, probs

    def call(self, inputs, training=False):
        dropout_probs = []
        reg_losses = 0
        inputs = self.conv1(inputs)
        middle_value = inputs
        inputs, reg_loss, dropout_prob = self.concrete1(inputs,
                                                        tf.shape(inputs),
                                                        training)
        reg_losses += reg_loss
        dropout_probs.append(dropout_prob)

        for block in self.residual_blocks:
            inputs, reg_loss = block(inputs, training=training)
            reg_losses += reg_loss
        inputs += middle_value

        inputs, reg_loss, dropout_prob = self.concrete2(inputs,
                                                        tf.shape(inputs),
                                                        training)
        reg_losses += reg_loss
        dropout_probs.append(dropout_prob)
        #self.conv2.kernel_regularizer.l2 = self.wd / (1-dropout_prob)
        inputs = self.conv2(inputs)

        inputs, reg_loss, dropout_prob = self.concrete3(inputs,
                                                        tf.shape(inputs),
                                                        training)
        reg_losses += reg_loss
        dropout_probs.append(dropout_prob)
        #self.conv3.kernel_regularizer.l2 = self.wd / (1 - dropout_prob)
        inputs = self.conv3(inputs)

        logits = tf.expand_dims(inputs[:, :, :, 0], -1, name='logits')
        probs = tf.sigmoid(logits, name='mask_probs')

        loc = inputs[:, :, :, 1:self.output_bands + 1]
        logvar = inputs[:, :, :, self.output_bands + 1:]

        mask_discrete = tf.cast(tf.greater(probs, 0.8), tf.float32)
        prediction = mask_discrete * loc

        return loc, logvar, probs, prediction, reg_losses, dropout_probs


class DCVDSR_New(DCBase):
    def __init__(self, output_bands, hidden_layers=(512,) * 3, *args,
                 **kwargs):
        super(DCVDSR_New, self).__init__(output_bands, *args, **kwargs)
        self.hidden_layers = hidden_layers
        # w_regularizer = keras.regularizers.l2(self.wd)
        self.conv1 = keras.layers.Conv2D(self.hidden_layers[0], 4,
                                         activation=tf.nn.relu,
                                         # kernel_regularizer=w_regularizer,
                                         # bias_regularizer=w_regularizer,
                                         padding="same")
        self.concrete_layers = []
        self.conv_layers = []
        for i, hunits in enumerate(self.hidden_layers):
            self.concrete_layers.append(bdl.ConcreteLayer(self.dd))
            # w_regularizer = keras.regularizers.l2(self.wd)
            self.conv_layers.append([
                keras.layers.Conv2D(hunits, 3, activation=tf.nn.relu,
                                    padding="same"),
                keras.layers.Conv2D(hunits, 5, activation=tf.nn.relu,
                                    padding="same"),
                keras.layers.Conv2D(hunits, 7, activation=tf.nn.relu,
                                    padding="same")
            ])

        self.concrete_output = bdl.ConcreteLayer(self.dd)
        # w_regularizer = keras.regularizers.l2(self.wd)
        self.conv_output = keras.layers.Conv2D(self.output_bands * 2 + 1, 3,
                                               activation=None,
                                               # kernel_regularizer=w_regularizer,
                                               # bias_regularizer=w_regularizer,
                                               padding="same")

    @tf.function(
        input_signature=[tf.TensorSpec([None, None, None, 16], tf.float32)])
    def inference(self, inputs):
        inputs = self.conv1(inputs)
        middle_value = inputs
        for i in range(len(self.hidden_layers)):
            inputs, reg_loss, dropout_prob = self.concrete_layers[i](inputs,
                                                                     tf.shape(inputs),
                                                                     False)
            output_3 = self.conv_layers[i][0](inputs)
            output_5 = self.conv_layers[i][1](inputs)
            output_7 = self.conv_layers[i][2](inputs)
            inputs = tf.concat([output_3, output_5, output_7], -1)

        inputs += middle_value

        inputs, reg_loss, dropout_prob = self.concrete_output(inputs,
                                                              tf.shape(inputs),
                                                              False)
        inputs = self.conv_output(inputs)
        
        logits = tf.expand_dims(inputs[:, :, :, 0], -1)
        probs = tf.sigmoid(logits)
        loc = inputs[:, :, :, 1:self.output_bands + 1]
        logvar = inputs[:, :, :, self.output_bands + 1:]
        return loc, logvar, probs

    @tf.function
    def call(self, inputs, training=False):
        reg_losses = 0
        dropout_probs = []
        inputs = self.conv1(inputs)
        middle_value = inputs
        for i in range(len(self.hidden_layers)):
            inputs, reg_loss, dropout_prob = self.concrete_layers[i](inputs,
                                                                     tf.shape(inputs),
                                                                     training)
            reg_losses += reg_loss
            dropout_probs.append(dropout_prob)
            # self.conv_layers[i].kernel_regularizer.l2 = float(self.wd / (1-dropout_prob))
            inputs = self.conv_layers[i](inputs)

        inputs += middle_value

        inputs, reg_loss, dropout_prob = self.concrete_output(inputs,
                                                              tf.shape(inputs),
                                                              training)
        reg_losses += reg_loss
        # self.conv_output.kernel_regularizer.l2 = float(self.wd / (1-dropout_prob))
        inputs = self.conv_output(inputs)

        logits = tf.expand_dims(inputs[:, :, :, 0], -1, name='logits')
        probs = tf.sigmoid(logits, name='mask_probs')

        loc = inputs[:, :, :, 1:self.output_bands + 1]
        logvar = inputs[:, :, :, self.output_bands + 1:]
        mask_discrete = tf.cast(tf.greater(probs, 0.8), tf.float32)
        prediction = mask_discrete * loc
        return loc, logvar, probs, prediction, reg_losses, dropout_probs



class DCVDSR(DCBase):
    def __init__(self, output_bands, hidden_layers=(512,)*3, *args, **kwargs):
        super(DCVDSR, self).__init__(output_bands, *args, **kwargs)
        self.hidden_layers = hidden_layers
        #w_regularizer = keras.regularizers.l2(self.wd)
        self.conv1 = keras.layers.Conv2D(self.hidden_layers[0], 3,
                                         activation=tf.nn.relu,
                                         #kernel_regularizer=w_regularizer,
                                         #bias_regularizer=w_regularizer,
                                         padding="same")
        self.concrete_layers = []
        self.conv_layers = []
        for i, hunits in enumerate(self.hidden_layers):
            self.concrete_layers.append(bdl.ConcreteLayer(self.dd))
            #w_regularizer = keras.regularizers.l2(self.wd)
            self.conv_layers.append(keras.layers.Conv2D(hunits, 3,
                                                        activation=tf.nn.relu,
                                                        #kernel_regularizer=w_regularizer,
                                                        #bias_regularizer=w_regularizer,
                                                        padding="same"))
        self.concrete_output = bdl.ConcreteLayer(self.dd)
        #w_regularizer = keras.regularizers.l2(self.wd)
        self.conv_output = keras.layers.Conv2D(self.output_bands * 2 + 1, 3,
                                               activation=None,
                                               #kernel_regularizer=w_regularizer,
                                               #bias_regularizer=w_regularizer,
                                               padding="same")

    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 16], tf.float32)])
    def inference(self, inputs):
        inputs = self.conv1(inputs)
        middle_value = inputs
        for i in range(len(self.hidden_layers)):
            inputs, reg_loss, dropout_prob = self.concrete_layers[i](inputs,
                                                                     tf.shape(inputs),
                                                                     False)
            inputs = self.conv_layers[i](inputs)

        inputs += middle_value

        inputs, reg_loss, dropout_prob = self.concrete_output(inputs,
                                                              tf.shape(inputs),
                                                              False)
        inputs = self.conv_output(inputs)
        
        logits = tf.expand_dims(inputs[:, :, :, 0], -1)
        probs = tf.sigmoid(logits)
        loc = inputs[:, :, :, 1:self.output_bands + 1]
        logvar = inputs[:, :, :, self.output_bands + 1:]
        return loc, logvar, probs


    @tf.function
    def call(self, inputs, training=False):
        reg_losses = 0
        dropout_probs = []
        inputs = self.conv1(inputs)
        middle_value = inputs
        for i in range(len(self.hidden_layers)):
            inputs, reg_loss, dropout_prob = self.concrete_layers[i](inputs,
                                                                     tf.shape(inputs),
                                                                     training)
            reg_losses += reg_loss
            dropout_probs.append(dropout_prob)
            #self.conv_layers[i].kernel_regularizer.l2 = float(self.wd / (1-dropout_prob))
            inputs = self.conv_layers[i](inputs)

        inputs += middle_value

        inputs, reg_loss, dropout_prob = self.concrete_output(inputs,
                                                              tf.shape(inputs),
                                                              training)
        reg_losses += reg_loss
        #self.conv_output.kernel_regularizer.l2 = float(self.wd / (1-dropout_prob))
        inputs = self.conv_output(inputs)

        logits = tf.expand_dims(inputs[:, :, :, 0], -1, name='logits')
        probs = tf.sigmoid(logits, name='mask_probs')

        loc = inputs[:, :, :, 1:self.output_bands + 1]
        logvar = inputs[:, :, :, self.output_bands + 1:]
        mask_discrete = tf.cast(tf.greater(probs, 0.8), tf.float32)
        prediction = mask_discrete * loc
        return loc, logvar, probs, prediction, reg_losses, dropout_probs

