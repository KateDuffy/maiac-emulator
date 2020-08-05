import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.ops import nn


@tf.function
def concrete_dropout(inputs, p, noise_shape, eps=1e-6, temp=0.1):
    drop_prob = tf.math.log(p + eps) - tf.math.log(1 - p + eps)
    unif_noise = tf.random.uniform(shape=noise_shape)
    drop_prob = drop_prob + tf.math.log(unif_noise + eps) - tf.math.log(
        1. - unif_noise + eps)
    drop_prob = tf.sigmoid(drop_prob / temp)
    random_tensor = 1. - drop_prob
    retain_prob = 1. - p
    inputs *= random_tensor
    inputs /= retain_prob
    return inputs


class ConcreteLayer(keras.layers.Layer):
    def __init__(self, dropout_decay, init_prob=0.25,
                 spatial=True, eps=tf.constant(1e-6)):
        super(ConcreteLayer, self).__init__()
        self.eps = eps
        self.spatial = spatial
        init = (np.log(init_prob) - np.log(1. - init_prob)).astype(np.float32)
        self.dropout_logit = self.add_weight(name="dropout_logit", shape=None,
                                             dtype=tf.float32,
                                             initializer=keras.initializers.Constant(init),
                                             trainable=True)
        self.dropout_decay = dropout_decay
        self.dropout_reg = 0.

    @tf.function()
    def call(self, inputs, input_shape, training=False):
        
        if self.spatial:    
            noise_shape = [int(input_shape[0]), 1, 1, int(input_shape[3])]
        else:
            noise_shape = input_shape

        reg_scale = tf.cast(input_shape[3], tf.float32) * self.dropout_decay

        dropout_prob = tf.sigmoid(self.dropout_logit)

        dropout_regularizer = dropout_prob * tf.math.log(
            dropout_prob + self.eps)
        dropout_regularizer += (1. - dropout_prob) * tf.math.log(
            1. - dropout_prob + self.eps)

        dropout_reg = float(reg_scale * dropout_regularizer)

        if training:
            output = concrete_dropout(inputs, dropout_prob, noise_shape,
                                      eps=self.eps)
        else:
            output = nn.dropout(inputs, noise_shape=noise_shape, seed=None,
                                rate=dropout_prob)
            
        return output, dropout_reg, dropout_prob

