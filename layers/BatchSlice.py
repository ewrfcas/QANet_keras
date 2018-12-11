# ! -*- coding: utf-8 -*-
from keras.engine.topology import Layer
import tensorflow as tf

class BatchSlice(Layer):
    def __init__(self, dim=2, **kwargs):
        self.dim = dim
        super(BatchSlice, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BatchSlice, self).build(input_shape)

    def call(self, x, mask=None):
        x, length = x # [bs, len, dim]
        length = tf.cast(tf.reduce_max(length), tf.int32)
        st = [0] * self.dim
        ed = [-1] * self.dim
        ed[1] = length
        x = tf.slice(x, st, ed)

        return x