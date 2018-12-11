# ! -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf


class Attention(Layer):
    def __init__(self, units, num_heads, dropout=0.0, bias=True, **kwargs):
        self.units = units
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.bias:
            self.b = self.add_weight(name='bias',
                                     shape=([1]),
                                     initializer='zero')
        super(Attention, self).build(input_shape)

    def split_last_dimension(self, x, n):
        old_shape = x.get_shape().dims
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        return tf.transpose(ret, [0, 2, 1, 3])

    def mask_logits(self, inputs, mask, mask_value=-1e30):
        mask = tf.cast(mask, tf.float32)
        return inputs + mask_value * (1 - mask)

    def dot_product_attention(self, x, mask=None, dropout=0.1, training=None):
        q, k, v = x
        logits = tf.matmul(q, k, transpose_b=True)  # [bs, 8, len, len]
        if self.bias:
            logits += self.b
        if mask is not None:  # [bs, len]
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.expand_dims(mask, axis=1)  # [bs,1,1,len]
            logits = self.mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = K.in_train_phase(K.dropout(weights, dropout), weights, training=training)
        x = tf.matmul(weights, v)
        return x

    def combine_last_two_dimensions(self, x):
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        ret.set_shape(new_shape)
        return ret

    def call(self, x, mask=None, training=None):
        memory, query, seq_mask = x
        Q = self.split_last_dimension(query, self.num_heads)
        memory = tf.split(memory, 2, axis=2)
        K = self.split_last_dimension(memory[0], self.num_heads)
        V = self.split_last_dimension(memory[1], self.num_heads)

        key_depth_per_head = self.units // self.num_heads
        Q *= (key_depth_per_head ** -0.5)
        x = self.dot_product_attention([Q, K, V], seq_mask, dropout=self.dropout, training=training)
        x = self.combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[1]
