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
                                     shape=(input_shape[0][-2],),
                                     initializer='zero')
        super(Attention, self).build(input_shape)

    def split_last_dimension(self, x, n):
        old_shape = x.get_shape().dims
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        return tf.transpose(ret, [0, 2, 1, 3])

    def mask_logits(self, inputs, mask, mask_value=-1e12):
        shapes = [x if x != None else -1 for x in inputs.shape.as_list()]
        mask = K.cast(mask, tf.int32)
        mask = K.one_hot(mask[:, 0], shapes[-1])
        mask = 1 - K.cumsum(mask, 1)
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, [shapes[0], 1, 1, shapes[-1]])
        return inputs + mask_value * (1 - mask)

    def dot_product_attention(self, x, seq_len=None, dropout=0.1):
        q, k, v = x
        logits = tf.matmul(q, k, transpose_b=True)
        if self.bias:
            logits += self.b
        if seq_len is not None:
            logits = self.mask_logits(logits, seq_len)
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = tf.nn.dropout(weights, 1.0 - dropout)
        x = tf.matmul(weights, v)
        return x

    def combine_last_two_dimensions(self, x):
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        ret.set_shape(new_shape)
        return ret

    def call(self, x, mask=None):
        memory, query, seq_len = x
        Q = self.split_last_dimension(query, self.num_heads)
        memory = tf.split(memory, 2, axis=2)
        K = self.split_last_dimension(memory[0], self.num_heads)
        V = self.split_last_dimension(memory[1], self.num_heads)

        key_depth_per_head = self.units // self.num_heads
        Q *= (key_depth_per_head ** -0.5)
        x = self.dot_product_attention([Q, K, V], seq_len, dropout=self.dropout)
        x = self.combine_last_two_dimensions(tf.transpose(x, [0,2,1,3]))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[1]