from keras import backend as K
from keras.engine.topology import Layer
from keras.regularizers import *
import tensorflow as tf

class context2query_attention(Layer):

    def __init__(self, output_dim, cont_limit, ques_limit, dropout, **kwargs):
        self.output_dim=output_dim
        self.cont_limit = cont_limit
        self.ques_limit = ques_limit
        self.dropout = dropout
        super(context2query_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: [(None, 400, 128), (None, 50, 128)]
        self.W0 = self.add_weight(name='W0',
                                  shape=(input_shape[0][-1], 1),
                                  initializer='glorot_uniform',
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[1][-1], 1),
                                  initializer='glorot_uniform',
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(1, 1, input_shape[0][-1]),
                                  initializer='glorot_uniform',
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.bias = self.add_weight(name='linear_bias',
                                    shape=(input_shape[1][1],),
                                    initializer='zero',
                                    regularizer=l2(3e-7),
                                    trainable=True)
        super(context2query_attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, axis=1, time_dim=1, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            seq_len=K.cast(seq_len,tf.int32)
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[time_dim])
            mask = 1 - K.cumsum(mask, 1)
            mask = K.expand_dims(mask, axis)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x, mask=None):
        x_cont, x_ques, cont_len, ques_len = x

        # get similarity matrix S
        subres0 = K.tile(K.dot(x_cont, self.W0), [1, 1, self.ques_limit])
        subres1 = K.tile(K.permute_dimensions(K.dot(x_ques, self.W1), pattern=(0, 2, 1)), [1, self.cont_limit, 1])
        subres2 = K.batch_dot(x_cont * self.W2, K.permute_dimensions(x_ques, pattern=(0, 2, 1)))
        S = subres0 + subres1 + subres2
        S += self.bias

        S_ = tf.nn.softmax(self.Mask(S, ques_len, axis=1, time_dim=2, mode='add'))
        S_T = K.permute_dimensions(tf.nn.softmax(self.Mask(S, cont_len, axis=2, time_dim=1, mode='add'), dim=1), (0, 2, 1))
        c2q = tf.matmul(S_, x_ques)
        q2c = tf.matmul(tf.matmul(S_, S_T), x_cont)
        result = K.concatenate([x_cont, c2q, x_cont * c2q, x_cont * q2c], axis=-1)

        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)