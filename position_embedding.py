from keras.engine.topology import Layer
import tensorflow as tf
import math


class Position_Embedding(Layer):
    def __init__(self, min_timescale=1.0, max_timescale=1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        super(Position_Embedding, self).__init__(**kwargs)

    def get_timing_signal_1d(self, length, channels):
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2
        log_timescale_increment = (math.log(float(self.max_timescale) / float(self.min_timescale)) / (tf.to_float(num_timescales) - 1))
        inv_timescales = self.min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        return signal

    def add_timing_signal_1d(self, x):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        signal = self.get_timing_signal_1d(length, channels)
        return x + signal

    def call(self, x, mask=None):
        return self.add_timing_signal_1d(x)

    def compute_output_shape(self, input_shape):
        return input_shape