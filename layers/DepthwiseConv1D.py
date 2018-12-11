from keras.engine.topology import Layer
from keras.initializers import VarianceScaling
from keras.regularizers import *
import tensorflow as tf


class DepthwiseConv1D(Layer):

    def __init__(self, kernel_size, filter, **kwargs):
        self.kernel_size = kernel_size
        self.filter = filter
        super(DepthwiseConv1D, self).__init__(**kwargs)

    def build(self, input_shape):
        init_relu = VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')
        self.depthwise_w = self.add_weight("depthwise_filter",
                                           shape=(self.kernel_size, 1, input_shape[-1], 1),
                                           initializer=init_relu,
                                           regularizer=l2(3e-7),
                                           trainable=True)
        self.pointwise_w = self.add_weight("pointwise_filter",
                                           (1, 1, input_shape[-1], self.filter),
                                           initializer=init_relu,
                                           regularizer=l2(3e-7),
                                           trainable=True)
        self.bias = self.add_weight("bias",
                                    input_shape[-1],
                                    regularizer=l2(3e-7),
                                    initializer=tf.zeros_initializer())
        super(DepthwiseConv1D, self).build(input_shape)

    def call(self, x, mask=None):
        x = K.expand_dims(x, axis=2)
        x = tf.nn.separable_conv2d(x,
                                   self.depthwise_w,
                                   self.pointwise_w,
                                   strides=(1, 1, 1, 1),
                                   padding="SAME")
        x += self.bias
        x = K.relu(x)
        outputs = K.squeeze(x, axis=2)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
