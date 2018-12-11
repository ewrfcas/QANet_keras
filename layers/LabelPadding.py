# ! -*- coding: utf-8 -*-
from keras.engine.topology import Layer
import tensorflow as tf

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

class LabelPadding(Layer):
    def __init__(self, max_len, **kwargs):
        self.max_len = max_len
        super(LabelPadding, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LabelPadding, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        tensor_shape = shape_list(x) # [bs, len]
        zero_paddings = tf.zeros((tensor_shape[0], self.max_len - tensor_shape[1]))
        x = tf.concat([x, zero_paddings], axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_len)