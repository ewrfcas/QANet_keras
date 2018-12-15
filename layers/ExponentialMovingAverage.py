import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import slot_creator
import keras.backend as K
from keras.backend import moving_averages
from tqdm import tqdm


class ExponentialMovingAverage():
    def __init__(self, model, decay, weights_list=None, temp_model='temp_model.h5',
                 name='ExponentialMovingAverage', type='cpu'):
        # EMA for keras, the example can be seen in https://github.com/ewrfcas/QANet_keras/blob/master/train_QANet.py
        # init before training, but after the model init.
        self.model = model
        self.scope_name = name
        self.temp_model = temp_model
        self.type = type
        self.decay = decay
        self._averages = {}

        if weights_list is None:
            weights_list = self.model.trainable_weights

        if self.type == 'gpu':
            self.sess = K.get_session()
            for weight in weights_list:
                if weight.dtype.base_dtype not in [tf.float16, tf.float32,
                                                   tf.float64]:
                    raise TypeError("The variables must be half, float, or double: %s" %
                                    weight.name)
                if weight in self._averages:
                    raise ValueError("Moving average already computed for: %s" % weight.name)

                # For variables: to lower communication bandwidth across devices we keep
                # the moving averages on the same device as the variables. For other
                # tensors, we rely on the existing device allocation mechanism.
                with ops.init_scope():
                    if isinstance(weight, tf.Variable):
                        avg = slot_creator.create_slot(weight,
                                                       weight.initialized_value(),
                                                       self.scope_name,
                                                       colocate_with_primary=True)
                        # NOTE(mrry): We only add `tf.Variable` objects to the
                        # `MOVING_AVERAGE_VARIABLES` collection.
                        ops.add_to_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, weight)
                    else:
                        avg = slot_creator.create_zeros_slot(weight,
                                                             self.scope_name,
                                                             colocate_with_primary=(weight.op.type in ["Variable",
                                                                                                       "VariableV2",
                                                                                                       "VarHandleOp"]))
                self._averages[weight] = avg

            with tf.name_scope(self.scope_name):
                decay = ops.convert_to_tensor(decay, name="decay")
                self.updates = []
                for var in weights_list:
                    self.updates.append(
                        moving_averages.assign_moving_average(self._averages[var], var, decay, zero_debias=False))

                self.assigns = []
                for weight in weights_list:
                    self.assigns.append(tf.assign(weight, self._averages[weight]))

            self.sess.run(tf.global_variables_initializer())

        elif self.type == 'cpu':
            print('CPU EMA getting weights...')
            for weight in tqdm(weights_list):
                self._averages[weight.name] = K.get_value(weight)

    def average_update(self):
        # run in the end of each batch
        if self.type == 'gpu':
            self.sess.run(self.updates)
        elif self.type == 'cpu':
            for weight in self.model.trainable_weights:
                old_val = self._averages[weight.name]
                self._averages[weight.name] = self.decay * old_val + (1.0 - self.decay) * K.get_value(weight)

    def assign_shadow_weights(self, backup=True):
        # run while you need to assign shadow weights (at end of each epoch or the total training)
        if backup:
            self.model.save_weights(self.temp_model)

        if self.type == 'gpu':
            self.sess.run(self.assigns)
        elif self.type == 'cpu':
            print('CPU EMA assigning weights...')
            for weight in tqdm(self.model.trainable_weights):
                K.set_value(weight, self._averages[weight.name])
