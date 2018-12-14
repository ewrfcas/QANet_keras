import tensorflow as tf
import keras.backend as K


class ExponentialMovingAverage():
    def __init__(self, model, decay, weights_list=None, temp_model='temp_model.h5',
                 name='ExponentialMovingAverage'):
        # EMA for keras, the example can be seen in https://github.com/ewrfcas/QANet_keras/blob/master/train_QANet.py
        # init before training, but after the model init.
        self.model = model
        self.scope_name = name
        self.temp_model = temp_model
        self.sess = K.get_session()
        with tf.variable_scope(self.scope_name):
            # shadow weights dict {key:model_weights, value:shadow_weights}
            self.shadow_weights = {}
            if weights_list is None:
                weights_list = self.model.trainable_weights
            for weight in weights_list:
                self.shadow_weights[weight] = tf.Variable(weight)

            # average
            self.average_list = []
            for weight in self.shadow_weights:
                self.average_list.append(tf.assign(self.shadow_weights[weight],
                                                   decay * self.shadow_weights[weight] + (1.0 - decay) * weight))

            # assign
            self.assign_list = []
            for weight in self.shadow_weights:
                self.assign_list.append(tf.assign(weight, self.shadow_weights[weight]))

        self.sess.run(tf.global_variables_initializer())

    def average_update(self):
        # run in the end of each batch
        self.sess.run(self.average_list)

    def assign_shadow_weights(self, backup=True):
        # run while you need to assign shadow weights (at end of each epoch or the total training)
        if backup:
            self.model.save_weights(self.temp_model)
        self.sess.run(self.assign_list)