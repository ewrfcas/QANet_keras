import numpy as np
import QANet_keras as QANet
import os
from keras.optimizers import *
import util
from keras.models import *
import json
import pandas as pd
from tqdm import tqdm
from keras.utils import multi_gpu_model
from ExponentialMovingAverage import *
import tensorflow as tf
from keras.callbacks import Callback

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# load trainset
context_word = np.load('dataset/train_contw_input.npy')
question_word = np.load('dataset/train_quesw_input.npy')
context_char = np.load('dataset/train_contc_input.npy')
question_char = np.load('dataset/train_quesc_input.npy')
start_label = np.load('dataset/train_y_start.npy')
end_label = np.load('dataset/train_y_end.npy')
start_label_fin = np.argmax(start_label, axis=-1)
end_label_fin = np.argmax(end_label, axis=-1)

# load valset
val_context_word = np.load('dataset/test_contw_input.npy')
val_question_word = np.load('dataset/test_quesw_input.npy')
val_context_char = np.load('dataset/test_contc_input.npy')
val_question_char = np.load('dataset/test_quesc_input.npy')
val_start_label = np.load('dataset/test_y_start.npy')
val_end_label = np.load('dataset/test_y_end.npy')
val_start_label_fin = np.argmax(val_start_label, axis=-1)
val_end_label_fin = np.argmax(val_end_label, axis=-1)
val_qid = np.load('dataset/test_qid.npy').astype(np.int32)
with open('dataset/test_eval.json', "r") as fh:
    eval_file = json.load(fh)

# load embedding matrix
word_mat = np.load('dataset/word_emb_mat3.npy')
char_mat = np.load('dataset/char_emb_mat3.npy')

# parameters
char_dim = 64
cont_limit = 400
ques_limit = 50
char_limit = 16
char_input_size = 1427

batch_size = 64
n_epochs = 25
early_stop = 10
it = 0
continue_train = False
do_ema = True
ems = []
f1s = []

np.random.seed(10000)
model = QANet.QANet(word_dim=300, char_dim=char_dim, cont_limit=cont_limit, ques_limit=ques_limit,
                    char_limit=char_limit, word_mat=word_mat, char_mat=char_mat, char_input_size=char_input_size,
                    filters=128, num_head=8, dropout=0.1)
# model=multi_gpu_model(model,gpus=2)
optimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy', 'mae', 'mae'], \
              loss_weights=[1, 1, 0, 0])


class MyCallback(Callback):
    def on_train_begin(self, logs={}):
        if do_ema:
            self.ema_trainable_weights_vals = ExponentialMovingAverage_TrainBegin(model)
        self.global_step = 1
        lr = min(0.001, 0.001 / np.log(1000) * np.log(self.global_step))
        K.set_value(model.optimizer.lr, lr)
        self.best_f1 = 0

    def on_batch_end(self, batch, logs={}):
        self.global_step += 1
        if self.global_step <= 1000:
            lr = min(0.001, 0.001 / np.log(1000) * np.log(self.global_step))
            K.set_value(model.optimizer.lr, lr)
        if do_ema:
            self.ema_trainable_weights_vals = ExponentialMovingAverage_BatchEnd(model, self.ema_trainable_weights_vals, \
                                                                                min(0.9999, (1 + self.global_step) / (
                                                                                            10 + self.global_step)))

    def on_epoch_end(self, epoch, logs={}):
        if not do_ema:
            results = model.predict([val_context_word, val_question_word, val_context_char, val_question_char], \
                                    verbose=1, batch_size=64)
            _, _, y_start_pred, y_end_pred = results
            y_start_pred = np.reshape(y_start_pred, (-1))
            y_end_pred = np.reshape(y_end_pred, (-1))
            answer_dict, remapped_dict = util.convert_tokens(eval_file, val_qid.tolist(), \
                                                             y_start_pred.tolist(), y_end_pred.tolist())
            metrics = util.evaluate(eval_file, answer_dict)
            print("Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))
            ems.append(metrics['exact_match'])
            f1s.append(metrics['f1'])
            if metrics['f1'] > self.best_f1:
                self.best_f1 = metrics['f1']
                model.save_weights('model/QANet_v99.h5')
            if epoch + 1 == 25:
                model.save_weights('model/QANet_v99_60k.h5')
        else:
            # validation with ema
            # save backup weights
            print('saving temp weights...')
            model.save_weights('temp_model2.h5')
            ExponentialMovingAverage_EpochEnd(model, self.ema_trainable_weights_vals)
            results = model.predict([val_context_word, val_question_word, val_context_char, val_question_char], \
                                    verbose=1, batch_size=64)
            _, _, y_start_pred, y_end_pred = results
            y_start_pred = np.reshape(y_start_pred, (-1))
            y_end_pred = np.reshape(y_end_pred, (-1))
            answer_dict, remapped_dict = util.convert_tokens(eval_file, val_qid.tolist(), y_start_pred.tolist(), \
                                                             y_end_pred.tolist())
            metrics_ema = util.evaluate(eval_file, answer_dict)
            print("After EMA, Exact Match: {}, F1: {}".format(metrics_ema['exact_match'], metrics_ema['f1']))
            ems.append(metrics_ema['exact_match'])
            f1s.append(metrics_ema['f1'])
            if metrics_ema['f1'] > self.best_f1:
                self.best_f1 = metrics_ema['f1']
                model.save_weights('model/QANet_ema_v99.h5')
            if epoch + 1 == 25:
                model.save_weights('model/QANet_ema_v99_60k.h5')
            # load backup
            print('loading temp weights...')
            model.load_weights('temp_model2.h5')
        result = pd.DataFrame([ems, f1s], index=['em', 'f1']).transpose()
        result.to_csv('log/result2.csv', index=None)


model.fit([context_word, question_word, context_char, question_char],
          [start_label, end_label, start_label_fin, end_label_fin], \
          batch_size=32, epochs=n_epochs, callbacks=[MyCallback()])