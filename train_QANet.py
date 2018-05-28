import numpy as np
import QANet_keras_v2 as QANet
import os
from keras.optimizers import *
import util
from keras.models import *
import json
from tqdm import tqdm
from keras.utils import multi_gpu_model
from ExponentialMovingAverage import *
import tensorflow as tf
from keras.callbacks import Callback

os.environ["CUDA_VISIBLE_DEVICES"] = '0,2'

# load trainset
context_word = np.load('dataset2/train_contw_input.npy')
question_word = np.load('dataset2/train_quesw_input.npy')
context_char = np.load('dataset2/train_contc_input.npy')
question_char = np.load('dataset2/train_quesc_input.npy')
context_length = np.load('dataset2/train_cont_len.npy')
question_length = np.load('dataset2/train_ques_len.npy')
start_label = np.load('dataset2/train_y_start.npy')
end_label = np.load('dataset2/train_y_end.npy')

# load valset
val_context_word = np.load('dataset2/dev_contw_input.npy')
val_question_word = np.load('dataset2/dev_quesw_input.npy')
val_context_char = np.load('dataset2/dev_contc_input.npy')
val_question_char = np.load('dataset2/dev_quesc_input.npy')
val_context_length = np.load('dataset2/dev_cont_len.npy')
val_question_length = np.load('dataset2/dev_ques_len.npy')
val_start_label = np.load('dataset2/dev_y_start.npy')
val_end_label = np.load('dataset2/dev_y_end.npy')
val_qid = np.load('dataset2/dev_qid.npy').astype(np.int32)
with open('dataset2/dev_eval.json', "r") as fh:
    eval_file = json.load(fh)

# load embedding matrix
embedding_matrix = np.load('word_emb_mat2.npy')

# parameters
char_dim = 64
cont_limit = 400
ques_limit = 50
char_limit = 16
char_input_size = 1426

batch_size = 64
n_epochs = 30
early_stop = 10
it = 0
continue_train = False
do_ema = True
use_hand_feat = False
hand_feat_dim = 0

if use_hand_feat:
    hand_feat = np.load('dataset2/train_hand_feat.npy')
    val_hand_feat = np.load('dataset2/dev_hand_feat.npy')
    hand_feat_dim = hand_feat.shape[-1]

np.random.seed(10000)
model = QANet.QANet(word_dim=300, char_dim=char_dim, cont_limit=cont_limit, ques_limit=ques_limit,
                    char_limit=char_limit, word_mat=embedding_matrix, char_input_size=char_input_size, filters=128,
                    num_head=1, hand_feat_dim=hand_feat_dim, dropout=0.1)
model = multi_gpu_model(model, gpus=2)
optimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')


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
            self.ema_trainable_weights_vals = ExponentialMovingAverage_BatchEnd(model, self.ema_trainable_weights_vals,
                                                                                min(0.9999, (1 + self.global_step) / (
                                                                                            10 + self.global_step)))

    def on_epoch_end(self, epoch, logs={}):
        if not do_ema:
            if use_hand_feat:
                results = model.predict(
                    [val_context_word, val_question_word, val_context_char, val_question_char, val_context_length,
                     val_question_length, val_hand_feat], verbose=1, batch_size=64)
            else:
                results = model.predict(
                    [val_context_word, val_question_word, val_context_char, val_question_char, val_context_length,
                     val_question_length], verbose=1, batch_size=64)
            y_start_pred, y_end_pred = results
            y_start_pred = np.argmax(y_start_pred, axis=1)
            y_end_pred = np.argmax(y_end_pred, axis=1)
            answer_dict, remapped_dict = util.convert_tokens(eval_file, val_qid.tolist(), y_start_pred.tolist(),
                                                             y_end_pred.tolist())
            metrics = util.evaluate(eval_file, answer_dict)
            print("Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))
            if metrics['f1'] > self.best_f1:
                self.best_f1 = metrics['f1']
                model.save_weights('model/QANet_v11.h5')
        else:
            # validation with ema
            # save backup weights
            print('saving temp weights...')
            model.save_weights('temp_model1.h5')
            ExponentialMovingAverage_EpochEnd(model, self.ema_trainable_weights_vals)
            if use_hand_feat:
                results = model.predict(
                    [val_context_word, val_question_word, val_context_char, val_question_char, val_context_length,
                     val_question_length, val_hand_feat], verbose=1, batch_size=64)
            else:
                results = model.predict(
                    [val_context_word, val_question_word, val_context_char, val_question_char, val_context_length,
                     val_question_length], verbose=1, batch_size=64)
            y_start_pred, y_end_pred = results
            y_start_pred = np.argmax(y_start_pred, axis=1)
            y_end_pred = np.argmax(y_end_pred, axis=1)

            answer_dict, remapped_dict = util.convert_tokens(eval_file, val_qid.tolist(), y_start_pred.tolist(),
                                                             y_end_pred.tolist())
            metrics_ema = util.evaluate(eval_file, answer_dict)
            print("After EMA, Exact Match: {}, F1: {}".format(metrics_ema['exact_match'], metrics_ema['f1']))
            if metrics_ema['f1'] > self.best_f1:
                self.best_f1 = metrics_ema['f1']
                model.save_weights('model/QANet_ema_v11.h5')
            # load backup
            print('loading temp weights...')
            model.load_weights('temp_model1.h5')


if not use_hand_feat:
    model.fit([context_word, question_word, context_char, question_char, context_length, question_length],
              [start_label, end_label], batch_size=64, epochs=n_epochs, callbacks=[MyCallback()])
else:
    model.fit([context_word, question_word, context_char, question_char, context_length, question_length, hand_feat],
              [start_label, end_label], batch_size=64, epochs=n_epochs, callbacks=[MyCallback()])