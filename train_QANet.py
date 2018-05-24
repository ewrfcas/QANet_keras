import numpy as np
import QANet_keras as QANet
import os
from keras.optimizers import *
import util
from keras.models import *
import json
from tqdm import tqdm
from keras.utils import multi_gpu_model
from ExponentialMovingAverage import *
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# load trainset
context_word = np.load('dataset/train_contw_input.npy')
question_word = np.load('dataset/train_quesw_input.npy')
context_char = np.load('dataset/train_contc_input.npy')
question_char = np.load('dataset/train_quesc_input.npy')
context_length = np.load('dataset/train_cont_len.npy')
question_length = np.load('dataset/train_ques_len.npy')
start_label = np.load('dataset/train_y_start.npy')
end_label = np.load('dataset/train_y_end.npy')

# load valset
val_context_word = np.load('dataset/dev_contw_input.npy')
val_question_word = np.load('dataset/dev_quesw_input.npy')
val_context_char = np.load('dataset/dev_contc_input.npy')
val_question_char = np.load('dataset/dev_quesc_input.npy')
val_context_length = np.load('dataset/dev_cont_len.npy')
val_question_length = np.load('dataset/dev_ques_len.npy')
val_start_label = np.load('dataset/dev_y_start.npy')
val_end_label = np.load('dataset/dev_y_end.npy')
val_qid = np.load('dataset/dev_qid.npy').astype(np.int32)
with open('dataset/dev_eval.json', "r") as fh:
    eval_file = json.load(fh)

# load embedding matrix
embedding_matrix = np.load('word_emb_mat.npy')

# parameters
char_dim = 64
cont_limit = 400
ques_limit = 50
char_limit = 16
char_input_size = 1426

batch_size = 32
n_epochs = 30
early_stop = 10
it = 0
continue_train = False
do_ema = True
use_hand_feat = True
multi_gpu = True
hand_feat_dim = 0

if use_hand_feat:
    hand_feat = np.load('dataset/train_hand_feat.npy')
    val_hand_feat = np.load('dataset/dev_hand_feat.npy')
    hand_feat_dim = hand_feat.shape[-1]

np.random.seed(10000)
model = QANet.QANet(word_dim=300, char_dim=char_dim, cont_limit=cont_limit, ques_limit=ques_limit,
                    char_limit=char_limit, word_mat=embedding_matrix, char_input_size=char_input_size, filters=128,
                    num_head=8, hand_feat_dim=hand_feat_dim, dropout=0.1)
optimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
if multi_gpu:
    model=multi_gpu_model(model,gpus=2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

# load_weights
if continue_train:
    model.load_weights('model/QANetv02.h5')

# load hand_feat
n_batch = int(context_word.shape[0] / batch_size)
global_step = 0
best_f1 = 0
best_f1_ema = 0
ems = []
f1s = []
ems_ema = []
f1s_ema = []
if do_ema:
    ema_trainable_weights_vals = ExponentialMovingAverage_TrainBegin(model)
for epoch in range(n_epochs):
    np.random.seed(epoch + 10086)
    # training step
    index = np.arange(context_word.shape[0])
    np.random.shuffle(index)
    context_word = context_word[index, ::]
    question_word = question_word[index, ::]
    context_char = context_char[index, ::]
    question_char = question_char[index, ::]
    context_length = context_length[index, ::]
    question_length = question_length[index, ::]
    start_label = start_label[index, ::]
    end_label = end_label[index, ::]
    if use_hand_feat:
        hand_feat = hand_feat[index, ::]

    sum_loss = 0
    last_train_str = ""
    for i in range(n_batch):
        global_step += 1
        if global_step <= 1000:
            lr = min(0.001, 0.001 / np.log(1000) * np.log(global_step))
            K.set_value(model.optimizer.lr, lr)
        # get batch dataset
        context_word_temp = context_word[i * batch_size:(i + 1) * batch_size, ::]
        question_word_temp = question_word[i * batch_size:(i + 1) * batch_size, ::]
        context_char_temp = context_char[i * batch_size:(i + 1) * batch_size, ::]
        question_char_temp = question_char[i * batch_size:(i + 1) * batch_size, ::]
        context_length_temp = context_length[i * batch_size:(i + 1) * batch_size, ::]
        question_length_temp = question_length[i * batch_size:(i + 1) * batch_size, ::]
        start_label_temp = start_label[i * batch_size:(i + 1) * batch_size, ::]
        end_label_temp = end_label[i * batch_size:(i + 1) * batch_size, ::]
        if use_hand_feat:
            hand_feat_temp = hand_feat[i * batch_size:(i + 1) * batch_size, ::]
            loss_value = model.train_on_batch(
                [context_word_temp, question_word_temp, context_char_temp, question_char_temp,
                 context_length_temp, question_length_temp, hand_feat_temp], [start_label_temp, end_label_temp])
        else:
            loss_value = model.train_on_batch(
                [context_word_temp, question_word_temp, context_char_temp, question_char_temp,
                 context_length_temp, question_length_temp], [start_label_temp, end_label_temp])
        if do_ema:
            ema_trainable_weights_vals = ExponentialMovingAverage_BatchEnd(model, ema_trainable_weights_vals,
                                                                           min(0.9999,
                                                                               (1 + global_step) / (10 + global_step)))
        sum_loss += loss_value[0]
        last_train_str = "\r[epoch:%d/%d, steps:%d/%d] -loss: %.4f" % (
        epoch + 1, n_epochs, i + 1, n_batch, sum_loss / (i + 1))
        print(last_train_str, end='      ', flush=True)

    # validating step
    print('\n')
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
        if metrics['f1'] > best_f1:
            it = 0
            best_f1 = metrics['f1']
            model.save_weights('model/QANet_v08.h5')
        else:
            it += 1
    else:
        # validation with ema
        # save backup weights
        print('saving temp weights...')
        model.save_weights('temp_model2.h5')
        ExponentialMovingAverage_EpochEnd(model, ema_trainable_weights_vals)
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
        if metrics_ema['f1'] > best_f1_ema:
            it = 0
            best_f1_ema = metrics_ema['f1']
            model.save_weights('model/QANet_ema_v08.h5')
        else:
            it += 1
        # load backup
        print('loading temp weights...')
        model.load_weights('temp_model2.h5')
    if it == early_stop:
        print('early stopped')
        break
    print('\n')