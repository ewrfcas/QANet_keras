import QANet_keras as QANet
import os
from keras.optimizers import *
from util import *
from keras.models import *
import json
import pandas as pd
import time
import pickle
from keras.utils import multi_gpu_model
from layers.ExponentialMovingAverage import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# load trainset
with open('dataset/train_total_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
train_data['start_label_fin'] = np.argmax(train_data['y_start'], axis=-1)
train_data['end_label_fin'] = np.argmax(train_data['y_end'], axis=-1)

# load valset
with open('dataset/dev_total_data.pkl', 'rb') as f:
    dev_data = pickle.load(f)
dev_data['start_label_fin'] = np.argmax(dev_data['y_start'], axis=-1)
dev_data['end_label_fin'] = np.argmax(dev_data['y_end'], axis=-1)
with open('dataset/test_eval.json', "r") as fh:
    eval_file = json.load(fh)

# load embedding matrix
word_mat = np.load('dataset/word_emb_mat.npy')
char_mat = np.load('dataset/char_emb_mat.npy')

# parameters
config = {
    'word_dim': 300,
    'char_dim': 64,
    'cont_limit': 400,
    'ques_limit': 50,
    'char_limit': 16,
    'ans_limit': 30,
    'char_input_size': 1233,
    'filters': 128,
    'num_head': 8,
    'dropout': 0.1,
    'batch_size': 24,
    'epoch': 25,
    'ema_decay': 0.9999,
    'learning_rate': 1e-3,
    'path': 'QA001',
    'use_cove': False
}

ems = []
f1s = []

cove_model = None
if config['use_cove']:
    cove_model = load_model('model/Keras_CoVe_nomask.h5')
    for layer in cove_model.layers:
        layer.trainable = False

model = QANet.QANet(config, word_mat=word_mat, char_mat=char_mat, cove_model=cove_model)
# model = multi_gpu_model(model, gpus=2)
model.summary()
optimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy', 'mae', 'mae'],
              loss_weights=[0.5, 0.5, 0, 0])

n_batch = train_data['context_id'].shape[0] // config['batch_size']
n_batch_val = dev_data['context_id'].shape[0] // config['batch_size']
if dev_data['context_id'].shape[0] % config['batch_size'] != 0:
    n_batch_val += 1

global_step = 1
lr = min(config['learning_rate'], config['learning_rate'] / np.log(1000) * np.log(global_step))
K.set_value(model.optimizer.lr, lr)
train_set = [train_data['context_id'], train_data['question_id'], train_data['context_char_id'],
             train_data['question_char_id'], train_data['y_start'], train_data['y_end'], train_data['start_label_fin'],
             train_data['end_label_fin']]
val_set = [dev_data['context_id'], dev_data['question_id'], dev_data['context_char_id'], dev_data['question_char_id']]

# ema_trainable_weights_vals = ExponentialMovingAverage_TrainBegin(model)
for epoch in range(config['epoch']):
    train_set = training_shuffle(train_set)
    t_start = time.time()
    last_train_str = "\r"

    # training step
    sum_loss = 0
    for i in range(n_batch):
        contw_input, quesw_input, contc_input, \
        quesc_input, y_start, y_end, y1f, y2f = next_batch(train_set, config['batch_size'], i)
        y_start, y_end = slice_for_batch(contw_input, y_start, y_end)
        loss_value, _, _, _, _ = model.train_on_batch([contw_input, quesw_input, contc_input, quesc_input],
                                                      [y_start, y_end, y1f, y2f])
        global_step += 1
        if global_step <= 1000:
            lr = min(config['learning_rate'], config['learning_rate'] / np.log(1000) * np.log(global_step))
            K.set_value(model.optimizer.lr, lr)
            # print('lr:',K.get_value(model.optimizer.lr))
        # ema_trainable_weights_vals = ExponentialMovingAverage_BatchEnd(model, ema_trainable_weights_vals, config['ema_decay'])
        sum_loss += loss_value
        last_train_str = "\r[epoch:%d/%d, steps:%d/%d] -ETA:%ds -loss:%.4f" % (
            epoch + 1, config['epoch'], i + 1, n_batch, cal_ETA(t_start, i, n_batch), sum_loss / (i + 1))
        print(last_train_str, end='      ', flush=True)

    # validating step
    # print('saving temp weights...')
    # model.save_weights('temp_model.h5')
    # ExponentialMovingAverage_EpochEnd(model, ema_trainable_weights_vals)
    y1s = []
    y2s = []
    last_val_str = "\r"
    for i in range(n_batch_val):
        contw_input, quesw_input, contc_input, quesc_input = next_batch(val_set, config['batch_size'], i)
        _, _, y1, y2 = model.predict_on_batch([contw_input, quesw_input, contc_input, quesc_input])
        y1s.extend(np.squeeze(y1, axis=-1))
        y2s.extend(np.squeeze(y2, axis=-1))
        last_val_str = last_train_str + " [validate:%d/%d]" % (i + 1, n_batch_val)
        print(last_val_str, end='      ', flush=True)
    y1s = np.array(y1s).astype(np.int32)
    y2s = np.array(y2s).astype(np.int32)
    answer_dict, _ = convert_tokens(eval_file, dev_data['qid'].tolist(), y1s.tolist(), y2s.tolist())
    metrics = evaluate(eval_file, answer_dict)
    ems.append(metrics['exact_match'])
    f1s.append(metrics['f1'])
    print(last_val_str, " -EM:%.2f%%, -F1:%.2f%%" % (metrics['exact_match'], metrics['f1']), end=' ', flush=True)
    print('\n')
    result = pd.DataFrame([ems, f1s], index=['em', 'f1']).transpose()
    result.to_csv('logs/result_' + config['path'] + '.csv', index=None)
    model.save_weights('model/QANet_model_' + config['path'] + '.h5')
    # load backup
    # print('loading temp weights...')
    # model.load_weights('temp_model.h5')
