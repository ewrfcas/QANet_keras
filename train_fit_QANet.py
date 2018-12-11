import QANet_keras as QANet
import os
from keras.optimizers import Adam
from util import *
from keras.models import load_model
import json
import pandas as pd
import pickle
from keras.callbacks import Callback
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
    'path': 'QA001fit',
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
model.summary()
optimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy', 'mae', 'mae'],
              loss_weights=[0.5, 0.5, 0, 0])


class QANet_callback(Callback):
    def __init__(self):
        self.global_step = 1
        self.max_f1 = 0
        super(Callback, self).__init__()

    def on_train_begin(self, logs=None):
        lr = min(config['learning_rate'], config['learning_rate'] / np.log(1000) * np.log(self.global_step))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        lr = min(config['learning_rate'], config['learning_rate'] / np.log(1000) * np.log(self.global_step))
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        _, _, y1, y2 = self.model.predict(x=[dev_data['context_id'], dev_data['question_id'],
                                             dev_data['context_char_id'], dev_data['question_char_id']],
                                          batch_size=config['batch_size'],
                                          verbose=1)
        y1s = np.squeeze(y1, axis=-1).astype(np.int32)
        y2s = np.squeeze(y2, axis=-1).astype(np.int32)
        answer_dict, _ = convert_tokens(eval_file, dev_data['qid'].tolist(), y1s.tolist(), y2s.tolist())
        metrics = evaluate(eval_file, answer_dict)
        ems.append(metrics['exact_match'])
        f1s.append(metrics['f1'])
        print("-EM:%.2f%%, -F1:%.2f%%" % (metrics['exact_match'], metrics['f1']))
        result = pd.DataFrame([ems, f1s], index=['em', 'f1']).transpose()
        result.to_csv('logs/result_' + config['path'] + '.csv', index=None)
        if f1s[-1] > self.max_f1:
            self.max_f1 = f1s[-1]
            model.save_weights('model/QANet_model_' + config['path'] + '.h5')

qanet_callback = QANet_callback()
qanet_callback.set_model(model)

model.fit(x=[train_data['context_id'], train_data['question_id'],
             train_data['context_char_id'], train_data['question_char_id']],
          y=[train_data['y_start'], train_data['y_end'], train_data['start_label_fin'],
             train_data['end_label_fin']],
          batch_size=config['batch_size'],
          epochs=config['epoch'],
          callbacks=[qanet_callback])
