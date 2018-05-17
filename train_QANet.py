import numpy as np
import QANet_keras as QANet
import os
from keras.optimizers import *
from keras.callbacks import *
import keras.backend as K
import util
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load trainset
context_word=np.load('dataset/train_contw_input.npy') 
question_word=np.load('dataset/train_quesw_input.npy') 
context_char=np.load('dataset/train_contc_input.npy') 
question_char=np.load('dataset/train_quesc_input.npy') 
context_length=np.load('dataset/train_cont_len.npy') 
question_length=np.load('dataset/train_ques_len.npy') 
start_label=np.load('dataset/train_y_start.npy') 
end_label=np.load('dataset/train_y_end.npy')

# load valset
val_context_word=np.load('dataset/dev_contw_input.npy') 
val_question_word=np.load('dataset/dev_quesw_input.npy') 
val_context_char=np.load('dataset/dev_contc_input.npy') 
val_question_char=np.load('dataset/dev_quesc_input.npy') 
val_context_length=np.load('dataset/dev_cont_len.npy') 
val_question_length=np.load('dataset/dev_ques_len.npy') 
val_start_label=np.load('dataset/dev_y_start.npy') 
val_end_label=np.load('dataset/dev_y_end.npy')
val_qid=np.load('dataset/dev_qid.npy').astype(np.int32)
with open('dataset/dev_eval.json', "r") as fh:
    eval_file = json.load(fh)

# load embedding matrix
embedding_matrix=np.load('word_emb_mat.npy')

char_dim=200
cont_limit=400
ques_limit=50
char_limit=16
char_input_size=1373

model=QANet.QANet(word_dim=300, char_dim=char_dim, cont_limit=cont_limit, ques_limit=ques_limit, char_limit=char_limit, word_mat=embedding_matrix, char_input_size=char_input_size, filters=128, num_head=8, dropout=0.1)

optimizer=Adam(lr=0.001,beta_1=0.8,beta_2=0.999,epsilon=1e-7)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

# train on batch
batch_size=32
n_epochs=20
early_stop=10
continue_train=True

# load_weights
if continue_train:
    model.load_weights('model/QANetv02.h5')

n_batch = int(context_word.shape[0] / batch_size)
global_step=0
best_f1=0
for epoch in range(n_epochs):
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

    sum_loss = 0
    last_train_str = ""
    for i in range(n_batch):
        global_step+=1
        if global_step<=1000:
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
        loss_value=model.train_on_batch([context_word_temp,question_word_temp,context_char_temp,question_char_temp,
                                         context_length_temp,question_length_temp],[start_label_temp,end_label_temp])
        
        sum_loss += loss_value[0]
        last_train_str = "\r[epoch:%d/%d, steps:%d/%d] -loss: %.4f"%(epoch + 1, n_epochs, i + 1, n_batch, sum_loss / (i + 1))
        print(last_train_str, end='      ', flush=True)

    # validating step
    results=model.predict([val_context_word,val_question_word,val_context_char,val_question_char,val_context_length,
                          val_question_length],verbose=1,batch_size=32)
    y_start_pred, y_end_pred = results
    y_start_pred = np.argmax(y_start_pred, axis=1)
    y_end_pred = np.argmax(y_end_pred, axis=1)
    
    answer_dict, remapped_dict = util.convert_tokens(eval_file, val_qid.tolist(), y_start_pred.tolist(), y_end_pred.tolist())
    metrics = util.evaluate(eval_file, answer_dict)
    print("Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))
    if metrics['f1']>best_f1:
        best_f1=metrics['f1']
        model.save_weights('model/QANetv03.h5')