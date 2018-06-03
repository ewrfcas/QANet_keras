from keras.optimizers import *
from keras.initializers import *
from QANet_keras import QANet

embedding_matrix = np.random.random((10000,300))
embedding_matrix_char = np.random.random((1000,64))
model=QANet(word_mat=embedding_matrix, char_mat=embedding_matrix_char)

optimizer=Adam(lr=0.001,beta_1=0.8,beta_2=0.999,epsilon=1e-7)
model.compile(optimizer=optimizer, loss=['categorical_crossentropy','categorical_crossentropy','mae','mae'], loss_weights=[1, 1, 0, 0])

# load data
char_dim=200
cont_limit=400
ques_limit=50
char_limit=16

context_word = np.random.randint(0, 10000, (300, cont_limit))
question_word = np.random.randint(0, 10000, (300, ques_limit))
context_char = np.random.randint(0, 96, (300, cont_limit, char_limit))
question_char = np.random.randint(0, 96, (300, ques_limit, char_limit))
start_label = np.random.randint(0, 2, (300, cont_limit))
end_label = np.random.randint(0, 2, (300, cont_limit))
start_label_fin = np.argmax(start_label,axis=-1)
end_label_fin = np.argmax(end_label,axis=-1)

model.fit([context_word,question_word,context_char,question_char],[start_label,end_label,start_label_fin,end_label_fin],batch_size=8)
