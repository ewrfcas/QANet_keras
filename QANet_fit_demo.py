from keras.optimizers import *
from keras.initializers import *
from QANet_keras import QANet

embedding_matrix = np.random.random((10000,300))
model=QANet(word_mat=embedding_matrix)

optimizer=Adam(lr=0.001,beta_1=0.8,beta_2=0.999,epsilon=1e-7)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

# load data
char_dim=200
cont_limit=400
ques_limit=50
char_limit=16

context_word = np.random.randint(0, 10000, (300, cont_limit))
question_word = np.random.randint(0, 10000, (300, ques_limit))
context_char = np.random.randint(0, 96, (300, cont_limit, char_limit))
question_char = np.random.randint(0, 96, (300, ques_limit, char_limit))
context_length = np.random.randint(5, cont_limit, (300, 1))
question_length = np.random.randint(5, ques_limit, (300, 1))
start_label = np.random.randint(0, 2, (300, cont_limit))
end_label = np.random.randint(0, 2, (300, cont_limit))

model.fit([context_word,question_word,context_char,question_char,context_length,question_length],[start_label,end_label],batch_size=8)
