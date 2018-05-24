from keras.layers import *
from keras.regularizers import *
from keras.models import *
from context2query_attention import context2query_attention
from multihead_attention import Attention as SelfAttention
from position_embedding import Position_Embedding as PosEmbedding
from keras import layers
from keras.optimizers import *
from keras.callbacks import *
from keras.initializers import *

regularizer = l2(3e-7)
VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=2018)

def highway(highway_layers, x, num_layers=2, dropout=0.0):
    # reduce dim
    x = highway_layers[0](x)
    for i in range(num_layers):
        T = highway_layers[i * 2 + 1](x)
        H = highway_layers[i * 2 + 2](x)
        H = Dropout(dropout)(H)
        x = Lambda(lambda v: v[0] * v[1] + v[2] * (1 - v[1]))([H, T, x])
    return x

def conv_block(conv_layers, x, num_conv=4, dropout= 0.0):
    x = Lambda(lambda v: K.expand_dims(v, axis=2))(x)
    for i in range(num_conv):
        residual = x
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = conv_layers[i][0](x)
        x = conv_layers[i][1](x)
        x = layers.add([x, residual])
    x = Lambda(lambda v: tf.squeeze(v, axis=2))(x)
    return x

def attention_block(attention_layer, x, len, dropout=0.0):
    residual = x
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = attention_layer([x,x,x,len,len])
    x = layers.add([x, residual])
    return x

def feed_forward_block(FeedForward_layers, x, dropout=0.0):
    residual = x
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = FeedForward_layers[0](x)
    x = FeedForward_layers[1](x)
    x = layers.add([x, residual])
    return x

def mask_logits(inputs, mask, mask_value = -1e12, axis=1, time_dim=1):
    mask = K.cast(mask, tf.int32)
    mask = K.one_hot(mask[:, 0], K.shape(inputs)[time_dim])
    mask = 1 - K.cumsum(mask, 1)
    mask = tf.cast(mask, tf.float32)
    if axis!=0:
        mask = tf.expand_dims(mask, axis)
    return inputs + mask_value * (1 - mask)

def QANet(word_dim=300, char_dim=200, cont_limit=400, ques_limit=50, char_limit=16, word_mat=None, char_input_size=1000, filters=128, num_head=8, hand_feat_dim=0, dropout=0.1):

    # Input Embedding Layer
    contw_input = Input((cont_limit,))
    quesw_input = Input((ques_limit,))
    contc_input = Input((cont_limit, char_limit))
    quesc_input = Input((ques_limit, char_limit))
    cont_len = Input((None,))
    ques_len = Input((None,))
    if hand_feat_dim!=0:
        handcraft_input = Input((cont_limit, hand_feat_dim))

    # embedding word
    xw_cont = Embedding(word_mat.shape[0], word_dim, weights=[word_mat], input_length=cont_limit, mask_zero=False, trainable=False)(contw_input)
    xw_ques = Embedding(word_mat.shape[0], word_dim, weights=[word_mat], input_length=ques_limit, mask_zero=False, trainable=False)(quesw_input)

    # embedding char
    CharEmbedding = Embedding(char_input_size, char_dim, input_length=char_limit, mask_zero=False, name='char_embedding')
    xc_cont=CharEmbedding(contc_input)
    xc_ques=CharEmbedding(quesc_input)
    char_conv = Conv1D(filters, 5, activation='relu', kernel_regularizer=regularizer, name='char_conv')
    xc_cont = Lambda(lambda x: tf.reshape(x, (-1, char_limit, char_dim)))(xc_cont)
    xc_ques = Lambda(lambda x: tf.reshape(x, (-1, char_limit, char_dim)))(xc_ques)
    xc_cont = char_conv(xc_cont)
    xc_ques = char_conv(xc_ques)
    xc_cont = GlobalMaxPooling1D()(xc_cont)
    xc_ques = GlobalMaxPooling1D()(xc_ques)
    xc_cont = Lambda(lambda x: tf.reshape(x, (-1, cont_limit, filters)))(xc_cont)
    xc_ques = Lambda(lambda x: tf.reshape(x, (-1, ques_limit, filters)))(xc_ques)

    # highwayNet
    x_cont = Concatenate()([xw_cont, xc_cont])
    x_ques = Concatenate()([xw_ques, xc_ques])

    # highway shared layers
    highway_layers = [Conv1D(filters, 1, kernel_regularizer=regularizer)]
    for i in range(2):
        highway_layers.append(Conv1D(filters, 1, kernel_regularizer=regularizer, activation='sigmoid'))
        highway_layers.append(Conv1D(filters, 1, kernel_regularizer=regularizer, activation='linear'))
    x_cont = highway(highway_layers, x_cont, num_layers=2, dropout=dropout)
    x_ques = highway(highway_layers, x_ques, num_layers=2, dropout=dropout)

    # build shared layers
    # shared convs
    DepthwiseConv_share_1 = []
    for i in range(4):
        DepthwiseConv_share_1.append([DepthwiseConv2D((7, 1), activation='relu', kernel_regularizer=regularizer,padding='same', depth_multiplier=1),
                                      Conv2D(filters, 1, padding='same', kernel_regularizer=regularizer)])
    # shared attention
    head_size = filters//num_head
    SelfAttention_share_1 = SelfAttention(num_head, head_size)
    # shared feed-forward
    FeedForward_share_1 = []
    FeedForward_share_1.append(Conv1D(filters, 1, kernel_regularizer=regularizer, activation='relu'))
    FeedForward_share_1.append(Conv1D(filters, 1, kernel_regularizer=regularizer, activation='linear'))

    # context part
    x_cont = PosEmbedding()(x_cont)
    x_cont = conv_block(DepthwiseConv_share_1, x_cont, 4, dropout)
    x_cont = attention_block(SelfAttention_share_1, x_cont, cont_len, dropout)
    x_cont = feed_forward_block(FeedForward_share_1, x_cont, dropout)

    # question part
    x_ques = PosEmbedding()(x_ques)
    x_ques = conv_block(DepthwiseConv_share_1, x_ques, 4, dropout)
    x_ques = attention_block(SelfAttention_share_1, x_ques, ques_len, dropout)
    x_ques = feed_forward_block(FeedForward_share_1, x_ques, dropout)

    # Context_to_Query_Attention_Layer
    x = context2query_attention(512, cont_limit, ques_limit, dropout)([x_cont, x_ques, cont_len, ques_len])
    # handcraft
    if hand_feat_dim != 0:
        x = Concatenate()([x, handcraft_input])
    x = Conv1D(filters, 1, kernel_regularizer=regularizer, activation='linear')(x)

    # Model_Encoder_Layer
    # shared layers
    DepthwiseConv_share_2 = []
    SelfAttention_share_2 = []
    FeedForward_share_2 = []
    for i in range(7):
        DepthwiseConv_share_2_temp = []
        for i in range(2):
            DepthwiseConv_share_2_temp.append([DepthwiseConv2D((5, 1), activation='relu',
                                                               kernel_regularizer=regularizer, padding='same',
                                                               depth_multiplier=1),
                                               Conv2D(filters, 1, padding='same', kernel_regularizer=regularizer)])
        DepthwiseConv_share_2.append(DepthwiseConv_share_2_temp)
        SelfAttention_share_2.append(SelfAttention(num_head, head_size))
        FeedForward_share_2.append([Conv1D(filters, 1, kernel_regularizer=regularizer, activation='relu'),
                                    Conv1D(filters, 1, kernel_regularizer=regularizer, activation='linear')])

    outputs = [x]
    for i in range(3):
        x = outputs[-1]
        for j in range(7):
            x = PosEmbedding()(x)
            x = conv_block(DepthwiseConv_share_2[j], x, 2, dropout)
            x = attention_block(SelfAttention_share_2[j], x, cont_len, dropout)
            x = feed_forward_block(FeedForward_share_2[j], x, dropout)
        outputs.append(x)

    # Output_Layer
    x_start = Concatenate()([outputs[1], outputs[2]])
    x_start = Conv1D(1, 1, activation='linear')(x_start)
    x_start = Lambda(lambda x:tf.squeeze(x,axis=-1))(x_start)
    x_start = Lambda(lambda x:mask_logits(x[0], x[1], axis=0, time_dim=1))([x_start,cont_len])
    x_start = Lambda(lambda x:K.softmax(x),name='start')(x_start)

    x_end = Concatenate()([outputs[1], outputs[3]])
    x_end = Conv1D(1, 1, activation='linear')(x_end)
    x_end = Lambda(lambda x: tf.squeeze(x, axis=-1))(x_end)
    x_end = Lambda(lambda x: mask_logits(x[0], x[1], axis=0, time_dim=1))([x_end, cont_len])
    x_end = Lambda(lambda x: K.softmax(x),name='end')(x_end)

    if hand_feat_dim==0:
        return Model(inputs=[contw_input, quesw_input, contc_input, quesc_input, cont_len, ques_len],
                     outputs=[x_start,x_end])
    else:
        return Model(inputs=[contw_input, quesw_input, contc_input, quesc_input, cont_len, ques_len, handcraft_input],
                     outputs=[x_start, x_end])

embedding_matrix = np.random.random((10000,300))
model=QANet(word_mat=embedding_matrix)

optimizer=Adam(lr=0.001,beta_1=0.8,beta_2=0.999,epsilon=1e-7)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

# call backs
class LRSetting(Callback):
    def on_batch_begin(self, batch, logs=None):
        lr = min(0.001, 0.001 / np.log(999.) * np.log(batch + 1))
        K.set_value(self.model.optimizer.lr, lr)
lr_setting = LRSetting()
check_point = ModelCheckpoint('model/QANetv02.h5', monitor='val_loss', verbose=0, save_best_only=True,save_weights_only=True, mode='auto', period=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

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
