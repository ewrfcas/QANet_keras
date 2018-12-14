from keras.layers import *
from keras.regularizers import *
from keras.models import *
from layers.context2query_attention import context2query_attention
from layers.multihead_attention import Attention as MultiHeadAttention
from layers.position_embedding import Position_Embedding as PositionEmbedding
from layers.layer_norm import LayerNormalization
from layers.layer_dropout import LayerDropout
from layers.QAoutputBlock import QAoutputBlock
from layers.BatchSlice import BatchSlice
from layers.DepthwiseConv1D import DepthwiseConv1D
from layers.LabelPadding import LabelPadding
from keras.initializers import VarianceScaling
import tensorflow as tf
import keras.backend as K

regularizer = l2(3e-7)
init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')
init_relu = VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def highway(highway_layers, x, num_layers=2, dropout=0.0):
    # reduce dim
    x = highway_layers[0](x)
    for i in range(num_layers):
        T = highway_layers[i * 2 + 1](x)
        H = highway_layers[i * 2 + 2](x)
        H = Dropout(dropout)(H)
        x = Lambda(lambda v: v[0] * v[1] + v[2] * (1 - v[1]))([H, T, x])
    return x


def conv_block(conv_layers, x, num_conv=4, dropout=0.0, l=1., L=1.):
    for i in range(num_conv):
        residual = x
        x = LayerNormalization()(x)
        if i % 2 == 0:
            x = Dropout(dropout)(x)
        x = conv_layers[i](x)
        x = LayerDropout(dropout * (l / L))([x, residual])
    return x


def attention_block(attention_layer, x, seq_mask, dropout=0.0, l=1., L=1.):
    residual = x
    x = LayerNormalization()(x)
    x = Dropout(dropout)(x)
    x1 = attention_layer[0](x)
    x2 = attention_layer[1](x)
    x = attention_layer[2]([x1, x2, seq_mask])
    x = LayerDropout(dropout * (l / L))([x, residual])
    return x


def feed_forward_block(FeedForward_layers, x, dropout=0.0, l=1., L=1.):
    residual = x
    x = LayerNormalization()(x)
    x = Dropout(dropout)(x)
    x = FeedForward_layers[0](x)
    x = FeedForward_layers[1](x)
    x = LayerDropout(dropout * (l / L))([x, residual])
    return x


def QANet(config, word_mat=None, char_mat=None, cove_model=None):
    # parameters
    word_dim = config.word_dim
    char_dim = config.char_dim
    cont_limit = config.cont_limit
    char_limit = config.char_limit
    ans_limit = config.ans_limit
    filters = config.filters
    num_head = config.num_head
    dropout = config.dropout

    # Input Embedding Layer
    contw_input_ = Input((None,))
    quesw_input_ = Input((None,))
    contc_input_ = Input((None, char_limit))
    quesc_input_ = Input((None, char_limit))

    # get mask
    c_mask = Lambda(lambda x: tf.cast(x, tf.bool))(contw_input_)  # [bs, c_len]
    q_mask = Lambda(lambda x: tf.cast(x, tf.bool))(quesw_input_)
    cont_len = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x, tf.int32), axis=1), axis=1))(c_mask)
    ques_len = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x, tf.int32), axis=1), axis=1))(q_mask)

    # slice
    contw_input = BatchSlice(dim=2)([contw_input_, cont_len])
    quesw_input = BatchSlice(dim=2)([quesw_input_, ques_len])
    contc_input = BatchSlice(dim=3)([contc_input_, cont_len])
    quesc_input = BatchSlice(dim=3)([quesc_input_, ques_len])
    c_mask = BatchSlice(dim=2)([c_mask, cont_len])
    q_mask = BatchSlice(dim=2)([q_mask, ques_len])
    c_maxlen = tf.cast(tf.reduce_max(cont_len), tf.int32)
    q_maxlen = tf.cast(tf.reduce_max(ques_len), tf.int32)

    # embedding word
    WordEmbedding = Embedding(word_mat.shape[0], word_dim, weights=[word_mat], trainable=False, name='word_embedding')
    xw_cont = WordEmbedding(contw_input)
    xw_ques = WordEmbedding(quesw_input)

    # cove
    if cove_model is not None:
        x_cont_cove = cove_model(xw_cont)
        x_ques_cove = cove_model(xw_ques)
        xw_cont = Concatenate()([xw_cont, x_cont_cove])
        xw_ques = Concatenate()([xw_ques, x_ques_cove])

    # embedding char
    CharEmbedding = Embedding(char_mat.shape[0], char_dim, weights=[char_mat], name='char_embedding')
    xc_cont = CharEmbedding(contc_input)
    xc_ques = CharEmbedding(quesc_input)
    char_conv = Conv1D(filters, 5,
                       activation='relu',
                       kernel_initializer=init_relu,
                       kernel_regularizer=regularizer,
                       name='char_conv')
    xc_cont = Lambda(lambda x: tf.reshape(x, (-1, char_limit, char_dim)))(xc_cont)
    xc_ques = Lambda(lambda x: tf.reshape(x, (-1, char_limit, char_dim)))(xc_ques)
    xc_cont = char_conv(xc_cont)
    xc_ques = char_conv(xc_ques)
    xc_cont = GlobalMaxPooling1D()(xc_cont)
    xc_ques = GlobalMaxPooling1D()(xc_ques)
    xc_cont = Lambda(lambda x: tf.reshape(x, (-1, c_maxlen, filters)))(xc_cont)
    xc_ques = Lambda(lambda x: tf.reshape(x, (-1, q_maxlen, filters)))(xc_ques)

    # highwayNet
    x_cont = Concatenate()([xw_cont, xc_cont])
    x_ques = Concatenate()([xw_ques, xc_ques])

    # highway shared layers
    highway_layers = [Conv1D(filters, 1,
                             kernel_initializer=init,
                             kernel_regularizer=regularizer,
                             name='highway_input_projection')]
    for i in range(2):
        highway_layers.append(Conv1D(filters, 1,
                                     kernel_initializer=init,
                                     kernel_regularizer=regularizer,
                                     activation='sigmoid',
                                     name='highway' + str(i) + '_gate'))
        highway_layers.append(Conv1D(filters, 1,
                                     kernel_initializer=init,
                                     kernel_regularizer=regularizer,
                                     activation='linear',
                                     name='highway' + str(i) + '_linear'))
    x_cont = highway(highway_layers, x_cont, num_layers=2, dropout=dropout)
    x_ques = highway(highway_layers, x_ques, num_layers=2, dropout=dropout)

    # build shared layers
    # shared convs
    Encoder_DepthwiseConv1 = []
    for i in range(4):
        Encoder_DepthwiseConv1.append(DepthwiseConv1D(7, filters))

    # shared attention
    Encoder_SelfAttention1 = [Conv1D(2 * filters, 1,
                                     kernel_initializer=init,
                                     kernel_regularizer=regularizer),
                              Conv1D(filters, 1,
                                     kernel_initializer=init,
                                     kernel_regularizer=regularizer),
                              MultiHeadAttention(filters, num_head, dropout=dropout, bias=False)]
    # shared feed-forward
    Encoder_FeedForward1 = []
    Encoder_FeedForward1.append(Conv1D(filters, 1,
                                       kernel_initializer=init,
                                       kernel_regularizer=regularizer,
                                       activation='relu'))
    Encoder_FeedForward1.append(Conv1D(filters, 1,
                                       kernel_initializer=init,
                                       kernel_regularizer=regularizer,
                                       activation='linear'))

    # Context Embedding Encoder Layer
    x_cont = PositionEmbedding()(x_cont)
    x_cont = conv_block(Encoder_DepthwiseConv1, x_cont, 4, dropout)
    x_cont = attention_block(Encoder_SelfAttention1, x_cont, c_mask, dropout)
    x_cont = feed_forward_block(Encoder_FeedForward1, x_cont, dropout)

    # Question Embedding Encoder Layer
    x_ques = PositionEmbedding()(x_ques)
    x_ques = conv_block(Encoder_DepthwiseConv1, x_ques, 4, dropout)
    x_ques = attention_block(Encoder_SelfAttention1, x_ques, q_mask, dropout)
    x_ques = feed_forward_block(Encoder_FeedForward1, x_ques, dropout)

    # Context_to_Query_Attention_Layer
    x = context2query_attention(512, c_maxlen, q_maxlen, dropout)([x_cont, x_ques, c_mask, q_mask])
    x = Conv1D(filters, 1,
               kernel_initializer=init,
               kernel_regularizer=regularizer,
               activation='linear')(x)

    # Model_Encoder_Layer
    # shared layers
    Encoder_DepthwiseConv2 = []
    Encoder_SelfAttention2 = []
    Encoder_FeedForward2 = []
    for i in range(7):
        DepthwiseConv_share_2_temp = []
        for i in range(2):
            DepthwiseConv_share_2_temp.append(DepthwiseConv1D(5, filters))

        Encoder_DepthwiseConv2.append(DepthwiseConv_share_2_temp)
        Encoder_SelfAttention2.append([Conv1D(2 * filters, 1,
                                              kernel_initializer=init,
                                              kernel_regularizer=regularizer),
                                       Conv1D(filters, 1,
                                              kernel_initializer=init,
                                              kernel_regularizer=regularizer),
                                       MultiHeadAttention(filters, num_head, dropout=dropout, bias=False)])
        Encoder_FeedForward2.append([Conv1D(filters, 1,
                                            kernel_initializer=init,
                                            kernel_regularizer=regularizer,
                                            activation='relu'),
                                     Conv1D(filters, 1,
                                            kernel_initializer=init,
                                            kernel_regularizer=regularizer,
                                            activation='linear')])

    outputs = [x]
    for i in range(3):
        x = outputs[-1]
        for j in range(7):
            x = PositionEmbedding()(x)
            x = conv_block(Encoder_DepthwiseConv2[j], x, 2, dropout, l=j, L=7)
            x = attention_block(Encoder_SelfAttention2[j], x, c_mask, dropout, l=j, L=7)
            x = feed_forward_block(Encoder_FeedForward2[j], x, dropout, l=j, L=7)
        outputs.append(x)

    # Output_Layer
    x_start = Concatenate()([outputs[1], outputs[2]])
    x_start = Conv1D(1, 1,
                     kernel_initializer=init,
                     kernel_regularizer=regularizer,
                     activation='linear')(x_start)
    x_start = Lambda(lambda x: tf.squeeze(x, axis=-1))(x_start)
    x_start = Lambda(lambda x: mask_logits(x[0], x[1]))([x_start, c_mask])
    x_start = Lambda(lambda x: K.softmax(x), name='start')(x_start)  # [bs, len]

    x_end = Concatenate()([outputs[1], outputs[3]])
    x_end = Conv1D(1, 1,
                   kernel_initializer=init,
                   kernel_regularizer=regularizer,
                   activation='linear')(x_end)
    x_end = Lambda(lambda x: tf.squeeze(x, axis=-1))(x_end)
    x_end = Lambda(lambda x: mask_logits(x[0], x[1]))([x_end, c_mask])
    x_end = Lambda(lambda x: K.softmax(x), name='end')(x_end)  # [bs, len]

    x_start_fin, x_end_fin = QAoutputBlock(ans_limit, name='qa_output')([x_start, x_end])

    # if use model.fit, the output shape must be padded to the max length
    x_start = LabelPadding(cont_limit, name='start_pos')(x_start)
    x_end = LabelPadding(cont_limit, name='end_pos')(x_end)

    return Model(inputs=[contw_input_, quesw_input_, contc_input_, quesc_input_],
                 outputs=[x_start, x_end, x_start_fin, x_end_fin])
