# changed from https://github.com/alno/kaggle-allstate-claims-severity/blob/master/keras_util.py by ewrfcas

from keras import backend as K
from tqdm import tqdm


def ExponentialMovingAverage_TrainBegin(model):
    # run when training begins
    # ema_trainable_weights_vals save the latest weights of model with ema
    ema_trainable_weights_vals = {}
    for weight in tqdm(model.trainable_weights):
        ema_trainable_weights_vals[weight.name] = K.get_value(weight)
    return ema_trainable_weights_vals


def ExponentialMovingAverage_BatchEnd(model, ema_trainable_weights_vals, decay=0.999):
    # run when each batch ends
    for weight in model.trainable_weights:
        old_val = ema_trainable_weights_vals[weight.name]
        ema_trainable_weights_vals[weight.name] = decay * old_val + (1.0 - decay) * K.get_value(weight)
    return ema_trainable_weights_vals


def ExponentialMovingAverage_EpochEnd(model, ema_trainable_weights_vals):
    # run when each epoch ends, generate model with ema to evaluating
    for weight in tqdm(model.trainable_weights):
        K.set_value(weight, ema_trainable_weights_vals[weight.name])
