from keras import backend as K
from keras.losses import categorical_crossentropy
import tensorflow as tf

def w_categorical_crossentropy(y_true, y_pred):
    weight = [1.0000, 3.0687, 3.1699, 87.9267, 6.1566]
    weight = K.constant(value=weight, dtype='float32')
    weight = tf.reduce_sum(tf.multiply(weight, y_true))
    loss = categorical_crossentropy(y_true, y_pred)
    w_loss = tf.multiply(weight, loss)
    return w_loss
