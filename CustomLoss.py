import tensorflow as tf
from tensorflow import keras


class CustomLosses:

    def __init__(self):
        pass

    def pearson_loss(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = keras.backend.mean(x)
        my = keras.backend.mean(y)
        xm, ym = x - mx, y - my
        r_num = keras.backend.sum(tf.multiply(xm, ym))
        r_den = keras.backend.sqrt(tf.multiply(keras.backend.sum(keras.backend.square(xm)),
                                               keras.backend.sum(keras.backend.square(ym)))) + 1e-12
        r = r_num / r_den
        r = keras.backend.maximum(keras.backend.minimum(r, 1.0), -1.0)
        return 1 - r
