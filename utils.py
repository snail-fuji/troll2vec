import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping, TensorBoard
from keras import optimizers
from keras import losses
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
from keras import initializers
from keras import backend as K

import numpy as np


# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def auc_roc(y_true, y_pred):
    y_pred = tf.maximum(tf.minimum(y_pred, 1), 0)
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

class CenterLossLayer(Layer):
    def __init__(self, alpha, classes, hidden, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.classes = classes
        self.hidden = hidden

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.classes, self.hidden),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)
        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
