"""
Prediction of the sequence-specific cleavage activity of Cas9 variants

conda create -n astroboi_tf_2 python=3.6 tensorflow=2.1.0 h5py=2.10.0
conda activate astroboi_tf_2

conda install -c anaconda pandas=1.1.3 xlrd=1.2.0 pydot=1.4.1 pydotplus=2.0.2 scikit-learn=0.23.2
conda install -c conda-forge matplotlib=3.3.3

conda install -c anaconda pandas
conda install -c anaconda xlrd
conda install -c conda-forge matplotlib
conda install -c anaconda pydot
conda install -c anaconda pydotplus
conda install -c anaconda scikit-learn

with CUDA
CUDA 10.1
cudnn 7.6.0
conda activate astroboi_cuda_2
"""
# import os
# import sys
import numpy as np
# from time import time
# from scipy import stats
# from sklearn import metrics
import random as py_random

import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# to obtain reproducible results
np.random.seed(1)
py_random.seed(1)
tf.random.set_seed(1)


class DeepLayers:
    def __init__(self, kernel_init='he_normal', bias_init='zeros'):
        self.kernel_init = kernel_init
        self.bias_init = bias_init

    def conv_bn_act_drp(self, x, opt_dict, l_name):
        __x = layers.Conv2D(opt_dict["knl_num"], opt_dict["filt_shp"], strides=opt_dict["strd"],
                            padding=opt_dict["pad"], kernel_initializer=self.kernel_init,
                            bias_initializer=self.bias_init, name=l_name)(x)
        # use BatchNormalization
        if opt_dict["bn_flag"]:
            __x = layers.BatchNormalization()(__x)
        __x = layers.Activation(opt_dict['acti'])(__x)
        # no dropout if dropout_rate is 0.0
        if opt_dict["drp"] != 0.0:
            __x = layers.Dropout(opt_dict["drp"])(__x)
        return __x, x

    def conv_act(self, x, opt_dict, l_name):
        __x = layers.Conv2D(opt_dict["knl_num"], opt_dict["filt_shp"], strides=opt_dict["strd"],
                            padding=opt_dict["pad"], kernel_initializer=self.kernel_init,
                            bias_initializer=self.bias_init, activation=opt_dict['acti'], name=l_name)(x)
        return __x, x

    def dense_bn_act_drp(self, x, opt_dict, l_name):
        x = layers.Dense(opt_dict["w_num"], kernel_initializer=self.kernel_init, bias_initializer=self.bias_init,
                         name=l_name)(x)
        # use BatchNormalization
        if opt_dict["bn_flag"]:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(opt_dict['acti'])(x)
        # no dropout if dropout_rate is 0.0
        if opt_dict["drp"] != 0.0:
            x = layers.Dropout(opt_dict["drp"])(x)
        return x

class DeepModels:

    def __init__(self):
        self.main_input = layers.Input(shape=(4, 30, 1), name='main_input')
        self.feature_input = layers.Input(shape=(1,), name='feature_input')

    def fine_tune_model(self, x, opts_dict):
        deep_layers = DeepLayers()
        x = deep_layers.dense_bn_act_drp(x, opts_dict["fc_1"], "fc_1")

        x = deep_layers.dense_bn_act_drp(x, opts_dict["feat"], "feat_0")

        f = deep_layers.dense_bn_act_drp(self.feature_input, opts_dict["feat"], "feat")
        concated = layers.concatenate([x, f])
        xx = deep_layers.dense_bn_act_drp(concated, opts_dict["last"], "last")
        y = layers.Dense(1, name='output')(xx)
        return y

    def model_1cnv_incptn_w_mx(self, opts_dict):
        deep_layers = DeepLayers()

        x1, _ = deep_layers.conv_bn_act_drp(self.main_input, opts_dict["conv1"], "conv1")

        x2, _ = deep_layers.conv_bn_act_drp(x1, opts_dict["conv2"], "conv2")
        x2, _ = deep_layers.conv_bn_act_drp(x2, opts_dict["conv2_1"], "conv2_1")
        x2, _ = deep_layers.conv_bn_act_drp(x2, opts_dict["conv2_1"], "conv2_2")
        x2, _ = deep_layers.conv_bn_act_drp(x2, opts_dict["conv2_1"], "conv2_3")

        x3, _ = deep_layers.conv_bn_act_drp(x1, opts_dict["conv3"], "conv3")
        x3, _ = deep_layers.conv_bn_act_drp(x3, opts_dict["conv3_1"], "conv3_1")

        x5, _ = deep_layers.conv_bn_act_drp(x1, opts_dict["conv5"], "conv5")

        mx_pool = layers.MaxPool2D(pool_size=opts_dict["conv5"]["filt_shp"], strides=opts_dict["conv5"]["strd"],
                                   padding=opts_dict["conv5"]["pad"], name='mx_pool_incp')(x1)

        x_concat = layers.concatenate([x2, x3, x5, mx_pool], axis=3)
        x_concat, _ = deep_layers.conv_bn_act_drp(x_concat, opts_dict["conv_concate"],"conv_concate")

        x = layers.Flatten()(x_concat)
        x = deep_layers.dense_bn_act_drp(x, opts_dict["fc_1"], "fc_1")

        x = deep_layers.dense_bn_act_drp(x, opts_dict["feat"], "feat_0")

        f = deep_layers.dense_bn_act_drp(self.feature_input, opts_dict["feat"], "feat")
        concated = layers.concatenate([x, f])
        xx = deep_layers.dense_bn_act_drp(concated, opts_dict["last"], "last")
        y = layers.Dense(1, name='output')(xx)
        return y

    def model_1cnv_incptn_wo_mx(self, opts_dict):
        deep_layers = DeepLayers()

        x1, _ = deep_layers.conv_bn_act_drp(self.main_input, opts_dict["conv1"], "conv1")

        x2, _ = deep_layers.conv_bn_act_drp(x1, opts_dict["conv2"], "conv2")
        x2, _ = deep_layers.conv_bn_act_drp(x2, opts_dict["conv2_1"], "conv2_1")
        x2, _ = deep_layers.conv_bn_act_drp(x2, opts_dict["conv2_1"], "conv2_2")
        x2, _ = deep_layers.conv_bn_act_drp(x2, opts_dict["conv2_1"], "conv2_3")

        x3, _ = deep_layers.conv_bn_act_drp(x1, opts_dict["conv3"], "conv3")
        x3, _ = deep_layers.conv_bn_act_drp(x3, opts_dict["conv3_1"], "conv3_1")

        x5, _ = deep_layers.conv_bn_act_drp(x1, opts_dict["conv5"], "conv5")

        x_concat = layers.concatenate([x2, x3, x5], axis=3)
        x_concat, _ = deep_layers.conv_bn_act_drp(x_concat, opts_dict["conv_concate"], "conv_concate")

        x = layers.Flatten()(x_concat)
        x = deep_layers.dense_bn_act_drp(x, opts_dict["fc_1"], "fc_1")

        x = deep_layers.dense_bn_act_drp(x, opts_dict["feat"], "feat_0")

        f = deep_layers.dense_bn_act_drp(self.feature_input, opts_dict["feat"], "feat")
        concated = layers.concatenate([x, f])
        xx = deep_layers.dense_bn_act_drp(concated, opts_dict["last"], "last")
        y = layers.Dense(1, name='output')(xx)
        return y

    def model_legacy_1cnv(self, opts_dict):
        deep_layers = DeepLayers()

        x, _ = deep_layers.conv_bn_act_drp(self.main_input, opts_dict["conv1"], "conv1")
        x = layers.Flatten(name='flatten')(x)
        x = deep_layers.dense_bn_act_drp(x, opts_dict["fc_1"], "fc_1_l")

        x = deep_layers.dense_bn_act_drp(x, opts_dict["feat"], "fc_2_1")

        f = deep_layers.dense_bn_act_drp(self.feature_input, opts_dict["feat"], "feat_l")
        multiplied = layers.Multiply(name='multiple')([x, f])
        xx = deep_layers.dense_bn_act_drp(multiplied, opts_dict["last"], "last_l")
        y = layers.Dense(1, name='output')(xx)
        return y
