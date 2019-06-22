"""
CNN 신경망 모델을 학습시키는 모듈

5avg_vec.pickle 파일이 같은 디렉토리 안에 있어야함.
"""


from __future__ import print_function

import yaml
import json
import random
import pickle
import numpy as np
import scipy.stats as stats
import tensorflow as tf
import codecs, json
import os, sys

sys.path.append("../data")
from news_labeling_util import labelingNewsIterArr

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.engine.topology import Layer
from keras import optimizers
from keras.layers import Input, Conv1D, MaxPooling1D, add, Dense
from keras.models import Model, load_model
from tensorflow.contrib.data import sliding_window_batch

with open('EB-CNN.yaml', 'r', encoding='UTF8') as f_yaml:
    parser = yaml.load(f_yaml)
inp = parser['cnn_input']
oup = parser['cnn_output']
bs = parser['cnn_bs']
l = parser['cnn_window']
dd = parser['data_day']
ul = parser['long_term']  # U - Long-term
um = parser['mid_term']  # U - mid-term
us = parser['short_term']  # U - short-term


class ConvolutionalNN(Layer):

    def __init__(self, output_dim=oup, input_dim=inp, **kwargs):
        self.output_dim = output_dim  # k
        self.input_dim = input_dim  # d
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ConvolutionalNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # print("output shape: ",self.compute_output_shape(input_shape))
        mean = 0.0
        std = 1.0
        # T : k*d*d
        k = self.output_dim
        d = self.input_dim
        self.l = l

    def call(self, inputs, mask=None):
        d = self.input_dim
        batch_size = K.shape(inputs)[0]
        self.bs = np.int(K.get_value(batch_size))
        # print(self.bs," = bs")

        ult = inputs
        # print(ult,":ult")
        umt = K.tf.slice(inputs, [0, 0, 0], [self.bs, um, d])
        # print(umt, ":umt")
        ust = K.tf.slice(inputs, [0, 0, 0], [self.bs, us, d])
        # print(ust, ":ust")

        # ult = K.reshape(ult,(self.bs, ul, d))

        ql_vec = Conv1D(d, self.l, activation='linear', padding='valid', data_format='channels_last', strides=1)(ult)
        # print("ql_vec:",ql_vec)
        vl_vec = MaxPooling1D(pool_size=ul + 1 - self.l, strides=None, padding='valid', data_format='channels_last')(
            ql_vec)
        # print("vl_vec:",vl_vec)
        vl_vec = K.reshape(vl_vec, (self.bs, d))
        # print("vl_vec:", vl_vec)

        # umt = K.reshape(umt, (self.bs,um, d))

        qm_vec = Conv1D(d, self.l, activation='linear', padding='valid', data_format='channels_last', strides=1)(umt)
        # print("qm_vec:",qm_vec)
        vm_vec = MaxPooling1D(pool_size=um + 1 - self.l, strides=None, padding='valid', data_format='channels_last')(
            qm_vec)
        # print("vm_vec:",vm_vec)
        vm_vec = K.reshape(vm_vec, (self.bs, d))
        # print("vm_vec:", vm_vec)
        vs_vec = K.reshape(ust, (self.bs, d))
        # print("vs_vec:",vs_vec)

        vc_vec = K.concatenate([K.concatenate([vl_vec, vm_vec], axis=1)
                                   , vs_vec], axis=1)

        # print("vc_vec:", vc_vec)
        rs = K.sum(vc_vec)
        rs = K.print_tensor(rs, message="rs")
        vc_vec = K.print_tensor(vc_vec, message="vc_vec")
        vc_vec2 = Dense(5000, activation='sigmoid')(vc_vec)
        vc_vec2 = K.print_tensor(vc_vec2, message="vc_vec2")
        vc_vec3 = Dense(2500, activation='sigmoid')(vc_vec2)
        vc_vec3 = K.print_tensor(vc_vec3, message="vc_vec3")
        vc_vec4 = Dense(1000, activation='sigmoid')(vc_vec3)
        vc_vec4 = K.print_tensor(vc_vec4, message="vc_vec4")
        vc_vec5 = Dense(250, activation='sigmoid')(vc_vec4)
        vc_vec5 = K.print_tensor(vc_vec5, message="vc_vec5")
        Y = Dense(10, activation='sigmoid')(vc_vec5)
        Y = K.print_tensor(Y, message="Y")
        ycls = Dense(2, activation='softmax')(Y)
        # print(ycls,":ycls")
        ycls = K.print_tensor(ycls, message="Value of ycls")
        return ycls

    def compute_output_shape(self, input_shape):
        # print (input_shape)
        batch_size = input_shape[0]
        return (batch_size, 2)


def binary_classification_loss(y_true, y_pre):
    tp = K.categorical_crossentropy(y_true, y_pre)
    # tp = K.print_tensor(tp, message="Value of tp")
    temp = K.tf.reduce_mean(tp)
    temp = K.print_tensor(temp, message="Value of temp")
    return temp


def main():
    """
  with open('labelled.json', encoding='utf-8') as f:
      json_labelled = json.load(f)

  labelled = np.asarray(json.loads(json_labelled))
  w_day_avg = []
  u_vec_avg = []
  label_avg = []
  for i in range(len(labelled)):
      w_day_avg.append(labelled[i][0])
      u_vec_avg.append(labelled[i][1])
      label_avg.append(labelled[i][2])

  w_day_avg = np.array(w_day_avg)
  u_vec_avg = np.array(u_vec_avg)
  label_avg = np.array(label_avg)
  """

    w_day_avg = []
    u_vec_avg = []
    label_avg = []

    with open('./5avg_vec.pickle', 'rb') as f:
        while True:
            try:
                pic = pickle.load(f)
                w_day_avg.append(pic['day'])
                u_vec_avg.append(pic['u-vector'])
                label_avg.append(pic['label'])
                print(pic['day'])
            except (EOFError):
                break
    w_day_avg = np.asarray(w_day_avg)
    u_vec_avg = np.array(u_vec_avg, dtype=np.int32)
    label_avg = np.array(label_avg, dtype=np.int32)

    m = len(u_vec_avg) // ul

    ln = m * ul - ul + 1  # subtract 1 to make spare batch
    ln = ln - divmod(ln, bs)[1] + ul - 1  # make multiple of batch size

    lnt = ((m // 5) * 4) * ul - ul + 1
    lnt = lnt - divmod(lnt, bs)[1] + ul - 1  # make multiple of batch size

    m = (len(u_vec_avg) - (lnt - ul + 1)) // ul
    df = m * ul - ul + 1
    df = df - divmod(df, bs)[1] + ul - 1

    # print(np.array(u_vec[:ln, :, :].shape))
    input1 = Input(shape=(ul, inp,), batch_shape=(bs, ul, inp), dtype='float32')

    cnn = ConvolutionalNN(output_dim=oup, input_dim=inp)(input1)
    model = Model(inputs=[input1], outputs=[cnn])

    #     sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss=binary_classification_loss, optimizer=adam, metrics=[binary_classification_loss, 'accuracy'])

    # Y_train = np.array([random.randint(0,1) for i in range(3*ln)])
    # Y_train = np.reshape(Y_train, (ln, 3, 1))
    # Y_t = Y_train*(-1)+1
    # Y_train = np.concatenate((Y_train,Y_t),axis=3)
    # Y_train =Y_train.astype(np.float32)
    # print(":Y_train_",Y_train.shape)

    dataset = tf.data.Dataset.from_tensor_slices((np.array(u_vec_avg[:lnt][:])))
    dataset = dataset.apply(sliding_window_batch(window_size=ul))

    iterator = dataset.make_one_shot_iterator()  # iterator
    next_element = iterator.get_next()

    numpy_data = np.zeros((lnt - ul + 1, ul, inp))
    numpy_label = np.concatenate((np.array(label_avg[:lnt - ul + 1]).reshape(lnt - ul + 1, 1),
                                  1 - np.array(label_avg[:lnt - ul + 1]).reshape(lnt - ul + 1, 1)), axis=1)
    #     numpy_data = np.zeros((10,ul, inp))
    #     numpy_label = np.concatenate((np.array(label_avg[:10]).reshape(10, 1),
    #                                   1 - np.array(label_avg[:10]).reshape(10, 1)), axis=1)

    sess = tf.Session()  # tensorflow session
    for i in range(lnt - ul + 1):
        #     for i in range(10):
        data_ = sess.run(
            next_element)  # data_ contains the data and label_ contains the labels that we fed in the previous step
        numpy_data[i, :, :] = data_

    MODEL_SAVE_FOLDER_PATH = './modelCNN/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{binary_classification_loss:.4f}.hdf5'
    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='binary_classification_loss',
                                    verbose=1, save_best_only=True)
    cb_early_stopping = EarlyStopping(monitor='binary_classification_loss', patience=100)

    model.fit(numpy_data, numpy_label, epochs=parser['cnn_epoch'], batch_size=bs,
              callbacks=[cb_early_stopping, cb_checkpoint])

    dataset = tf.data.Dataset.from_tensor_slices((np.array(u_vec_avg[lnt - ul + 1:lnt - ul + 1 + df][:])))
    dataset = dataset.apply(sliding_window_batch(window_size=ul))

    iterator = dataset.make_one_shot_iterator()  # iterator
    next_element = iterator.get_next()
    numpy_data = np.zeros((df - ul + 1, ul, inp))
    numpy_label = np.concatenate((np.array(label_avg[lnt - ul + 1:lnt - ul + 1 + df - ul + 1]).reshape(df - ul + 1, 1),
                                  1 - np.array(label_avg[lnt - ul + 1:lnt - ul + 1 + df - ul + 1]).reshape(df - ul + 1,
                                                                                                           1)), axis=1)
    sess = tf.Session()  # tensorflow session
    for i in range(df - ul + 1):
        data_ = sess.run(
            next_element)  # data_ contains the data and label_ contains the labels that we fed in the previous step
        numpy_data[i, :, :] = data_

    score = model.evaluate(numpy_data, numpy_label, batch_size=bs)
    print(score[1], ":loss ", score[2], ":accuracy")

    model.save(parser['cnn_mod_name'])
    print("Save model: %s" % (parser['cnn_mod_name']))


if __name__ == "__main__":
    main()