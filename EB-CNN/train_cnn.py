from __future__ import print_function

import yaml
import json
import random
import numpy as np
import scipy.stats as stats
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import SGD
from keras.layers import Input, Conv1D, MaxPooling1D, add, Dense
from keras.models import Model, load_model
from tensorflow.contrib.data import sliding_window_batch


with open('EB-CNN.yaml', 'r', encoding='UTF8') as f_yaml:
    parser = yaml.load(f_yaml)
inp = parser['cnn_input']
oup = parser['cnn_output']
bs = parser['cnn_bs']
l= parser['cnn_window']
dd = parser['data_day']
ul = parser['long_term']      # U - Long-term
um = parser['mid_term']       # U - mid-term
us = parser['short_term']     # U - short-term


class ConvolutionalNN(Layer):

  def __init__(self, output_dim=oup, input_dim=inp, **kwargs):
    self.output_dim = output_dim #k
    self.input_dim = input_dim   #d
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(ConvolutionalNN, self).__init__(**kwargs)


  def build(self, input_shape):
    #print("output shape: ",self.compute_output_shape(input_shape))
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
    #print(self.bs," = bs")

    ult = inputs
    #print(ult,":ult")
    umt = K.tf.slice(inputs , [0,0,0,0], [self.bs,um,dd,d] )
    #print(umt, ":umt")
    ust =  K.tf.slice(inputs , [0,0,0,0], [self.bs,us,dd,d] )
    #print(ust, ":ust")


    ult = K.reshape(ult,(self.bs, ul, dd,d))
    ulong = []
    for i in range(dd):
      ulong.append(ult[:,:,i, :])
    #print("ulong:",ulong)

    ql_vec = Conv1D(d, self.l, activation='linear', padding='valid',data_format='channels_last', strides=1)(K.reshape(ulong[0][:,:,:], (self.bs, ul, d)))
    #print("ql_vec:",ql_vec)
    vl_vec = MaxPooling1D(pool_size=ul+1-self.l, strides=None, padding='valid',data_format='channels_last')(ql_vec)
    #print("vl_vec:",vl_vec)
    vl_vec = K.reshape(vl_vec, (self.bs, 1, 1,d))

    for i in range(1,dd):
      ql_vec = Conv1D(d, self.l, activation='linear', padding='valid',data_format='channels_last', strides=1)(K.reshape(ulong[i][:,:,:], (self.bs, ul, d)))
      #print("ql_vec:",ql_vec)
      vl_vect = MaxPooling1D(pool_size=ul+1 - self.l, strides=None, padding='valid', data_format='channels_last')(ql_vec)
      #print("vl_vect:", vl_vect)
      vl_vec = K.concatenate([vl_vec, K.reshape(vl_vect, (self.bs, 1, 1,d))],axis=1)
    #print("vl_vec:", vl_vec)

    umt = K.reshape(umt, (self.bs,um,dd, d))
    umid = []
    for i in range(dd):
      umid.append(umt[:,:,i, :])
    #print("umid:",umid)

    qm_vec = Conv1D(d, self.l, activation='linear', padding='valid',data_format='channels_last', strides=1)(
      K.reshape(umid[0][:,:,:], (self.bs, um, d)))
    #print("qm_vec:",qm_vec)
    vm_vec = MaxPooling1D(pool_size=um+1 - self.l, strides=None, padding='valid', data_format='channels_last')(qm_vec)
    #print("vm_vec:",vm_vec)
    vm_vec = K.reshape(vm_vec, (self.bs, 1, 1, d))

    for i in range(1, dd):
      qm_vec = Conv1D(d, self.l, activation='linear', padding='valid',data_format='channels_last', strides=1)(
        K.reshape(umid[i][:,:,:], (self.bs, um, d)))
      #print("qm_vec:",qm_vec)
      vm_vect = MaxPooling1D(pool_size=um+1 - self.l, strides=None, padding='valid', data_format='channels_last')(qm_vec)
      #print("vm_vect:", vm_vect)
      vm_vec = K.concatenate([vm_vec, K.reshape(vm_vect, (self.bs, 1, 1,d))], axis=1)
    #print("vm_vec:", vm_vec)


    vs_vec = K.reshape(ust,(self.bs, dd, 1, d))
    #print("vs_vec:",vs_vec)

    vc_vec = K.concatenate([K.concatenate([vl_vec,vm_vec],axis=2)
                            , vs_vec], axis=2)

    #print("vc_vec:", vc_vec)

    Y = Dense( 5, activation='sigmoid')(vc_vec)
    ycls = Dense( 2, activation='sigmoid')(Y)
    #print(ycls,":ycls")
    #ycls = K.print_tensor(ycls, message="Value of ycls")
    return ycls


  def compute_output_shape(self, input_shape):
    # print (input_shape)
    batch_size = input_shape[0]
    return (batch_size,  dd, 3  , 2)

def binary_classification_loss(y_true, y_pre):
  tp = K.binary_crossentropy(y_true, y_pre)
  tp = K.print_tensor(tp, message="Value of tp")
  temp = K.mean(K.binary_crossentropy(y_true, y_pre),(0,1,2,3))
  #temp = K.print_tensor(temp, message="Value of temp")
  return temp


def main():

  with open(parser['u_vector_path'], 'rb') as f:
    json_u = json.load(f)

  u_vec = np.asarray(json.loads(json_u))  # ('cnn_data_num','cnn_input')
  #print(u_vec,":u_vec")

  m = len(u_vec)//ul

  ln = m * ul -ul+1 # subtract 1 to make spare batch
  ln = ln - divmod(ln, bs)[1] +ul-1 # make multiple of batch size

  lnt = (m //4 )*ul -ul+1
  lnt = lnt - divmod(lnt, bs)[1] +ul-1# make multiple of batch size
  #print(np.array(u_vec[:ln, :, :].shape))
  input1 = Input(shape=(ul,dd,inp,), batch_shape=(bs,ul,dd,inp) ,dtype='float32')

  cnn= ConvolutionalNN(output_dim=oup, input_dim=inp)(input1)
  model = Model(inputs=[input1], outputs=[cnn])

  sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss=binary_classification_loss, optimizer=sgd)

  Y_train = np.array([random.randint(0,1) for i in range(3*ln*dd)])
  Y_train = np.reshape(Y_train, (ln, dd,3, 1))
  Y_t = Y_train*(-1)+1
  Y_train = np.concatenate((Y_train,Y_t),axis=3)
  Y_train =Y_train.astype(np.float32)
  #print(":Y_train_",Y_train.shape)


  dataset  =  tf.data.Dataset.from_tensor_slices((u_vec[:lnt,:,:]))
  dataset = dataset.apply(sliding_window_batch(window_size=ul))

  iterator = dataset.make_one_shot_iterator()  # iterator
  next_element = iterator.get_next()
  numpy_data = np.zeros((lnt-ul+1,ul,dd,inp))
  numpy_label = Y_train[:lnt-ul+1,:,:,:]
  sess = tf.Session()  # tensorflow session
  for i in range(lnt-ul+1):
    data_= sess.run(
      next_element)  # data_ contains the data and label_ contains the labels that we fed in the previous step
    numpy_data[i,:,:,:] = data_

  model.fit(numpy_data,numpy_label, epochs=parser['cnn_epoch'], batch_size=bs)

  dataset  =  tf.data.Dataset.from_tensor_slices((u_vec[:ln,:,:]))
  dataset = dataset.apply(sliding_window_batch(window_size=ul))

  iterator = dataset.make_one_shot_iterator()  # iterator
  next_element = iterator.get_next()
  numpy_data = np.zeros((ln-ul+1,ul,dd,inp))
  numpy_label = Y_train[:ln-ul+1,:,:,:]
  sess = tf.Session()  # tensorflow session
  for i in range(ln-ul+1):
    data_= sess.run(
      next_element)  # data_ contains the data and label_ contains the labels that we fed in the previous step
    numpy_data[i,:,:,:] = data_

  score = model.evaluate(numpy_data, numpy_label, batch_size=bs)
  print(score,":score")

  model.save(parser['cnn_mod_name'])
  print("Save model: %s"%(parser['cnn_mod_name']))




if __name__ == "__main__" :
    main()