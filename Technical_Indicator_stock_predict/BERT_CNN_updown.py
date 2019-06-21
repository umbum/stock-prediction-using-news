#-*- coding: cp949 -*-
#-*- coding: utf-8 -*- 

import keras
import tensorflow as tf
from keras import regularizers, optimizers, utils, models, initializers, constraints
from keras.layers import GRU, Dense, Activation, Input, Add, Dropout, LeakyReLU,Conv1D,BatchNormalization,GlobalMaxPooling1D,Flatten
from keras.models import Model
import os

r_value = 0.00
_abspath = os.path.abspath(__file__)
m_name = _abspath.split('/')[-1].split('.')[0][-2:]
def get_model(argDic):
	inputs = Input(shape=(None,1))		#자료의 수, 순서열의 길이, x 벡터의 크기

	#l = Dropout(0.2)(inputs)
	l = Conv1D(64,3,strides=1, padding = 'valid', activation  = 'relu', kernel_regularizer=regularizers.l2(r_value))(inputs)
	l = BatchNormalization()(l)
	l = LeakyReLU()(l)
	l = Dropout(0.2)(l)

	l = Conv1D(64,3,strides=1, padding = 'valid', activation  = 'relu', kernel_regularizer=regularizers.l2(r_value))(l)
	l = BatchNormalization()(l)
	l = LeakyReLU()(l)
	l = Dropout(0.2)(l)

	l = Conv1D(64,3,strides=1, padding = 'valid', activation  = 'relu', kernel_regularizer=regularizers.l2(r_value))(l)
	l = BatchNormalization()(l)
	l = LeakyReLU()(l)

	l = GlobalMaxPooling1D()(l)
	#x = Dense(, use_bias=True,  kernel_regularizer=regularizers.l2(r_value), name = 'code')(l)
	#x = Dense(50, use_bias=True,  kernel_regularizer=regularizers.l2(r_value), name = 'code')(l)

	binary_out = Dense(2, activation = 'softmax', name='binary_out')(l)

	return [Model(inputs,binary_out), m_name]
	
