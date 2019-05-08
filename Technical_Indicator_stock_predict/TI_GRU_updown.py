#-*- coding: cp949 -*-
#-*- coding: utf-8 -*- 

import keras
import tensorflow as tf
from keras import regularizers, optimizers, utils, models, initializers, constraints
from keras.layers import GRU, Dense, Activation, Input, Add, Dropout, LeakyReLU
from keras.models import Model
import os

r_value = 0.0001
_abspath = os.path.abspath(__file__)
m_name = _abspath.split('/')[-1].split('.')[0][-2:]
def get_model(argDic):
	inputs = Input(shape=(None,12))		#자료의 수, 순서열의 길이, x 벡터의 크기

	l = GRU(argDic['gru_units'], activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(r_value), recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0, recurrent_dropout=0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)(inputs)
	l = Dropout(argDic['dr'])(l)
	
	l = GRU(argDic['gru_units'], activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(r_value), recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0, recurrent_dropout=0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)(l)
	l = Dropout(argDic['dr'])(l)

	l = GRU(argDic['gru_units'], activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(r_value), recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0, recurrent_dropout=0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False, name = 't-vector')(l)
	#l = Dropout(argDic['dr'])(l)
	x = Dense(64, use_bias=True,  kernel_regularizer=regularizers.l2(r_value), name = 'code')(l)

	binary_out = Dense(2, activation = 'softmax', name='binary_out')(x)

	return [Model(inputs,binary_out), m_name]
	
