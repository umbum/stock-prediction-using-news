#!/usr/bin/python
#-*- coding: cp949 -*-
#-*- coding: utf-8 -*- 

import numpy as np
from TI_GRU import get_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.optimizers import *
from keras.models import Model
import os
import yaml
import keras.backend.tensorflow_backend as KK

def get_session(gpu_fraction=0.1):
	'''Assume that you have 8GB of GPU memory and want to allocate ~1.6GB'''
 
	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
 
	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, allow_soft_placement=True, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))
 
KK.set_session(get_session())

def normalize_data(data, length):
	min_max_scaler = MinMaxScaler()

	if length == 1:
		n_data = min_max_scaler.fit_transform(data.reshape(-1,1))
		return n_data

	n_data =  min_max_scaler.fit_transform(data[:,0].reshape(-1,1))

	for i in range(length-1):
		i = i+1
		tmp = min_max_scaler.fit_transform(data[:,i].reshape(-1,1))
		n_data = np.concatenate((n_data,tmp),axis = 1)

	return n_data

def load_data(stock_name, time_step):
	data_temp = np.load(stock_name+'_data.npy')
	x_temp = np.load(stock_name+'_x_label.npy')
	y_temp = np.load(stock_name+'_y_label.npy')
	
	x_temp = normalize_data(x_temp[34:],12)
	close = normalize_data(data_temp[34:,4],1)

	
	x_label = []
	y_label = []

	for index in range(len(x_temp) - time_step): 
		x_label.append(x_temp[index: index + time_step])

	for y in y_temp[34+time_step:]:		
		
		y = int(y)
		if y >0:
			y_label.append([1,0])
		elif y == 0:
			y_label.append([0,1])

	return (np.array(x_label), np.array(y_label),np.array(close[time_step:]))

if __name__ == '__main__':
	#yaml파일 로드
	_abspath = os.path.abspath(__file__)
	dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
	with open(dir_yaml, 'r', encoding='UTF8') as f_yaml:
		parser = yaml.load(f_yaml)
	
	data_temp = np.load('sk_data.npy')
	close_temp = []
	for i in data_temp[134:,4]:
		close_temp.append(float (i))
	
	close_temp = np.array(close_temp)
	close_temp = close_temp[:,np.newaxis]
	
	close_min = min(close_temp)
	close_max = max(close_temp)
	
	under = close_max - close_min
	
	close = normalize_data(data_temp[34:,4],1)
	
	
	x_label, y_label, close = load_data('sk',100)

	leng = len(x_label)
	
	x_dev_label = x_label[:int(leng*0.9),:,:]
	x_val_label = x_label[int(leng*0.9):,:,:]
	close_dev = close[:int(leng*0.9),:]
	close_val = close[int(leng*0.9):,:]

	model, m_name = get_model(argDic = parser['model'])

	save_dir = os.path.split(_abspath)[0]+'/'+ parser['name'] + '/'

	#실험 결과 저장 폴더(들) 생성
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	#현재 실험 hyper-parameter들을 텍스트 파일에 저장
	f_params = open(save_dir + 'f_params.txt', 'w')
	for k, v in parser.items():
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.write('DNN model params\n')
	for k, v in parser['model'].items():
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.write('model_name: %s\n'%m_name)
	f_params.close()

	with open(save_dir + 'summary.txt' ,'w+') as f_summary:
		model.summary(print_fn=lambda x: f_summary.write(x + '\n'))

	#모델 optimizer 및 objective funciton 정의
	if parser['optimizer'] == 'SGD':
		optimizer = eval(parser['optimizer'])(lr=parser['lr'],  decay = parser['opt_decay'])
	elif parser['optimizer'] == 'Adam':
		optimizer = eval(parser['optimizer'])(lr=parser['lr'], decay = parser['opt_decay'], amsgrad = bool(parser['amsgrad']))
	elif parser['optimizer'] == 'RMSprop':
		optimizer = eval(parser['optimizer'])(lr=parser['lr'], decay = parser['opt_decay'])

	#model.compile(loss='categorical_crossentropy',optimizer='adam')
	model.compile(loss = parser['loss_function'],optimizer = optimizer,metrics = ['accuracy'])

	f_eer = open(save_dir + 'eers.txt', 'w', buffering=1)	# epoch 별 성능 기록할 텍스트 문서 생성

	for epoch in range(parser['epoch']):
		
		#model.fit(x_dev_label,close_temp[:int(leng*0.9),:], batch_size = 100, epochs = 1, verbose=1)
		hist = model.fit(x_dev_label,close_dev, batch_size = parser['batch_size'], epochs = 1, verbose=1)

		result = model.predict(x_val_label, batch_size=parser['batch_size'], verbose=1, steps=None)

		close_predict = (result*under)+close_min
		
		#val_mae = sum(abs(result-close_temp[int(leng*0.9):,:]))/len(result)
		val_mae = sum(abs(close_predict-close_temp[int(leng*0.9):]))/len(close_predict)

		print('epoch: %d, val_mae: %f \n'%(int(epoch), val_mae))
		f_eer.write('epoch: %d, val_mae: %f \n'%(int(epoch), val_mae))

		#model.save_weights(save_dir + '%d-%.4f-%.4f.h5'%(epoch, hist.history['loss'][0], val_mae))
	f_eer.close()
		