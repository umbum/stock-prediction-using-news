#-*- coding: cp949 -*-
#-*- coding: utf-8 -*- 


import numpy as np
import pickle
from BERT_CNN_updown import get_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.optimizers import *
from keras.models import Model
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os
import yaml
import keras.backend.tensorflow_backend as KK
import matplotlib.pyplot as plt


# gpu분할
def get_session(gpu_fraction=0.3):
	'''Assume that you have 8GB of GPU memory and want to allocate ~1.6GB'''

	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, allow_soft_placement=True, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))

KK.set_session(get_session())

def load_data(stock_name, time_step):		
	with open(stock_name, 'rb') as file:
		data_list = []
		while True:
			try:
				data = pickle.load(file)
			except EOFError:
				break
			data_list.append(data)

	date = []
	x_temp = []
	y_temp = []
	for i in range(len(data_list)):
		date.append(data_list[i]['day'])
		x_temp.append(data_list[i]['u-vector'])
		y_temp.append(data_list[i]['label'])
	
	date = np.array(date) 
	x_temp = np.array(x_temp)  #(2387,3072)
	y_temp = np.array(y_temp)  #(2387 ,)

	date = np.flip(date,0)	#과거 데이터 부터
	x_temp = np.flip(x_temp,0)	#과거 데이터 부터
	y_temp = np.flip(y_temp,0)	#과거 데이터 부터
	y_temp = y_temp.astype(np.int)
	


	
	return (np.array(x_temp[:-1]), np.array(y_temp[1:]),np.array(date[1:]))

def calculate_eer(model,data_x,data_y):
	y_score = []

	results = model.predict(data_x)
	
	#print(results)
	for i in range(len(results)):
		if results[i][1]>0.5:
			y_score.append(1)
		else:
			y_score.append(0)

	fpr, tpr, thresholds = roc_curve(data_y, y_score, pos_label=1)
	eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
	#print(data_y)
	#print(y_score)
	#print(sum(data_y))
	#print(sum(y_score))
	return eer

if __name__ == '__main__':
		#yaml파일 로드
	_abspath = os.path.abspath(__file__)
	dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
	with open(dir_yaml, 'r', encoding='UTF8') as f_yaml:
		parser = yaml.load(f_yaml)

	x_label, y_label, date = load_data('5avg_vec.pickle',100)
	x_label = x_label.reshape(x_label.shape[0], x_label.shape[1],1)

	print(x_label.shape) #(2286, 3072)
	
	#print(y_label.shape)
	#print(normal_close.shape)
	
	leng = len(x_label)			
	
	# development_set을 0.9 validation_set을 0.1로 맞춘다.
	x_dev_label = x_label[:int(leng*0.9),:,:]	
	x_val_label = x_label[int(leng*0.9):,:,:]
	y_dev_label = y_label[:int(leng*0.9)]	
	y_val_label = y_label[int(leng*0.9):]


	model, m_name = get_model(argDic = parser['model'])

	save_dir = parser['save_dir'] + parser['name'] + '/'

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

	
	model.compile(loss = parser['loss_function'],optimizer = optimizer,metrics = ['accuracy'])

	f_eer = open(save_dir + 'eers.txt', 'w', buffering=1)	# epoch 별 성능 기록할 텍스트 문서 생성

	#한번 epoch을 돌때마다 학습 및 평가하고 모델을 저정한다.
	for epoch in range(parser['epoch']):
		
		hist = model.fit(x_dev_label,y_dev_label, batch_size = parser['batch_size'], epochs = 1, verbose=1)

		val_eer = calculate_eer(model, x_val_label,y_val_label)
		

		print('epoch: %d, predict rate: %f \n'%(int(epoch), val_eer))
		f_eer.write('epoch: %d, val_eer: %f \n'%(int(epoch), val_eer))

		
		model.save_weights(save_dir + '%d-%.4f-%.4f.h5'%(epoch, hist.history['loss'][0], val_eer))
	f_eer.close()
		