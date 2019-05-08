#-*- coding: cp949 -*-
#-*- coding: utf-8 -*- 

# 저장된 데이터를 불러와 학습을 용이하게 하기 위해 정규화 시킨다.
# 21개 주가를 정규화 시켜 18개는 학습에, 3개는 평가에 사용한다.
# 정규화된 데이터를 이용하여 예측 모델을 학습시킨다.
# 매 epoch 마다 학습 및 평가하여 모델을 저장한다.
# 종가를 예측하는 회귀모델

import numpy as np
from TI_GRU_close import get_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.optimizers import *
from keras.models import Model
import os
import yaml
import keras.backend.tensorflow_backend as KK
import matplotlib.pyplot as plt

# gpu분할
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

def normalize_data(data, length):		# 데이터 범위를 0,1 사이로 변환시킨다. 
	min_max_scaler = MinMaxScaler()

	if length == 1:						#데이터가 1차원 일 경우
		n_data = min_max_scaler.fit_transform(data.reshape(-1,1))
		return n_data

	#데이터가 N차원 일 경우
	n_data =  min_max_scaler.fit_transform(data[:,0].reshape(-1,1))

	for i in range(length-1):
		i = i+1
		tmp = min_max_scaler.fit_transform(data[:,i].reshape(-1,1))
		n_data = np.concatenate((n_data,tmp),axis = 1)

	return n_data

#주가 데이터를 불러오고 정규화까지 시킨다.
def load_data(stock_name, time_step):		
	data_temp = np.load(stock_name+'_data.npy')
	x_temp = np.load(stock_name+'_x_label.npy')
	
	x_temp = normalize_data(x_temp[34:-1],12)			# 34번째 데이터 부터 데이터가 존재, 마지막데이터는 예측 할 수 없기에 뺀다
	close = normalize_data(data_temp[35+time_step:, 4],1)			# time_stpe이후 35번째 부터 예측 가능
	
	x_label = []

	for index in range(len(x_temp) - time_step): # 100개씩 타임스탭을 가진 x_label을 생성
		x_label.append(x_temp[index: index + time_step])

	return (np.array(x_label), np.array(close))

def save_graph(y1, y2, epoch, val_mae, save_dir = None):
	x = range(len(y1))
	plt.plot(x, y1, label = 'target')
	plt.plot(x, y2, label = 'predict')

	plt.xlabel('day')
	plt.ylabel('close')

	plt.title('%d_prediction val = %.5f'%(epoch,val_mae))
	plt.legend()

	if save_dir != None:
		plt.savefig(save_dir+'%d_prediction.png'%epoch, format='png')
		plt.clf() 
		return 0
	plt.show()

def predict_model(model, path, x_label, under, close_min ):
	model.load_weights(path)
	result = model.predict(x_val_label, batch_size=parser['batch_size'], verbose=1, steps=None)
	close_predict = (result*under)+close_min
	save_graph(x_label, close_predict, 325, 1808, )

def predict_model2(model, path, x_label, close_data):
	model.load_weights(path)
	result = model.predict(x_val_label, batch_size=parser['batch_size'], verbose=1, steps=None)
	close_predict = (result[:close_data[-3,-1]-1]*close_data[-3,1])+close_data[-3,0]
	tmp = (result[(close_data[-3,-1])-1:(close_data[-3,-1])+(close_data[-2,-1])-1]*close_data[-2,1])+close_data[-2,0]
	close_predict = np.append(close_predict,tmp)
	tmp = (result[(close_data[-3,-1])+(close_data[-2,-1])-1:]*close_data[-1,1])+close_data[-1,0]
	close_predict = np.append(close_predict,tmp)
	close_predict = close_predict[:,np.newaxis]	# 데이터를 1차원 늘린다.
	save_graph(x_label, close_predict, 325, 1808, )

def split_data(x_data,norm_close_data, non_norm_close_data):
	leng = int(len(x_data)*0.9)

	dev_x_label = x_data[0]
	dev_norm_close = norm_close_data[0]
	
	val_x_lable = x_data[leng]
	val_non_norm_close = non_norm_close_data[leng]

	for i in range(1,len(x_data)):
		if i < leng:

			dev_x_label = np.concatenate((dev_x_label,x_data[i]))
			dev_norm_close = np.concatenate((dev_norm_close,norm_close_data[i]))

		elif i > leng:
			val_x_lable = np.concatenate((val_x_lable,x_data[i]))
			val_non_norm_close = np.concatenate((val_non_norm_close,non_norm_close_data[i]))

	return(dev_x_label, dev_norm_close, val_x_lable, val_non_norm_close)

if __name__ == '__main__':
	#yaml파일 로드
	dir_yaml = '03_model_close_train.yaml'
	with open(dir_yaml, 'r', encoding='UTF8') as f_yaml:
		parser = yaml.load(f_yaml)

	filename=[]
	x_label = []
	normal_close = []
	non_norm_close = []
	close_data=[]

	for root, dirs, files in os.walk('./data'):
		for file in files:
			filename.append(file[:-5])

	for name in filename:
		data_temp = np.load('./data_label/'+name+'_data.npy')
		
		close_temp = []
	
		for i in data_temp[135:,4]:		# 34+100+1 -> 34개부터 기술적 지표가 모두 있음, timestep 100, 하루뒤 예측 1
			close_temp.append(float (i))

		close_temp = np.array(close_temp)

		close_min = min(close_temp)
		close_max = max(close_temp)
		
		under = close_max - close_min		# 추후 정규화된 결과를 종가로 다시 계산하기 위해
		close_data.append([int(close_min), int(under), int(len(close_temp))])

		close_temp = close_temp[:,np.newaxis]	# 데이터를 1차원 늘린다.
		non_norm_close.append(close_temp)

	non_norm_close = np.array(non_norm_close)
	close_data = np.array(close_data)

	for name in filename:
		x_temp, close_temp = load_data('./data_label/'+name, 100)
		x_label.append(x_temp)
		normal_close.append(close_temp)

	x_label = np.array(x_label)
	normal_close = np.array(normal_close)
	
	x_dev_label, dev_norm_close, x_val_label, val_non_norm_close = split_data(x_label, normal_close, non_norm_close)
	
	model, m_name = get_model(argDic = parser['model'])

	#predict_model2(model,'C:/Users/thwjd/source/stock_predict/networks/18_predict_model001/20-0.0272-5422.6158.h5'
	#		   ,val_non_norm_close,close_data)
	#exit()

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
		
		hist = model.fit(x_dev_label,dev_norm_close, batch_size = parser['batch_size'], epochs = 1, verbose=1)

		#학습된 모델로 validation_set을 평가한다.
		result = model.predict(x_val_label, batch_size=parser['batch_size'], verbose=1, steps=None)

		#평가 결과를 역정규화하여 종가 값으로 표현한다.
		close_predict = (result[:close_data[-3,-1]-1]*close_data[-3,1])+close_data[-3,0]
		tmp = (result[(close_data[-3,-1])-1:(close_data[-3,-1])+(close_data[-2,-1])-1]*close_data[-2,1])+close_data[-2,0]
		close_predict = np.append(close_predict,tmp)
		tmp = (result[(close_data[-3,-1])+(close_data[-2,-1])-1:]*close_data[-1,1])+close_data[-1,0]
		close_predict = np.append(close_predict,tmp)

		close_predict = close_predict[:,np.newaxis]	# 데이터를 1차원 늘린다.
		
		#오차는 validation_set과 평가 결과사이 차이의 절대값에 대한 평균으로 계산한다.
		val_mae = sum(abs(close_predict-val_non_norm_close))/len(close_predict)

		print('epoch: %d, val_mae: %f \n'%(int(epoch), val_mae))
		f_eer.write('epoch: %d, val_mae: %f \n'%(int(epoch), val_mae))

		#save_graph(non_norm_close[int(leng*0.9):], close_predict,epoch,val_mae ,save_dir)
		model.save_weights(save_dir + '%d-%.4f-%.4f.h5'%(epoch, hist.history['loss'][0], val_mae))
	f_eer.close()
		