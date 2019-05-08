#-*- coding: cp949 -*-
#-*- coding: utf-8 -*- 

#	data 폴더에 저장된 엑셀들을 불러온다.
#	일자, 시가, 고가, 저가, 종가, 거래량은 주가이름_data로 저장
#	MACD, EMA, Momentum, SlowK, SlowD, ROC, William_R, A/D, RSI, OBV, upperband, lowerband는 기술적 지표로서 학습에 사용될 주가이름_X_label로 저장
#	전날대비 증가량은 주가이름_y_label로 저장

from openpyxl import load_workbook
import numpy as np
import os

if __name__ == '__main__': 
	
	for root, dirs, files in os.walk('./data'):
		for file in files:
			filename = file[:-5]
			
			filedata = load_workbook('./data/'+file)

			sheet1 = filedata['Sheet1']

			sheet2 = filedata.active

			print(sheet1['D4'].value)
			print(sheet2['D4'].value)
			datas = []
			x_labels = []
			y_labels = []

			for i in sheet1.rows:
				data=[]

				for d in i:
					data.append(d.value)

		
				datas.append(data[:6])
				x_labels.append(data[6:-1])
				y_labels.append(data[-1])

			np.save('./data_label/'+filename+'_data.npy',datas)
			np.save('./data_label/'+filename+'_x_label.npy',x_labels)
			np.save('./data_label/'+filename+'_y_label.npy',y_labels)
			
