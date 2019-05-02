#!/usr/bin/python
#-*- coding: cp949 -*-
#-*- coding: utf-8 -*- 

#	저장된 엑셀을 불러온다.
#	일자, 시가, 고가, 저가, 종가, 거래량은 data로 저장
#	MACD, EMA, Momentum, SlowK, SlowD, ROC, William_R, A/D, RSI, OBV, upperband, lowerband는 기술적 지표로서 학습에 사용될 X_label로 저장
#	전날대비 증가량은 y_label로 저장

from openpyxl import load_workbook
import numpy as np

if __name__ == '__main__':
	sk_data = load_workbook(filename='chart.xlsx')

	sheet1 = sk_data['Sheet1']

	sheet2 = sk_data.active

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

	np.save('sk_data.npy',datas)
	np.save('sk_x_label.npy',x_labels)
	np.save('sk_y_label.npy',y_labels)
