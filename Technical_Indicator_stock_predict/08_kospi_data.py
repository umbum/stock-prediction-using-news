#-*- coding: cp949 -*-
#-*- coding: utf-8 -*- 

import pandas
import numpy as np
import talib
def csv2np(cvs_date):
	i= []
	for data in cvs_date[1:]:
		newstr = data.replace(",", "")
		i.append(newstr)
	i = [float(a) for a in i]
	i.reverse()
	return np.array(i)

def change2np(cvs_date):
	i= []
	for data in cvs_date[1:]:
		newstr = data.replace("%", "")
		if(float(newstr)>0):
			i.append(1)
		else:
			i.append(0)
	i.reverse()
	return np.array(i)

if __name__ == '__main__':

	csv_data = pandas.read_csv('c:/users/thwjd/desktop/kospi.csv', names=['dates','closes','opens','highs','lows','vol','change'])
	date = np.array(csv_data.dates[1:])
	date = np.flip(date)
	close = csv2np(csv_data.closes)
	high = csv2np(csv_data.highs)
	low = csv2np(csv_data.lows)
	change = change2np(csv_data.change)
	np.save('kospi_y.npy',change)
	exit()
	print(change)

	a = np.vstack((date,close,high,low))
	a= a.T

	macd, macdsignal,macdhist = talib.MACD(close)
	ema = talib.EMA(close)
	momentum = talib.MOM(close)
	slowk, slowd = talib.STOCH(high,low,close)
	roc = talib.ROC(close)
	willr = talib.WILLR(high,low,close)
	rsi = talib.RSI(close)
	upband, middle,lowband = talib.BBANDS(close)

	a = np.vstack((macd, macdsignal, macdhist, ema, momentum, slowk, slowd, roc, willr, rsi, middle))
	a= a.T
	np.save('kospi_x.npy',a)



