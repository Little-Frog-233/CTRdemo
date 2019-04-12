# coding:utf-8
'''
数据较为稀疏时采用fm算法
'''
import random
import numpy as np
import pandas as pd
import os
import math
from math import exp
from random import normalvariate  # 正态分布
from sklearn.metrics import f1_score, precision_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import log_loss
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

current_path = os.path.realpath(__file__)
father_path = os.path.dirname(os.path.dirname(current_path))
trainData = os.path.join(father_path, 'FM/data/diabetes_train.txt')
testData = os.path.join(father_path, 'FM/data/diabetes_test.txt')


def preprocessData(data):
	'''
	处理数据的函数
	:param data: [pd.dataframe]
	:return: x[np.array],y[pd.dataframe]
	'''
	feature = data.iloc[:, :-1]  # 取特征
	label = data.iloc[:, -1]
	# 将数组按列进行归一化
	feature = minmax_scale(feature, axis=0)
	return feature, label


def benchmark(model, testset, label):
	pred = model.predict(testset)
	rmse = accuracy_score(label, pred)
	print("accuracy:", rmse)
	return rmse


class fm:
	def __init__(self, k=20, iter=200, label=True, rate=0.01, rating=0.5, batch_size=1):
		'''
		模型初始化
		:param k: 辅助向量的大小
		:param iter: 迭代次数
		:param label: 是否输出标签，True:输出标签，False:输出概率
		:param rate: 学习率大小
		:param rating: 二分类判断的概率，默认为0.5
		'''
		self.k = k
		self.iter = iter
		self.label = label
		self.rate = rate
		self.rating = rating
		self.batch_size = batch_size

	def sigmoid(self, inx):
		# return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
		return 1.0 / (1 + exp(-inx))

	def sigmoid_np(self, inx):
		# return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
		return 1.0 / (1 + np.exp(-inx))

	# def fit(self,X,y,out=False):
	# 	'''
	# 	模型训练函数
	# 	:param X: [pd.dataframe]
	# 	:param y: [pd.dataframe]
	# 	:param out:
	# 	:return:
	# 	'''
	# 	X = np.mat(X) #将X转化为矩阵
	# 	y = np.array(y.map(lambda x: 1 if x==1 else -1)) # 将标签转化为1和-1
	# 	m, n = np.shape(X)  # 矩阵的行列数，即样本数和特征数
	# 	alpha = self.rate
	# 	# 初始化参数
	# 	# w = random.randn(n, 1)#其中n是特征的个数
	# 	w = np.zeros((n, 1))  # 一阶特征的系数
	# 	w_0 = 0.
	# 	v = normalvariate(0, 0.2) * np.ones((n, self.k))  # 即生成辅助向量，用来训练二阶交叉特征的系数
	#
	# 	for it in range(self.iter):
	# 		for x in range(m):  # 随机优化，每次只使用一个样本
	# 			# 二阶项的计算
	# 			inter_1 = X[x] * v
	# 			inter_2 = np.multiply(X[x], X[x]) * np.multiply(v, v)  # 二阶交叉项的计算
	# 			interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.  # 二阶交叉项计算完成
	#
	# 			p = w_0 + X[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
	# 			loss = 1 - self.sigmoid(y[x] * p[0, 0])  # 计算损失
	#
	# 			w_0 = w_0 + alpha * loss * y[x]
	#
	# 			for i in range(n):
	# 				if X[x, i] != 0:
	# 					w[i, 0] = w[i, 0] + alpha * loss * y[x] * X[x, i]
	# 					for j in range(self.k):
	# 						v[i, j] = v[i, j] + alpha * loss * y[x] * (
	# 							X[x, i] * inter_1[0, j] - v[i, j] * X[x, i] * X[x, i])
	# 		if out:
	# 			if it % 100 == 0:
	# 				print("第{}次迭代后的损失为{}".format(it, loss))
	# 	self.w_0 = w_0
	# 	self.w = w
	# 	self.v = v

	def fit_logit(self, X, y, out=False):
		'''
		模型训练函数
		:param X: [pd.dataframe]
		:param y: [pd.dataframe]
		:param out:
		:return:
		'''
		X = np.mat(X)  # 将X转化为矩阵
		if type(y) == pd.DataFrame:
			y = np.array(y.map(lambda x: 1 if x == 1 else 0))  # 将标签转化为1和-1

		m, n = np.shape(X)  # 矩阵的行列数，即样本数和特征数
		alpha = self.rate
		# 初始化参数
		# w = random.randn(n, 1)#其中n是特征的个数
		w = np.zeros((n, 1))  # 一阶特征的系数
		w_0 = 0.
		v = normalvariate(0, 0.1) * np.ones((n, self.k))  # 即生成辅助向量，用来训练二阶交叉特征的系数

		for it in range(self.iter):
			for x in range(m):
				# 二阶项的计算
				inter_1 = X[x] * v
				inter_2 = np.multiply(X[x], X[x]) * np.multiply(v, v)  # 二阶交叉项的计算
				interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.  # 二阶交叉项计算完成
				p = w_0 + X[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
				logit = self.sigmoid(p[0, 0])
				loss = -(y[x] * math.log(logit) + (1 - y[x]) * math.log(1 - logit))  # 交叉熵损失函数
				w_0 = w_0 - alpha * ((logit - y[x]) * 1)
				for i in range(n):
					if X[x, i] != 0:
						# 梯度优化
						gradient = (logit - y[x])
						w[i, 0] = w[i, 0] - alpha * gradient * X[x, i]
						for j in range(self.k):
							v[i, j] = v[i, j] - alpha * gradient * (
								X[x, i] * inter_1[0, j] - v[i, j] * X[x, i] * X[x, i])
			if out:
				if it % 10 == 0:
					print("第{}次迭代后的损失为{}".format(it, loss))
		self.w_0 = w_0
		self.w = w
		self.v = v

	def fit_SGD(self, X, y, out=False):
		'''
		已放弃改进
		去他妈的
		:param X: [pd.dataframe]
		:param y: [pd.dataframe]
		:param out:
		:return:
		'''
		X = np.mat(X)  # 将X转化为矩阵
		if type(y) == pd.DataFrame:
			y = np.array(y.map(lambda x: 1 if x == 1 else 0))  # 将标签转化为1和-1
		m, n = np.shape(X)  # 矩阵的行列数，即样本数和特征数
		alpha = self.rate
		# 初始化参数
		# w = random.randn(n, 1)#其中n是特征的个数
		w = np.zeros((n, 1))  # 一阶特征的系数
		w_0 = 0.
		v = normalvariate(0, 0.1) * np.ones((n, self.k))  # 即生成辅助向量，用来训练二阶交叉特征的系数

		for it in range(self.iter):
			for _ in range(self.batch_size):
				batch = set()
				for num in range(m):
					batch.add(random.randint(0, m - 1))
				# for x in batch:  # 随机优化，每次只使用一个样本
				batch = [i for i in range(m)]
				length = len(batch)
				# 二阶项的计算
				inter_1 = X[batch] * v
				inter_2 = np.multiply(X[batch], X[batch]) * np.multiply(v, v)  # 二阶交叉项的计算
				interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2, axis=1) / 2. # 二阶交叉项计算完成
				p = w_0 + X[batch] * w + interaction # 计算预测的输出，即FM的全部项之和
				logit = self.sigmoid_np(p)
				loss = log_loss(y[batch],logit) # 交叉熵损失函数
				w_0 = w_0 - alpha * ((np.sum(logit - y[batch]) / length) * 1)
				for i in range(n):
					# 梯度优化
					gradient = (logit - y[batch])
					w[i, 0] = w[i, 0] - (alpha * (np.sum(np.multiply(gradient, X[batch, i]), axis=1) / length))[0]
					for j in range(self.k):
						v[i, j] = v[i, j] - alpha * (np.sum(np.multiply(gradient, (
						np.multiply(X[batch,i],inter_1[:, j]) - v[i, j] * np.multiply(X[batch, i], X[batch, i]))), axis=1) / length)[0]
			if out:
				if it % 10 == 0:
					print("第{}次迭代后的损失为{}".format(it, loss))
		self.w_0 = w_0
		self.w = w
		self.v = v

	def predict(self, X):
		X = np.mat(X)
		m, n = np.shape(X)
		allItem = 0
		result = []
		for x in range(m):  # 计算每一个样本的误差
			allItem += 1
			inter_1 = X[x] * self.v
			inter_2 = np.multiply(X[x], X[x]) * np.multiply(self.v, self.v)
			interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
			p = self.w_0 + X[x] * self.w + interaction # 计算预测的输出
			pre = self.sigmoid(p[0, 0])
			if self.label:
				if pre >= self.rating:
					result.append(1)
				else:
					result.append(0)
			else:
				result.append(pre)
		return np.array(result)


if __name__ == '__main__':
	# train = pd.read_csv(trainData, header=None)
	# test = pd.read_csv(testData, header=None)
	# dataTrain, labelTrain = preprocessData(train)
	# dataTest, labelTest = preprocessData(test)
	# model = fm(k=25, iter=200, rate=0.01)
	# print("开始训练")
	# model.fit_SGD(dataTrain, labelTrain, True)
	# benchmark(model, dataTest, labelTest)
	X, y = make_classification(n_samples=1000, n_features=10, n_clusters_per_class=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
	model = fm(k=25, iter=100, rate=0.001)
	print("开始训练")
	model.fit_logit(X_train, y_train, True)
	benchmark(model, X_test, y_test)

