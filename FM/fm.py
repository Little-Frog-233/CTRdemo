#coding:utf-8
import numpy as np
import pandas as pd
import os
from math import exp
from random import normalvariate  # 正态分布
from sklearn.metrics import f1_score,precision_score,accuracy_score,confusion_matrix
from sklearn.preprocessing import minmax_scale

current_path = os.path.realpath(__file__)
father_path = os.path.dirname(os.path.dirname(current_path))
trainData = os.path.join(father_path,'FM/data/diabetes_train.txt')
testData = os.path.join(father_path,'FM/data/diabetes_test.txt')

def preprocessData(data):
	'''
	处理数据的函数
	:param data: [pd.dataframe]
	:return: x[np.array],y[pd.dataframe]
	'''
	feature=np.array(data.iloc[:,:-1])   #取特征
	label=data.iloc[:,-1]
	#将数组按列进行归一化
	_feature = minmax_scale(feature,axis=0)
	return _feature,label

def benchmark(model, testset, label):
	pred = model.predict(testset)
	rmse = accuracy_score(label, pred)
	print("accuracy:", rmse)
	return rmse

class fm:
	def __init__(self,k=20,iter=200,label=True,rate=0.01,rating=0.5):
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

	def sigmoid(self,inx):
		# return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
		return 1.0 / (1 + exp(-inx))

	def fit(self,X,y,out=False):
		'''
		模型训练函数
		:param X: [pd.dataframe]
		:param y: [pd.dataframe]
		:param out:
		:return:
		'''
		X = np.mat(X) #将X转化为矩阵
		y = np.array(y.map(lambda x: 1 if x==1 else -1)) # 将标签转化为1和-1
		m, n = np.shape(X)  # 矩阵的行列数，即样本数和特征数
		alpha = self.rate
		# 初始化参数
		# w = random.randn(n, 1)#其中n是特征的个数
		w = np.zeros((n, 1))  # 一阶特征的系数
		w_0 = 0.
		v = normalvariate(0, 0.2) * np.ones((n, self.k))  # 即生成辅助向量，用来训练二阶交叉特征的系数

		for it in range(self.iter):
			for x in range(m):  # 随机优化，每次只使用一个样本
				# 二阶项的计算
				inter_1 = X[x] * v
				inter_2 = np.multiply(X[x], X[x]) * np.multiply(v, v)  # 二阶交叉项的计算
				interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.  # 二阶交叉项计算完成

				p = w_0 + X[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
				loss = 1 - self.sigmoid(y[x] * p[0, 0])  # 计算损失

				w_0 = w_0 + alpha * loss * y[x]

				for i in range(n):
					if X[x, i] != 0:
						w[i, 0] = w[i, 0] + alpha * loss * y[x] * X[x, i]
						for j in range(self.k):
							v[i, j] = v[i, j] + alpha * loss * y[x] * (
								X[x, i] * inter_1[0, j] - v[i, j] * X[x, i] * X[x, i])
			if out:
				if it % 100 == 0:
					print("第{}次迭代后的损失为{}".format(it, loss))
		self.w_0 = w_0
		self.w = w
		self.v = v

	def predict(self,X):
		X = np.mat(X)
		m, n = np.shape(X)
		allItem = 0
		result = []
		for x in range(m):  # 计算每一个样本的误差
			allItem += 1
			inter_1 = X[x] * self.v
			inter_2 = np.multiply(X[x], X[x]) * np.multiply(self.v, self.v)
			interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
			p = self.w_0 + X[x] * self.w + interaction  # 计算预测的输出
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
	train=pd.read_csv(trainData)
	test = pd.read_csv(testData)
	dataTrain, labelTrain = preprocessData(train)
	dataTest, labelTest = preprocessData(test)
	model = fm(k=25,iter=300)
	print("开始训练")
	model.fit(dataTrain, labelTrain,True)
	benchmark(model,dataTest,labelTest)