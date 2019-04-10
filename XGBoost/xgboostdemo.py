#coding:utf-8
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.metrics import f1_score,precision_score,accuracy_score,confusion_matrix
from sklearn.preprocessing import minmax_scale

current_path = os.path.realpath(__file__)
father_path = os.path.dirname(os.path.dirname(current_path))
trainData = os.path.join(father_path,'XGBoost/data/diabetes_train.txt')
testData = os.path.join(father_path,'XGBoost/data/diabetes_test.txt')


def benchmark(model, testset, label):
	pred = model.predict(testset)
	rmse = accuracy_score(label, pred)
	print("accuracy:", rmse)
	return rmse

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
	return feature,label

if __name__ == '__main__':
	train = pd.read_csv(trainData)
	test = pd.read_csv(testData)
	dataTrain, labelTrain = preprocessData(train)
	dataTest, labelTest = preprocessData(test)
	model = xgb.XGBClassifier(objective='binary:logistic',colsample_bytree=0.8,learning_rate=0.01,max_depth=5,alpha=10,n_estimators=300,subsample=0.7,base_score=0.5)
	model.fit(dataTrain,labelTrain)
	benchmark(model, dataTest, labelTest)


