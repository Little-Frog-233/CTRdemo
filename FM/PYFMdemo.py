#coding:utf-8
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from pyfm import pylibfm
from sklearn.metrics import f1_score,precision_score,accuracy_score,confusion_matrix
from sklearn.datasets import make_classification
from sklearn.preprocessing import minmax_scale


def benchmark(model, testset, label):
	pred = model.predict(testset)
	rmse = accuracy_score(label, pred)
	print("accuracy:", rmse)
	return rmse

def get_label(prob,rating=0.5):
    if prob >= rating:
        return 1
    else:
        return 0

# X, y = make_classification(n_samples=500,n_features=5, n_clusters_per_class=1)
# #训练数据转换
# data = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X]
# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)
# v = DictVectorizer()
# X_train = v.fit_transform(X_train)
# X_test = v.transform(X_test)
# # print(X_train.toarray())

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
	feature=data.iloc[:,:-1]   #取特征
	label=data.iloc[:,-1]
	#将数组按列进行归一化
	feature = minmax_scale(feature,axis=0)#此处如果不归一化，预测结果全是Nan
	return feature,label

train=pd.read_csv(trainData,header=None)
test = pd.read_csv(testData,header=None)
X_train, y_train = preprocessData(train)
X_test, y_test = preprocessData(test)
X_train = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X_train]
X_test= [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X_test]
v = DictVectorizer()
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)

fm = pylibfm.FM(num_factors=15, num_iter=300, verbose=False, task="classification", initial_learning_rate=0.01, learning_rate_schedule="optimal")
fm.fit(X_train,y_train)
y_pred_label = [get_label(i) for i in fm.predict(X_test)]
print(y_pred_label)
print(accuracy_score(y_test, y_pred_label))
