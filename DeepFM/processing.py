# coding:utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 构造数据
# train_data, y = make_classification(n_samples=2000, n_features=5, n_informative=2, n_redundant=2, n_classes=2,
#                                     random_state=42)
# train = pd.DataFrame(train_data, columns=['int1', 'int2', 'float1', 's1', 's2'])
# train['clicked'] = y
# train['int1'] = train['int1'].map(int) + np.random.randint(0, 8)
# train['int2'] = train['int2'].map(int)
# train['s1'] = np.log(abs(train['s1'] + 1)).round().map(str)
# train['s2'] = np.log(abs(train['s2'] + 1)).round().map(str)
# # transform data
# categorical = ['int1', 'int2', 's1', 's2']
# numerical = ['float1']
# target = 'clicked'


def processing_feature(df, target=None, categorical=None, numerical=None):
	'''
	处理数据
	:param df:
	:param target: 预测值
	:param categorical: [list],离散变量的列名
	:param numerical: [list],连续变量的列名
	:return:df_index(数据在one-hot编码之后的索引)，df_value(数据的值，连续型变量为原值，离散变量为1)，
	'''
	# 分离目标变量
	if target:
		y = df[target]
		y = y.values.reshape(len(y), 1)#转换y的shape
	else:
		y = np.zero(df.shape[0])
		y = y.reshape(len(y), 1)
	df = df.drop(target, axis=1)
	m, n = df.shape
	cnt = 0
	df_index = df.copy()
	df_value = df.copy()
	df = pd.get_dummies(df, columns=categorical)
	# 离散变量的值全部转化为1
	for name in categorical:
		df_value[name] = np.ones(m)
	for i in range(len(df.columns)):
		cnt += 1
		col_name = df.columns[i]
		if '_' not in col_name:
			df_index[col_name] = np.array([i for j in range(m)])
		else:
			col = col_name.split('_')[0]
			for j in df.index:
				###根据index取得行数据，解决train_test_split之后出现Nan值的bug
				if df.loc[j, col_name] == 1:
					df_index.at[j,col] = i
	return df_index, df_value, y, cnt

def get_data():
	# 构造数据
	train_data, y = make_classification(n_samples=2000, n_features=5, n_informative=2, n_redundant=2, n_classes=2,
	                                    random_state=42)
	train = pd.DataFrame(train_data, columns=['int1', 'int2', 'float1', 's1', 's2'])
	train['clicked'] = y
	train['int1'] = train['int1'].map(int) + np.random.randint(0, 8)
	train['int2'] = train['int2'].map(int)
	train['s1'] = np.log(abs(train['s1'] + 1)).round().map(str)
	train['s2'] = np.log(abs(train['s2'] + 1)).round().map(str)
	# transform data
	categorical = ['int1', 'int2', 's1', 's2']
	numerical = ['float1']
	target = 'clicked'
	df_index, df_value, y, cnt = processing_feature(train, target=target, categorical=categorical, numerical=numerical)
	data = {}
	data['y_train'] = y
	data['xi'] = np.array(df_index.values).astype('int32')
	data['xv'] = np.array(df_value.values).astype('float32')
	data['feat_dim'] = cnt
	return data

def get_data_train_test(test_size=0.1,random_state=112):
	'''
	获取训练集和测试集
	:param test_size:
	:param random_state:
	:return:
	'''
	original_data, y = make_classification(n_samples=2000, n_features=5, n_informative=2, n_redundant=2, n_classes=2,
	                                    random_state=42)
	train = pd.DataFrame(original_data, columns=['int1', 'int2', 'float1', 's1', 's2'])
	train['clicked'] = y
	train['int1'] = train['int1'].map(int) + np.random.randint(0, 8)
	train['int2'] = train['int2'].map(int)
	train['s1'] = np.log(abs(train['s1'] + 1)).round().map(str)
	train['s2'] = np.log(abs(train['s2'] + 1)).round().map(str)

	train_data, test_data = train_test_split(train,test_size=test_size,random_state=random_state)
	# transform data
	categorical = ['int1', 'int2', 's1', 's2']
	numerical = ['float1']
	target = 'clicked'

	train_index, train_value, y_train, cnt_train = processing_feature(train_data, target=target, categorical=categorical, numerical=numerical)
	data_train = {}
	data_train['y_train'] = y_train
	data_train['xi'] = np.array(train_index.values).astype('int32')
	data_train['xv'] = np.array(train_value.values).astype('float32')
	data_train['feat_dim'] = cnt_train

	test_index, test_value, y_test, cnt_test = processing_feature(test_data, target=target, categorical=categorical, numerical=numerical)
	data_test = {}
	data_test['y_train'] = y_test
	data_test['xi'] = np.array(test_index.values).astype('int32')
	data_test['xv'] = np.array(test_value.values).astype('float32')
	data_test['feat_dim'] = cnt_test
	return data_train,data_test


if __name__ == '__main__':
	data_train,data_test = get_data_train_test()
	# data = get_data()
