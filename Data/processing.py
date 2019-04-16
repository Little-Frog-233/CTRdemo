# coding:utf-8
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


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
		# y = df[target]
		# y = y.values.reshape(len(y), 1)#转换y的shape
		y = pd.get_dummies(df[target])
		df = df.drop(target, axis=1)
	else:
		y = np.array(np.zero(df.shape[0])).astype('float32')
		y = y.reshape(len(y), 1)
	m, n = df.shape
	cnt = 0
	df_index = df.copy()
	df_value = df.copy()
	df = pd.get_dummies(df, columns=categorical)
	# 离散变量的值全部转化为1
	for name in categorical:
		df_value[name] = np.ones(m)
	for name in numerical:
		df_value[name] = minmax_scale(df_value[name])
	for i in range(len(df.columns)):
		cnt += 1
		col_name = df.columns[i]
			#连续变量
		if '_' not in col_name:
			df_index[col_name] = np.array([i for j in range(m)])
		else:
			# 离散变量
			col = col_name.split('_')[0]
			for j in df.index:
				###根据index取得行数据，解决train_test_split之后出现Nan值的bug
				if df.loc[j, col_name] == 1:
					df_index.at[j, col] = i
	return df_index, df_value, y, cnt

class feature_processing:
	'''
	对于训练集和测试集，需要一个相同的哑变量的columns
	先对训练集进行fit，再将测试集进行transform
	'''
	def __init__(self,target=None, categorical=None, numerical=None):
		self.target = target
		self.categorical = categorical
		self.numerical = numerical
		self.columns = None

	def fit_transform(self,df):
		'''

		:param df:
		:return:
		'''
		y = pd.get_dummies(df[self.target])
		df = df.drop(self.target, axis=1)
		m, n = df.shape
		cnt = 0
		df_index = df.copy()
		df_value = df.copy()
		df = pd.get_dummies(df, columns=self.categorical)
		self.columns = df.columns
		# 离散变量的值全部转化为1
		for name in self.categorical:
			df_value[name] = np.ones(m)
		for name in self.numerical:
			df_value[name] = minmax_scale(df_value[name])
		for i in range(len(self.columns)):
			cnt += 1
			col_name = df.columns[i]
			# 连续变量
			if '_' not in col_name:
				df_index[col_name] = np.array([i for j in range(m)])
			else:
				# 离散变量
				col = col_name.split('_')[0]
				for j in df.index:
					###根据index取得行数据，解决train_test_split之后出现Nan值的bug
					if df.loc[j, col_name] == 1:
						df_index.at[j, col] = i
		self.cnt = cnt
		return df_index, df_value, y, cnt


	def transform(self,df):
		'''

		:param df:
		:return:
		'''
		if len(self.columns) == 0:
			print('please fit first')
		if self.target in df.columns:
			y = pd.get_dummies(df[self.target])
			df = df.drop(self.target, axis=1)
		else:
			y = np.array(np.zero(df.shape[0])).astype('float32')
			y = y.reshape(len(y), 1)
		m, n = df.shape
		df_index = df.copy()
		df_value = df.copy()
		df = pd.get_dummies(df, columns=self.categorical)
		# 离散变量的值全部转化为1
		for name in self.categorical:
			df_value[name] = np.ones(m)
		for name in self.numerical:
			df_value[name] = minmax_scale(df_value[name])
		for i in range(len(self.columns)):
			if self.columns[i] not in df.columns:
				continue
			col_name = self.columns[i]
			# 连续变量
			if '_' not in col_name:
				df_index[col_name] = np.array([i for j in range(m)])
			else:
				# 离散变量
				col = col_name.split('_')[0]
				for j in df.index:
					###根据index取得行数据，解决train_test_split之后出现Nan值的bug
					if df.loc[j, col_name] == 1:
						df_index.at[j, col] = i
		return df_index, df_value, y, self.cnt


	def fit_transform_data(self,df):
		'''

		:param df:
		:return:
		'''
		df_index, df_value, y, cnt = self.fit_transform(df=df)
		data = {}
		data['y_train'] = np.array(y.values).astype('float32')
		data['xi'] = np.array(df_index.values).astype('int32')
		data['xv'] = np.array(df_value.values).astype('float32')
		data['feat_dim'] = cnt
		return data

	def transform_data(self,df):
		'''

		:param df:
		:return:
		'''
		df_index, df_value, y, cnt = self.transform(df=df)
		data = {}
		data['y_train'] = np.array(y.values).astype('float32')
		data['xi'] = np.array(df_index.values).astype('int32')
		data['xv'] = np.array(df_value.values).astype('float32')
		data['feat_dim'] = cnt
		return data




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
	data['y_train'] = np.array(y.values).astype('float32')
	data['xi'] = np.array(df_index.values).astype('int32')
	data['xv'] = np.array(df_value.values).astype('float32')
	data['feat_dim'] = cnt
	return data


def get_data_train_test(test_size=0.1, random_state=112):
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

	train_data, test_data = train_test_split(train, test_size=test_size, random_state=random_state)
	# transform data
	categorical = ['int1', 'int2', 's1', 's2']
	numerical = ['float1']
	target = 'clicked'

	train_index, train_value, y_train, cnt_train = processing_feature(train_data, target=target,
	                                                                  categorical=categorical, numerical=numerical)
	data_train = {}
	data_train['y_train'] = np.array(y_train.values).astype('float32')
	data_train['xi'] = np.array(train_index.values).astype('int32')
	data_train['xv'] = np.array(train_value.values).astype('float32')
	data_train['feat_dim'] = cnt_train

	test_index, test_value, y_test, cnt_test = processing_feature(test_data, target=target, categorical=categorical,
	                                                              numerical=numerical)
	data_test = {}
	data_test['y_test'] = np.array(y_test.values).astype('float32')
	data_test['xi'] = np.array(test_index.values).astype('int32')
	data_test['xv'] = np.array(test_value.values).astype('float32')
	data_test['feat_dim'] = cnt_test
	return data_train, data_test

def get_data_df(df,target=None,categorical=None,numerical=None):
	'''
	读取指定df，并生成data字典：data{'xi':x_index,'xv':x_value,'y_train':y,'feat_dim':number of feature}
	:param df:
	:param target:
	:param categorical:
	:param numerical:
	:return:
	'''
	df_index, df_value, y, cnt = processing_feature(df, target=target, categorical=categorical, numerical=numerical)
	data = {}
	data['y_train'] = np.array(y.values).astype('float32')
	data['xi'] = np.array(df_index.values).astype('int32')
	data['xv'] = np.array(df_value.values).astype('float32')
	data['feat_dim'] = cnt
	return data


if __name__ == '__main__':
	current_path = os.path.realpath(__file__)
	father_path = os.path.dirname(os.path.dirname(current_path))
	train_path = os.path.join(father_path, 'FM/data/diabetes_train.txt')
	test_path = os.path.join(father_path,'FM/data/diabetes_test.txt')
	train_data = pd.read_csv(train_path, header=None)
	train_df = pd.DataFrame(train_data.values, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'click'])
	print(train_df.head())

	test_data = pd.read_csv(test_path,header=None)
	test_df = pd.DataFrame(test_data.values, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'click'])
	# print(test_df.head())

	model = feature_processing(target='click',numerical=['c2','c3','c5','c7'],categorical=['c1','c4','c6','c8'])
	# train_index, train_value, y_train, cnt_train = model.fit_transform(train_df)
	# test_index, test_value, y_test, cnt_test = model.transform(test_df)
	data_train = model.fit_transform_data(train_df)
	data_test = model.transform_data(test_df)
