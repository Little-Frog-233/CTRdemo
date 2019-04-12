# coding:utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

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
print(train.head())


def processing_feature(df, target=None, categorical=None, numerical=None):
	'''

	:param df:
	:param target: 预测值
	:param categorical: [list],离散变量的列名
	:param numerical: [list],连续变量的列名
	:return:
	'''
	# 分离目标变量
	y = df[target]
	y = y.reshape(len(y), 1)#转换y的shape
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
			for j in range(m):
				if df.iloc[j][col_name] == 1:
					df_index.at[j, col] = i
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
	data['xv'] = np.array(df_index.values).astype('float32')
	data['feat_dim'] = cnt
	return data


if __name__ == '__main__':
	df_index, df_value, y = processing_feature(train, target=target, categorical=categorical, numerical=numerical)
	print(type(df_index.values.tolist()))
