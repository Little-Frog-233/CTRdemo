#coding:utf-8
'''
就是垃圾
参考github,增加了一个从pd.Dataframe转化为(field, index, value) format的脚本
'''
import ffm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from FFM.FFMFormatPandas import FFMFormatPandas

def get_label(prob,rating=0.5):
    if prob >= rating:
        return 1
    else:
        return 0

X, y = make_classification(n_samples=1000,n_features=20, n_clusters_per_class=1)

# prepare the data
# (field, index, value) format

# X = [[(1, 2, 1), (2, 3, 1), (3, 5, 1)],
#      [(1, 0, 1), (2, 3, 1), (3, 7, 1)],
#      [(1, 1, 1), (2, 3, 1), (3, 7, 1), (3, 9, 1)],]
#
# y = [1, 1, 0]
#
# ffm_data = ffm.FFMData(X, y)
#
#
# # train the model for 10 iterations
#
# n_iter = 10
#
# model = ffm.FFM(eta=0.1, lam=0.0001, k=4)
# model.init_model(ffm_data)
#
# for i in range(n_iter):
#     print('iteration %d, ' % i, end='')
#     model.iteration(ffm_data)
#
#     y_pred = model.predict(ffm_data)
#
#     y_pred_label = [get_label(i) for i in y_pred]
#     auc = accuracy_score(y, y_pred_label)
#     print('train auc %.4f' % auc)

#生成数据
train, y = make_classification(n_samples=2000, n_features=5, n_informative=2, n_redundant=2, n_classes=2, random_state=42)
train = pd.DataFrame(train, columns=['int1','int2','float1','s1','s2'])
train['int1'] = train['int1'].map(int) + np.random.randint(0, 8)
train['int2'] = train['int2'].map(int)
train['s1'] = np.log(abs(train['s1'] +1 )).round().map(str)
train['s2'] = np.log(abs(train['s2'] +1 )).round().map(str)
train['clicked'] = y

# transform data
categorical=['int1', 'int2', 's1', 's2']
numerical = ['float1']
target = 'clicked'
train_data, val_data = train_test_split(train, test_size=0.2)
y_test = np.array(val_data['clicked'].values)

ffm_train = FFMFormatPandas()
ffm_train.fit(train, target=target, categorical=categorical, numerical=numerical)
train_data = ffm_train.transform_convert(train_data)
val_data = ffm_train.transform_convert(val_data)


model = ffm.FFM(eta=0.1, lam=0.0001, k=4)
model.init_model(train_data)
n_iter = 10
for i in range(n_iter):
    print('iteration %d, ' % i, end='')
    model.iteration(train_data)

    y_pred = model.predict(val_data)

    y_pred_label = [get_label(i) for i in y_pred]
    accuracy = accuracy_score(y_test, y_pred_label)
    print('train auc %.4f' % accuracy)

# predict
# val_proba = model.predict_proba(val_data)
