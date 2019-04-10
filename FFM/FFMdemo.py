#coding:utf-8
'''
就是垃圾
'''
import ffm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000,n_features=20, n_clusters_per_class=1)

# prepare the data
# (field, index, value) format

# X = [[(1, 2, 1), (2, 3, 1), (3, 5, 1)],
#      [(1, 0, 1), (2, 3, 1), (3, 7, 1)],
#      [(1, 1, 1), (2, 3, 1), (3, 7, 1), (3, 9, 1)],]
#
# y = [1, 1, 0]
#
ffm_data = ffm.FFMData(X, y)

def get_label(prob,rating=0.5):
    if prob >= rating:
        return 1
    else:
        return 0

# train the model for 10 iterations

n_iter = 10

model = ffm.FFM(eta=0.1, lam=0.0001, k=4)
model.init_model(ffm_data)

for i in range(n_iter):
    print('iteration %d, ' % i, end='')
    model.iteration(ffm_data)

    y_pred = model.predict(ffm_data)

    y_pred_label = [get_label(i) for i in y_pred]
    auc = accuracy_score(y, y_pred_label)
    print('train auc %.4f' % auc)