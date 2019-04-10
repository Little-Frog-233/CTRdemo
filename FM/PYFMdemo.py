#coding:utf-8
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from pyfm import pylibfm
from sklearn.metrics import f1_score,precision_score,accuracy_score,confusion_matrix
from sklearn.datasets import make_classification


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

X, y = make_classification(n_samples=1000,n_features=100, n_clusters_per_class=1)
data = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X]

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)

v = DictVectorizer()
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)

fm = pylibfm.FM(num_factors=25, num_iter=100, verbose=False, task="classification", initial_learning_rate=0.001, learning_rate_schedule="optimal")

fm.fit(X_train,y_train)
y_pred_label = [get_label(i) for i in fm.predict(X_test)]
print(accuracy_score(y_test, y_pred_label))
