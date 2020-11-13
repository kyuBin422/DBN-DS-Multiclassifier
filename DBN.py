# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   date：          6/1/2020
-------------------------------------------------
   Change Activity:
                   6/1/2020:
-------------------------------------------------
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from SupervisedDBNClassification import SupervisedDBNClassification
from sklearn.metrics import accuracy_score, classification_report

# 统一分类
X = np.load("achieve/ProcessedData.npy")
label = np.load("achieve/label.npy")
choice = np.random.choice(X.shape[0], int(X.shape[0] / 100))
X = X[choice, :]
label = label[choice]
ss = StandardScaler()
X = ss.fit_transform(X)
prob = []
x = np.zeros((X.shape[0], 22, X.shape[1]), dtype=np.int8)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        key = int(X[i, j] * 1e5)
        x[i, 0, j] = int(key >= 0)
        key = key if key >= 0 else -key
        key = np.binary_repr(key)
        key = '0' * (21 - key.__len__()) + key
        for loop, bit in enumerate(key):
            x[i, loop, j] = int(bit)

X_train, X_test, y_train, y_test = train_test_split(x, label, test_size=0.33, random_state=42)

for loop in range(X.shape[1]):
    classifier = SupervisedDBNClassification(hidden_layers_structure=[10, 10],
                                             learning_rate_rbm=0.01,
                                             learning_rate=0.6,
                                             n_epochs_rbm=30,
                                             n_iter_backprop=30,
                                             batch_size=27,
                                             activation_function='sigmoid',
                                             dropout_p=0.1)

    classifier.fit(X_train[:, :, loop], y_train)

    Y_pred = classifier.predict_proba_dict(X_test[:, :, loop])
    prob.append(Y_pred)
    # print('Done.\nAccuracy: %f' % accuracy_score(y_test, Y_pred))


def getIdx(prob):
    prob = np.array(prob)
    m1 = np.product(prob[:, 0]) / (1 - np.product(prob, 0).sum())
    m2 = np.product(prob[:, 1]) / (1 - np.product(prob, 0).sum())
    return int(0 if m1 > m2 else 2)


idx = []
tmp = np.load("prob.npy", allow_pickle=True)
data = np.zeros((tmp.shape[0], 2, tmp.shape[1]))
for i in range(tmp.shape[0]):
    for j in range(tmp.shape[1]):
        data[i, 0, j] = tmp[i, j][0]
        data[i, 1, j] = tmp[i, j][2]
for i in range(data.shape[2]):
    idx.append(getIdx(data[:, :, i]))
classification_report(y_test, idx)
