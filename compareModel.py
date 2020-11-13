# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   date：          6/3/2020
-------------------------------------------------
   Change Activity:
                   6/3/2020:
-------------------------------------------------
"""
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X = np.load("achieve/ProcessedData.npy")
label = np.load("achieve/label.npy")

ss = StandardScaler()
X = ss.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.33, random_state=42)

clf = SVC()
function = [KNeighborsClassifier, MLPClassifier, SVC]
for key in function:
    clf = key()
    clf.fit(X_train, y_train)
    predictY = clf.predict(X_test)
    print(str(key) + 'Done.Accuracy: %f', accuracy_score(y_test, predictY))

plt.plot(predictY,label="predictY")
plt.plot(y_test,label="y_test")
plt.show()