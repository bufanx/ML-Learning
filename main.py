from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from .PlayML.Model_selection import train_test_split
from .PlayML.LogisticRegression import LogisticRegression

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y < 2, :2]
y = y[y < 2]

plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
