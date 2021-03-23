import numpy as np


class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train,y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        # 求逆矩阵 np.linalg.inv()
        # 点乘 dot()
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的结果向量"""
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[0] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train,y_train,使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])

            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            i_iter = 0

            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                i_iter += 1

            return theta

        ones = np.ones((100, 1))
        X_b = np.hstack([ones, X_train])
        inital_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b,y_train,inital_theta,eta,n_iters)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def __repr__(self):
        return "LinearRegression()"
