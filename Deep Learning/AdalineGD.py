import numpy as np


class AdalineGD(object):
    """
    eat: float
    学习效率,处于0和1

    n_iter:int
    对训练数据进行学习改进次数

    w_: 一维向量
    存储权重数值

    error_:
    存储每次迭代改进时,网络对数据进行错误判断的次数
    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """

        :param X:二维数组[n_samples,n_features]
        n_samples 表示X中含有训练数据条目数
        n_features 含有4个数据的一维向量,用于表示一条训练条目

        :param y:一维向量
        用于存储每一训练条目对应的正确分类

        :return:None
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eat * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)
