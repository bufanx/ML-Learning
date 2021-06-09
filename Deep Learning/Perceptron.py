import numpy as np


class Perceptron(object):
    """
    eta :学习率
    n_iter :权重向量的训练次数
    w_ :神经分叉的权重向量
    errors_ :用于记录神经元判断出错的次数
    """

    def __init__(self, eta=0.01, n_iter=10):
        """
        初始化权重向量为0
        加一是因为前面算法提到的w0,也就是步调函数的阈值
        :param eta: 学习率
        :param n_iter: n_iter :权重向量的训练次数
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        输入训练数据,培训神经元,X输入样本向量,y对应样本分类
        :param X:shape[n_samples,n_features]
        :param y:[1,-1]
        :return:
        """
        self.errors_ = []
        self.w_ = np.zeros(1 + X.shape[1])
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                """
                update = η*(y-y')
                """
                update = self.eta * (target - self.predict(xi))

                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass
            pass

    def net_input(self, X):
        """
        z = W0*1 + W1*X1 +...+Wn*Xn
        :param self:
        :param X:
        :return:
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        pass

    pass
