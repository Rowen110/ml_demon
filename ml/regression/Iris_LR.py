#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__author__ = 'fc.w'


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

if __name__ == "__main__":
    path = u'./resources/iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    # print data
    x, y = np.split(data, (4,), axis=1)  # 将数据的0到3列组成x，第4列得到y
    x = x[:, :2]  # 为了可视化，仅使用前两列特征
    # print x
    # print y

    '''
    对矩阵进行标准化处理，变换后的矩阵各特征均值为0，方差为1
    fit方法是用于从一个训练集中学习模型参数
    fit_transform就很高效的将模型训练和转化合并到一起，训练样本先做fit，得到mean，standard deviation，然后将这些参数用于transform（归一化训练数据），使得到的训练数据是归一化的 
    '''
    # x = StandardScaler().fit_transform(x)
    # # Logistic回归模型
    # lr = LogisticRegression()
    # # 根据数据[x,y]，计算回归参数；
    # lr.fit(x, y.ravel())

    # 等价形式
    '''
    sc: 特征标准化
    clf : 分类器
    '''
    lr = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(random_state=1))])
    lr.fit(x, y.ravel())

    # 画图
    N, M = 500, 500  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 三维曲线和曲面的绘制, 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = lr.predict(x_test)                  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)     # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)    # 样本的显示
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.savefig('2.png')
    plt.show()

    # 训练集上的预测结果
    y_hat = lr.predict(x)
    y = y.reshape(-1)
    result = y_hat == y
    print y_hat
    print result
    acc = np.mean(result)
    print '准确度: %.2f%%' % (100 * acc)

