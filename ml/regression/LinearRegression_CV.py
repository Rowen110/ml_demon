#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV

__author__ = 'fc.w'

"""
线性回归
"""
if __name__ == "__main__":
    # pandas读入数据 格式：TV、Radio、Newspaper、Sales
    data = pd.read_csv('./resources/Advertising.csv')
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data[['Sales']]
    # print x
    # print y

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    print x_train, y_train
    # 正则化: Lasso表示L1、Ridge表示L2
    model = Lasso()
    # model = Ridge()

    # alpha学习率
    alpha_can = np.logspace(-3, 2, 10)
    # param_grid: 最优化的参数的取值;   cv: 交叉验证参数;  n_jobs:并发数
    lasso_model = GridSearchCV(estimator=model, param_grid={'alpha': alpha_can}, cv=5, n_jobs=2)
    # 运行网格搜索
    lasso_model.fit(x, y)
    print '不同参数情况下的评价结果: ', lasso_model.cv_results_
    print '已取得最佳结果的参数的组合: ', lasso_model.best_params_
    print '成员提供优化过程期间观察到的最好的评分: ', lasso_model.best_score_

    y_hat = lasso_model.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)
    rmse = np.sqrt(mse)
    print "均方差：", mse
    print "均方根误差： ", rmse

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
