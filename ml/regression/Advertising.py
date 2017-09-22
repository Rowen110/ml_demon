# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

__author__ = 'fc.w'

"""
1. 数据加载四种方式
2. 画图
3. 线性回归demo、预测数据展示
"""
if __name__ == "__main__":
    path = '../../resources/regression/Advertising.csv'

    # # 1. 手写读取数据
    # f = file(path)
    # x = []
    # y = []
    # for i, d in enumerate(f):
    #     if i == 0:
    #         continue
    #     d = d.strip()
    #     if not d:
    #         continue
    #     d = map(float, d.split(','))
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    #
    # x = np.array(x)
    # y = np.array(y)
    # print x
    # print y

    # # 2. python自带库
    # f = file(path, 'rb')
    # print f
    # d = csv.reader(f)
    # for line in d:
    #     print line
    # f.close()

    # # 3. numpy读入
    # p = np.loadtxt(path, delimiter=',', skiprows=1)
    # print p

    # 4. pandas读入
    # TV、Radio、Newspaper、Sales
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data[['Sales']]
    # print x
    # print y

    # # 绘制1
    # plt.plot(data['TV'], y, 'ro', label='TV')
    # plt.plot(data['Radio'], y, 'g*', label='Radio')
    # plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()

    # 绘制2
    # plt.figure(figsize=(9, 12))  # 画图之前首先设置figure对象，此函数相当于设置一块自定义大小的画布，使得后面的图形输出在这块规定了大小的画布上，其中参数figsize设置画布大小
    # plt.subplot(311)  # 将figure设置的画布大小分成几个部分，参数311 表示3(row)x1(colu),即将画布分成3x1，1表示选择图形输出的区域在第一块，图形输出区域参数必须在“行x列”范围  ，此处必须在1和2之间选择——如果参数设置为subplot(111)，则表示画布整个输出，不分割成小块区域，图形直接输出在整块画布上
    # plt.plot(data['TV'], y, 'ro')
    # plt.title('TV')
    # plt.grid()
    #
    # plt.subplot(312)
    # plt.plot(data['Radio'], y, 'g*')
    # plt.title('Radio')
    # plt.grid()
    #
    # plt.subplot(313)
    # plt.plot(data['Newspaper'], y, 'mv')
    # plt.title('Newspaper')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    # 随机将数据分为测试集和训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print "model：", model
    print "系数：", linreg.coef_
    print "截距：", linreg.intercept_

    # 预测
    y_hat = linreg.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print "均方差：", mse
    print "均方根误差： ", rmse

    t = np.arange(len(x_test))
    # 测试结果
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    # 预测结果
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()


