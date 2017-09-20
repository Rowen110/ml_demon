# -*- coding: utf-8 -*-
import numpy as np

if __name__ == "__main__":
    # d = np.logspace(1, 2, 10, endpoint=True)
    # print d
    #
    # f = np.logspace(0, 10, 11, endpoint=True, base=2)
    # print f
    #
    # s = 'abcd'
    # g = np.fromstring(s, dtype=np.int8)
    # print g
    #
    # a = np.arange(11, 20)
    # print a
    # print a[3]
    # print a[3:6]
    # print a[:5]
    # print a[3:]
    # print a[1:9:2]
    # print a[::-1]
    #
    # # 切片数据是原始数据的一个视图
    # a[1:4] = 10, 20, 30
    # print a
    #
    # #
    # b = a[2:5]
    # b[0] = 3000
    # print a

    # a = np.random.rand(10)
    # print a
    # print a > 0.5
    # print a[a > 0.5]

    # # 行向量
    # a = np.arange(0, 60, 10)
    # print a
    # # 行转列向量
    # b = a.reshape((-1, 1))
    # print b
    #
    #
    # c = np.arange(6)
    # print c
    # f = c + b
    # print f
    #
    g = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)
    print g
    print g[(0, 1, 2, 3), (2, 3, 4, 5)]
    print g[3:, [0, 2, 5]]
