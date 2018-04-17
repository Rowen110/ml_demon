# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    ## 5.2 损失函数：logistic损失（-1， 1）/ SVM Hinge损失/ 0/1损失
    # x = np.array(np.linspace(start=-2, stop=3, num=1001, dtype=np.float))
    # y_logit = np.log(1 + np.exp(-x) / np.math.log(2))
    # y_boost = np.exp(-x)
    # y_01 = x < 0
    # y_hinge = 1.0 - x
    # y_hinge[y_hinge < 0] = 0
    # plt.plot(x, y_logit, 'r-', label='Logistic Loss', linewidth=2)
    # plt.plot(x, y_01, 'g-', label='0/1 Loss',  linewidth=2)
    # plt.plot(x, y_hinge, 'b-', label='Hinge Loss', linewidth=2)
    # plt.plot(x, y_boost, 'm--', label='Adaboost Loss', linewidth=2)
    #
    # plt.grid()
    # plt.legend(loc='upper right')

    # x ** x        x > 0
    # (-x) ** (-x)  x < 0
    def f(x):
        y = np.ones_like(x)
        i = x > 0
        y[i] = np.power(x[i], x[i])
        i = x < 0
        y[i] = np.power(-x[i], -x[i])
        return y

    # 5.3 x^x
    x = np.linspace(-1.3, 1.3, 101)
    y = f(x)
    plt.plot(x, y, 'g-', label='x^x', linewidth=2)
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()


