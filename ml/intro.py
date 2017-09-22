# -*- utf-8 -*-
import numpy as np

if __name__=="__main__":
    L = [1, 2, 3, 4, 5]
    print("L =  ", L)
    a = np.array(L)
    print("a = ", a)
    print(type(a))

    b = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=np.float)
    d = b.astype(np.int)
    print(d)

    print(b.shape)
    b.shape = 5, 3
    print(b)
    f = np.arange(1, 10, 0.5)
    print(f)

    g = np.linspace(1, 10, 20)
    print(g)

    d = np.logspace(1, 2, 10, endpoint=True)
    print(d)

    s = 'abcd'
    g = np.fromstring(s, dtype=np.int8)
    print(g)



    a = np.arange(10)
    print(a[3])
    print(a[3:6])
    print(a[:5])
    print(a[1:9:2])
    print(a[::-1])