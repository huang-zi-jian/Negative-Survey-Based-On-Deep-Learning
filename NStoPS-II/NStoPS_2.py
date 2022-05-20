'''
author: hzj
date: 2022-4-1
file info: 实现NStoPS-II算法
'''
import numpy as np


def NStoP2_2(R):
    '''

    :param R: 负调查结果
    :return: 正调查结果T
    '''

    # N为参与调查的人数；c为调查的类别数；N1表示当前有效的总人数；c1为当前有效的类别数
    N = sum(R)
    c = len(R)
    N1 = N
    c1 = c

    # 初始化正调查、答案数组
    t = np.empty(c)
    b = np.empty(c)

    for i in range(c):
        t[i] = N - (c - 1) * R[i]
        b[i] = 1

    while (True):
        # 如果当前正调查结果全是非负值，计算终止
        if min([t[i] for i in range(c)]) >= 0:
            break

        for i in range(c):
            if t[i] < 0:
                t[i] = 0
                b[i] = 0
                N1 = N1 - R[i]
                c1 = c1 - 1

        for i in range(c):
            if b[i] == 1:
                t[i] = N1 - (c1 - 1) * R[i]

        for i in range(c):
            if b[i] == 1:
                t[i] = N * t[i] / N1

    return t.tolist()


if __name__ == '__main__':
    print(NStoP2_2((33188, 32279, 28434, 22712, 22827, 28440, 32120)))
