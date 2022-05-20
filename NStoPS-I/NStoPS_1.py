'''
author: hzj
date: 2022-4-1
file info: 实现NStoPS-I算法
'''
import numpy as np


def NStoP2_1(R):
    '''

    :param R: 负调查结果
    :return: 正调查结果T
    '''

    # N为参与调查的人数；c为调查的类别数；error_min表示允许的最小误差；n为迭代次数
    N = sum(R)
    c = len(R)
    error_min = 10 ** -4
    n = 0

    # 初始化正调查迭代数组
    T_array = np.empty((1, c))
    T_array[0] = np.array([N / c] * c)
    # 定义mij概率矩阵
    m_array = np.empty((c, c))

    while (True):
        # delta_max为没每次迭代结果的最大误差，这里初始设置为-1
        delta_max = -1

        # 通过正调查数组计算mij矩阵
        for i in range(c):
            for j in range(c):
                if i != j:
                    # 因为NStoPS_I在迭代的过程中有可能出现 ti>=N 的情况，而没有实际意义，所以出现该情况这里一律用概率值为0处理
                    if N - T_array[n][i] <= 0:
                        m_array[i, j] = 0
                    else:
                        m_array[i, j] = T_array[n][j] / (N - T_array[n][i])
                else:
                    m_array[i, j] = 0

        # 迭代更新正调查结果
        t = np.zeros(c)
        for j in range(c):
            for i in range(c):
                if i != j:
                    t[j] = t[j] + R[i] * m_array[i, j]

        # 修正结果，这里t1为例，使得满足sum(ti)=N的条件 todo:可以考虑对整个t数组进行缩放，而不是只对t[0]进行校准
        # t[0] = t[0] + N - sum(t)

        t_sum = sum(t)
        for i in range(c):
            t[i] = N * t[i] / t_sum

        T_array = np.insert(T_array, n + 1, values=t, axis=0)
        # print(t)

        # 获取本次迭代的最大误差
        for j in range(c):
            delta_max = abs(T_array[n + 1][j] - T_array[n][j]) if delta_max < abs(
                T_array[n + 1][j] - T_array[n][j]) else delta_max

        if delta_max < error_min:
            return t.tolist()

        n += 1


if __name__ == '__main__':
    R = (143, 849, 3103, 5952, 5893, 3115, 945)
    print(NStoP2_1(R))
