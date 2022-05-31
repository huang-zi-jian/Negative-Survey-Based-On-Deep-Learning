'''
author: hzj
date: 2022-5-21
file info: 实现NStoPS
'''


def NStoPS(R):
    '''

    :param R: 负调查结果
    :return: 正调查结果T
    '''
    N = sum(R)
    c = len(R)
    return [N - (c - 1) * r for r in R]


if __name__ == '__main__':
    print(NStoPS((56, 61, 121, 105, 90)))
