'''
author: hzj
date: 2022-4-3
file info: 负调查数据集生成器
'''

import numpy as np
import pandas as pd
import random
import csv


class DataGen:
    def __init__(self, N_min, N_max, c_min, c_max):
        '''

        :param N_min: 随机负调查参与人数的最小值
        :param N_max: ...
        :param c_min: 随机负调查选项的最小值
        :param c_max: ...
        :param distribution: 正调查的数学分布， {'distribution': 'uniform'}为均匀分布；normal表示正态分布；exponential表示指数分布；log-normal表示对数正态分布
        :return: 正调查数据集和对应的负调查数据集
        '''
        self.N_min = N_min
        self.N_max = N_max
        self.c_min = c_min
        self.c_max = c_max
        # self.distributions = {'distribution': random.choice(['uniform', 'normal', 'exponential', 'log-normal'])}

    def generate_one(self, N, c):
        # 生成正调查数据集
        InvestigationDataSet = {
            'uniform': np.random.uniform(low=-0.5, high=c - 0.5, size=N),
            'normal': np.random.normal(loc=int(c / 2), scale=random.uniform(0.7, 0.8), size=N),
            'exponential': np.random.exponential(scale=random.uniform(0.7, 0.8), size=N),
            # 'log-normal': np.random.lognormal(mean=random.uniform(1.3, 1.5), sigma=random.uniform(0.9, 1), size=N)
        }[random.choice(['uniform', 'normal', 'exponential'])]
        # InvestigationDataSet = np.random.normal(loc=c / 2, scale=scale, size=N)
        # InvestigationDataSet = np.random.uniform(low=-0.5, high=c + 0.5, size=N)

        # bins初始设置最左边界为一个无穷小的值
        bins = [-1.0e12]
        for i in range(c):
            bins.append(i + 0.5)
        # 弹出bins最右边界，设置为一个无穷大的值
        bins.pop()
        bins.append(1.0e12)

        # 设置选项序号作为下面划分区间的标签
        lables = []
        for i in range(c):
            lables.append(i)

        # 将正调查数据集进行指定区间的统计划分，lables参数默认为None，这里将lables设置为False就可以使用自然数的类型标签
        statisData = pd.cut(InvestigationDataSet, bins=bins, right=False, labels=lables)
        # temp = statisData.value_counts()

        # 初始化正调查数据集和对应的负调查数据集
        result = np.zeros((2, c), dtype=np.int32)

        for participant in statisData:
            # 生成调查选项副本
            select = lables.copy()
            # 对当前参与者进行正调查计数
            result[0, participant] += 1
            # 除去当前参与者的正调查选项
            select.remove(participant)
            # 负调查随机选择
            negative = np.random.choice(select)
            # 对当前参与者进行负调查计数
            result[1, negative] += 1

        return result.tolist()

    def save_to_file(self, file, batch_size):
        header = ['tgt', 'src']

        with open(file, 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f)

            # 把header头写入csv文件
            writer.writerow(header)

            for _ in range(batch_size):
                N = random.randint(self.N_min, self.N_max)
                c = random.randint(self.c_min, self.c_max)

                # distribution = {'distribution': random.choice(['uniform', 'normal', 'exponential', 'log-normal'])}

                data = self.generate_one(N, c)
                print(data)
                writer.writerow(data)


# 将总数据划分为训练集和测试集
def split_data(
        split_file,
        train_file,
        test_file,
        train_ratio
):
    data = pd.read_csv(split_file)
    train_data = data.sample(frac=train_ratio, random_state=1)
    test_data = data[~data.index.isin(train_data.index)]

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)


if __name__ == '__main__':
    # data_gen = DataGen(N_min=400, N_max=450, c_min=5, c_max=5)
    # data_gen.save_to_file(file='DataSet20000.csv', batch_size=20000)
    split_data(
        split_file='DataSet20000.csv',
        train_file='trainDataSet20000.csv',
        test_file='testDataSet20000.csv',
        train_ratio=0.8
    )
