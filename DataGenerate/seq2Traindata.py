'''
author: hzj
date: 2022-4-14
file info: 将序列数据转成可训练数据
'''
import pandas
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchtext.legacy.data import Field


class SeqDataset(Dataset):
    def __init__(
            self,
            file,
            src_length,
            tgt_length,
            src_padding,
            tgt_padding,
            tgt_start,
            tgt_end,
            src_vocab: dict,
            tgt_vocab: dict):
        '''

        :param file:
        :param src_length:
        :param tgt_length:
        :param src_padding:
        :param tgt_padding:
        :param tgt_start:
        :param tgt_end:
        :param src_vocab:
        :param tgt_vocab:
        '''
        super(SeqDataset, self).__init__()

        self.seqs = pandas.read_csv(file)
        self.src_length = src_length
        self.tgt_length = tgt_length
        self.src_padding = src_padding
        self.tgt_padding = tgt_padding
        self.tgt_start = tgt_start
        self.tgt_end = tgt_end
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        # [1:-1]是剔除字符串的第一个和最后一个字符，这里也就是剔除列表符号
        input = self.seqs.inputs[index][1:-1].split(',')
        output = self.seqs.outputs[index][1:-1].split(',')

        # 数字字符转int
        input = np.array([int(i) for i in input])
        output = np.array([int(i) for i in output])

        # 对样本进行max_length长度的补齐，这里设置为10001，output初始位设置为10002
        enc_input = np.pad(input, (0, self.src_length - len(input)), mode='constant',
                           constant_values=(0, self.src_vocab[self.src_padding]))
        dec_input = np.pad(output, (1, self.tgt_length - len(output) - 1), mode='constant',
                           constant_values=(self.tgt_vocab[self.tgt_start], self.tgt_vocab[self.tgt_padding]))
        dec_output = np.pad(output, (0, self.tgt_length - len(output) - 1), mode='constant',
                            constant_values=(0, self.tgt_vocab[self.tgt_padding]))
        dec_output = np.pad(dec_output, (0, 1), mode='constant', constant_values=(0, self.tgt_vocab[self.tgt_end]))

        return np.array([self.src_vocab[i] for i in enc_input]), np.array(
            [self.tgt_vocab[i] for i in dec_input]), np.array([self.tgt_vocab[i] for i in dec_output])

        # return [self.src_vocab[i] for i in enc_input], [self.tgt_vocab[i] for i in dec_input], [self.tgt_vocab[i] for i
        #                                                                                         in dec_output]

# 计算数据集的词汇
def statis_vocab(file, src_header, tgt_header):
    '''

    :param file: 数据集文件
    :param src_header: csv文件中src的header
    :param tgt_header: csv文件中tgt的header
    :return: src和tgt数据集对应的词汇数据（无重复）
    '''
    data = pandas.read_csv(file)
    src_vocab = []
    tgt_vocab = []
    for index in range(len(data)):
        src_vocab = src_vocab + [int(i) for i in data[src_header][index][1:-1].split(',')]
        tgt_vocab = tgt_vocab + [int(i) for i in data[tgt_header][index][1:-1].split(',')]

    return set(src_vocab), set(tgt_vocab)




if __name__ == '__main__':
    # data = pandas.read_csv('DataSet.csv')
    #
    # src_vocab = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
    #              15: 15, 16: 16, 17: 17}
    # tgt_vocab = {0: 0, 1: 1, 2: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14,
    #              18: 15, 19: 16, 20: 17, 21: 18, 22: 19, 23: 20}
    #
    # dataset = SeqDataset(file='DataSet.csv', src_length=6, tgt_length=7, src_padding=0, tgt_padding=0, tgt_start=1,
    #                      tgt_end=2, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    # # item1 = dataset.__getitem__(1)
    # # item2 = dataset.__getitem__(2)
    # dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, drop_last=False)
    # for batch_idx, batch in enumerate(dataloader):
    #     enc_input, dec_input, dec_output = batch
    #     print('enc_input', enc_input)
    #     print('dec_input', dec_input)
    #     print('dec_output', dec_output)
    src_vocab, tgt_vocab = statis_vocab('DataSet.csv', src_header='src', tgt_header='tgt')
    print(src_vocab)
    print(len(src_vocab))
    print(tgt_vocab)
    print(len(tgt_vocab))
