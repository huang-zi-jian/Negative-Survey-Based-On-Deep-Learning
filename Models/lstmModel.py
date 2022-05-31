'''
author: hzj
date: 2022-5-18
file info: 实现基于lstm的seq2seq模型
'''
import torch
import torch.nn as nn
import random
import numpy as np
from torch import Tensor
import torch.optim as optim
import csv
import pandas
from torchtext.legacy.data import Field, TabularDataset, BucketIterator


class WarmupLR(object):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        super(WarmupLR, self).__init__()

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step_count = 0

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        self.last_epoch = last_epoch
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

    def get_lr(self):
        # step_num = self.step_count + 1
        # warmupLR
        return [
            base_lr * self.warmup_steps ** 0.5 * min(self.step_count ** -0.5,
                                                     self.step_count * self.warmup_steps ** -1.5)
            for base_lr in self.base_lrs]
        # NoamLR
        # return [
        #     base_lr * self.d_model ** -0.5 * min(self.step_count ** -0.5, self.step_count * self.warmup_steps ** -1.5)
        #     for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch == None:
            self.step_count += 1
            values = self.get_lr()

            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, lr = data
                param_group['lr'] = lr


def tokenize(text):
    return text.replace(' ', '')[1:-1].split(',')


SRC = Field(
    sequential=True,
    tokenize=tokenize,
    init_token=None,
    eos_token=None,
    pad_token='<pad>',
    unk_token='<unk>'
)
TGT = Field(
    sequential=True,
    tokenize=tokenize,
    init_token='<sos>',
    eos_token='<eos>',
    pad_token='<pad>',
    unk_token='<unk>'
)

# csv里的每一列都要设置，不需要的类可以这样：(列名，None)
fields = [('TGT', TGT), ('SRC', SRC)]
train_data, validation_data, test_data = TabularDataset.splits(
    path='../DataGenerate/',
    train='trainDataSet20000.csv',
    validation='testDataSet20000.csv',
    test='test.csv',
    format='csv',
    skip_header=True,
    fields=fields
)

# 设置最小词频min_freq，当一个单词在数据集中出现次数小于min_freq时会被转换为<unk>字符
SRC.build_vocab(train_data, min_freq=10)
TGT.build_vocab(train_data, min_freq=1)
# stoi返回defaultdict词表
src_vocab = SRC.vocab.stoi
tgt_vocab = TGT.vocab.stoi
src_idx2word = {src_vocab[key]: key for key in src_vocab}
tgt_idx2word = {tgt_vocab[key]: key for key in tgt_vocab}
src_vocab_size = len(SRC.vocab)
tgt_vocab_size = len(TGT.vocab)
batch_size = 128

# 使用BucketIterator迭代器处理数据集
train_iter, validation_iter, test_iter = BucketIterator.splits(
    datasets=(train_data, validation_data, test_data),
    # batch_sizes=(batch_size, batch_size, batch_size),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.SRC),
    shuffle=True
)


def initHidden(
        batch_size,
        num_layers,
        hidden_size,
        bidirectional: bool = True
):
    if bidirectional:
        return (torch.rand(num_layers * 2, batch_size, hidden_size),
                torch.rand(num_layers * 2, batch_size, hidden_size))
    else:
        return (torch.rand(num_layers, batch_size, hidden_size),
                torch.rand(num_layers, batch_size, hidden_size))


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            input_size,
            enc_hidden_size,
            dec_hidden_size,
            # num_layers,
            # bidirectional: bool = True
    ):
        super(Encoder, self).__init__()
        # 基于单/双方向的LSTM
        # self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(src_vocab_size, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=enc_hidden_size,
            # num_layers=num_layers,
            dropout=0.2,
            bidirectional=True
        )
        # 双向LSTM，所以拼接维度为：enc_hidden_size * 2
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = enc_hidden_size
        # self.num_layers = num_layers
        # self.bidirectional = bidirectional

    def forward(self, src):
        src_emb = self.dropout(self.embedding(src))
        # 提供细胞和隐藏层的初始状态
        hidden_0 = initHidden(
            batch_size=src.shape[1],
            num_layers=1,
            hidden_size=self.hidden_size,
            bidirectional=True
        )
        # [num_layers * num_directions, batch_size, hidden_size]
        enc_output, (h_n, c_n) = self.lstm(src_emb, hidden_0)
        s = torch.tanh(self.fc(torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)))

        return enc_output, s


class Attention(nn.Module):
    def __init__(
            self,
            enc_hidden_size,
            dec_hidden_size
    ):
        super(Attention, self).__init__()
        self.atten = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size, bias=False)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(self, s, enc_output):
        src_len = enc_output.shape[0]
        # 将s[batch_size, dec_hid_dim]拓展为s[batch_size, src_len, dec_hid_dim]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        # 将enc_output[src_len, batch_size, enc_hid_dim * 2]转置为enc_output[batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)

        # 将s和enc_output在最后一维上进行拼接，送入
        energy = self.atten(torch.cat((s, enc_output), dim=-1))
        energy = torch.tanh(energy)
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=-1)


class Decoder(nn.Module):
    def __init__(
            self,
            tgt_vocab_size,
            input_size,
            enc_hidden_size,
            dec_hidden_size,
            attention,
            # bidirectional: bool = False
    ):
        super(Decoder, self).__init__()
        # 基于单/双方向的LSTM
        # self.num_directions = 2 if bidirectional else 1

        self.attention = attention
        self.embedding = nn.Embedding(tgt_vocab_size, input_size)
        self.lstm = nn.LSTM(
            input_size=enc_hidden_size * 2 + input_size,
            hidden_size=dec_hidden_size,
            num_layers=1,
            dropout=0.2,
            bidirectional=False
        )
        self.fc_out = nn.Linear(enc_hidden_size * 2 + input_size + dec_hidden_size, tgt_vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = dec_hidden_size
        # self.bidirectional = bidirectional

    def forward(self, tgt, s, enc_output):
        # tgt = [batch_size, 1]
        tgt = tgt.unsqueeze(1)
        # tgt_emb = [1, batch_size, input_size]
        tgt_emb = self.dropout(self.embedding(tgt)).transpose(0, 1)

        # a = [batch_size, 1, src_len]
        a = self.attention(s, enc_output).unsqueeze(1)

        # enc_output = [batch_size, src_len, enc_hidden_size * 2]
        enc_output = enc_output.transpose(0, 1)
        # c = [1, batch_size, enc_hidden_size * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)

        # lstm_input = [1, batch_size, (enc_hidden_size * 2) + input_size]
        lstm_input = torch.cat((tgt_emb, c), dim=-1)

        hidden_0 = initHidden(
            batch_size=tgt.shape[0],
            num_layers=1,
            hidden_size=self.hidden_size,
            bidirectional=False
        )
        # dec_output = [1, batch_size, dec_hidden_size]
        dec_output, (h_n, c_n) = self.lstm(lstm_input, hidden_0)  # todo: 如何将GRU的思想转化到LSTM上

        tgt_emb = tgt_emb.squeeze(0)  # [batch_size, input_size]
        dec_output = dec_output.squeeze(0)  # [batch_size, dec_hidden_size]
        c = c.squeeze(0)  # [batch_size, enc_hidden_size * 2]
        # predict = [batch_size, tgt_vocab_size]
        predict = self.fc_out(torch.cat((dec_output, c, tgt_emb), dim=-1))

        return predict, h_n.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(
            self,
            enc_hidden_size,
            dec_hidden_size
    ):
        super(Seq2Seq, self).__init__()

        self.attention = Attention(
            enc_hidden_size=enc_hidden_size,
            dec_hidden_size=dec_hidden_size
        )
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            input_size=512,
            enc_hidden_size=enc_hidden_size,
            dec_hidden_size=dec_hidden_size,
        )
        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            input_size=512,
            enc_hidden_size=enc_hidden_size,
            dec_hidden_size=dec_hidden_size,
            attention=self.attention,
        )
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        '''

        :param src:
        :param tgt:
        :param teacher_forcing_ratio: teacher_forcing机制决定是预测值作为下一步输入还是实际值作为下一步输入
        :return:
        '''
        tgt_len = tgt.shape[0]
        batch_size = tgt.shape[1]
        # 初始化一个用于存储decoder输出的tensor
        outputs = torch.zeros(tgt_len, batch_size, self.tgt_vocab_size)

        enc_output, s = self.encoder(src)

        # 取第1个单词<sos>
        dec_input = tgt[0]
        for i in range(1, tgt_len):
            predict, s = self.decoder(dec_input, s, enc_output)
            # 将每一次的预测结果保存到outputs中
            outputs[i] = predict

            # 决定是否teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            # teacher forcing机制判断下一轮decoder的input是选择真实值还是上一轮的预测值
            dec_input = tgt[i] if teacher_force else predict.argmax(1)

        # 因为outputs[0]还是处于zeros初始化状态
        return outputs[1:]


# 模型训练函数
def train(
        model: Seq2Seq,
        epoch_num: int,
        lr: float
):
    model.train(mode=True)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab[TGT.pad_token])
    '''
    weight_decay是L2正则化理论中出现的概念（L2范数就是：loss+所有权重的平方开方和，这样就是为了约束参数的大小）
    参数很大，一个小的变动，都会产生很大的不同，所以加上L2范数很好地解决过拟合的问题
    pytorch中backward计算梯度，但是L2正则项不是通过backward计算的，在梯度下降的算法中，梯度=原始梯度+权重衰减系数*权重系数（结果和backward是一样的）
    '''
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    WarmupScheduler = WarmupLR(optimizer=optimizer, warmup_steps=20)

    for epoch in range(epoch_num):
        # 对于warmup策略，必须要在梯度更新前进行一次step将梯度调整到最低
        WarmupScheduler.step()

        loss_list = []
        for batch_idx, batch in enumerate(train_iter):
            src = batch.SRC
            tgt = batch.TGT

            output = model(src=src, tgt=tgt)
            output = output.reshape(-1, output.shape[-1])
            # item = tgt[1:].view(-1)
            loss = criterion(output, tgt[1:].view(-1))
            loss_list.append(round(loss.data.item(), 4))
            print('Epoch: %04d' % (epoch + 1) + '/' + str(epoch_num), 'loss= {:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # 将每个epoch的loss损失函数值记录到loss.csv文件中
        with open('lstm_loss.csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(loss_list)

        if epoch % 10 == 0:
            valid_loss_list = []
            for batch_idx, batch in enumerate(validation_iter):
                src_valid = batch.SRC
                tgt_valid = batch.TGT

                output = model(src=src_valid, tgt=tgt_valid)
                output = output.reshape(-1, output.shape[-1])
                loss = criterion(output, tgt_valid[1:].view(-1))
                valid_loss_list.append(round(loss.data.item(), 4))
                print('-----valid-loss = {:.6f}-----'.format(loss))

            with open('lstm_valid_loss.csv', mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(valid_loss_list)
            torch.save(model.state_dict(), '../lstm_models/' + str(epoch) + 'lstmModel.pt')

    # torch.save(model.state_dict(), 'model.pt')


def test(model: nn.Module):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter):
            src = batch.SRC
            tgt = batch.TGT

            output = model(src=src, tgt=tgt)
            # predict = output.argmax(dim=-1).reshape(-1, output.shape[0])
            predict = output.argmax(dim=-1).transpose(0, 1)
            source = src.transpose(0, 1)
            target = tgt[1:].transpose(0, 1)
            for i in range(len(predict)):
                print([src_idx2word[int(x)] for x in source[i]], '->', [tgt_idx2word[int(y)] for y in target[i]], '->',
                      [tgt_idx2word[int(z)] for z in predict[i]])

            # if batch_idx > 10:
            #     break


if __name__ == '__main__':
    model = Seq2Seq(
        enc_hidden_size=2048,
        dec_hidden_size=2048
    )
    load_state_dict = torch.load('../lstm_models/4lstmModel_1.pt')
    model.load_state_dict(load_state_dict)
    # train(
    #     model=model,
    #     epoch_num=200,
    #     lr=0.001
    # )
    test(model)
