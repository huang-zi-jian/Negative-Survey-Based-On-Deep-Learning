'''
author: hzj
date: 2022-5-6
file info: 搭建lstm模型
'''
import torch.nn as nn
import numpy as np
import torch
from torch import Tensor
import torch.optim as optim
import csv
import pandas
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from torch.nn.utils.rnn import pack_padded_sequence
from transformerModel.mode1 import NoamLR


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
    # init_token='<sos>',
    # eos_token='<eos>',
    pad_token='<pad>',
    unk_token='<unk>'
)

# csv里的每一列都要设置，不需要的类可以这样：(列名，None)
fields = [('TGT', TGT), ('SRC', SRC)]
train_data, validation_data, test_data = TabularDataset.splits(
    path='../DataGenerate/',
    train='DataSet10000.csv',
    validation='DataSet1000.csv',
    test='DataSet1000.csv',
    format='csv',
    skip_header=True,
    fields=fields
)

# 设置最小词频min_freq，当一个单词在数据集中出现次数小于min_freq时会被转换为<unk>字符
SRC.build_vocab(train_data, min_freq=1)
TGT.build_vocab(train_data, min_freq=10)
# stoi返回defaultdict词表
src_vocab = SRC.vocab.stoi
tgt_vocab = TGT.vocab.stoi
src_idx2word = {src_vocab[key]: key for key in src_vocab}
tgt_idx2word = {tgt_vocab[key]: key for key in tgt_vocab}
src_vocab_size = len(SRC.vocab)
tgt_vocab_size = len(TGT.vocab)
batch_size = 64

# 使用BucketIterator迭代器处理数据集
train_iter, validation_iter, test_iter = BucketIterator.splits(
    datasets=(train_data, validation_data, test_data),
    # batch_sizes=(batch_size, batch_size, batch_size),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.SRC),
    shuffle=True
)


# class Attention(nn.Module):
#     def __init__(self):
#         super(Attention, self).__init__()


class NegativeSurvey_LSTM(nn.Module):
    def __init__(
            self,
            input_size: int = 256,
            hidden_size: int = 1024,
            num_layers: int = 1
    ):
        super(NegativeSurvey_LSTM, self).__init__()
        self.input_size = input_size
        # self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.8,
            bidirectional=True
        )

        # 词嵌入对象以及位置embedding
        self.input_word_embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=input_size,
            padding_idx=src_vocab[SRC.pad_token]
        )

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.Q = nn.Linear(2 * hidden_size, 4096, bias=False)
        self.K = nn.Linear(2 * hidden_size, 4096, bias=False)
        self.projection = nn.Linear(2 * hidden_size, tgt_vocab_size, bias=False)

        self.hidden_size = hidden_size

    def attention_net(self, lstm_out):



    def forward(self, input: Tensor):
        input_word_emb = self.input_word_embedding(input)

        hidden_0 = self.initHidden(input_word_emb.shape[1])
        # out:[src_len, batch_size, hidden_dim]
        out, hidden_n = self.lstm(input_word_emb, hidden_0)
        H = self.tanh(out)

        # [src_len, batch_size, 4096]
        Q_w = self.Q(H).permute(1, 0, 2)
        K_w = self.K(H).permute(1, 2, 0)
        # 计算权重
        weights = self.softmax(torch.bmm(Q_w, K_w))
        out = torch.bmm(weights, out.permute(1, 0, 2))

        # 如果是双向LSTM，那么需要将模型输出的维度缩小一倍
        if self.lstm.bidirectional:
            out = self.projection(out)

        return out.permute(1, 0, 2), hidden_n

    def initHidden(self, batch_size):
        if self.lstm.bidirectional:
            return (torch.rand(self.num_layers * 2, batch_size, self.hidden_size),
                    torch.rand(self.num_layers * 2, batch_size, self.hidden_size))
        else:
            return (torch.rand(self.num_layers, batch_size, self.hidden_size),
                    torch.rand(self.num_layers, batch_size, self.hidden_size))


# 模型训练函数
def model_train_method(
        model: NegativeSurvey_LSTM,
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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    NoamScheduler = WarmupLR(optimizer=optimizer, warmup_steps=50)

    for epoch in range(epoch_num):
        # 对于warmup策略，必须要在梯度更新前进行一次step将梯度调整到最低
        NoamScheduler.step()

        loss_list = []
        for batch_idx, batch in enumerate(train_iter):
            src = batch.SRC
            tgt = batch.TGT

            output, hidden_n = model(input=src)
            output = output.reshape(-1, output.shape[-1])
            # item = tgt[1:].view(-1)
            loss = criterion(output, tgt.view(-1))
            loss_list.append(round(loss.data.item(), 4))
            print('Epoch: %04d' % (epoch + 1) + '/' + str(epoch_num), 'loss= {:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) todo:对所有参数的梯度进行规范化
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

                output, hidden_n = model(input=src_valid)
                output = output.reshape(-1, output.shape[-1])
                loss = criterion(output, tgt_valid.view(-1))
                valid_loss_list.append(round(loss.data.item(), 4))
                print('-----valid-loss = {:.6f}-----'.format(loss))

            with open('lstm_valid_loss.csv', mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(valid_loss_list)
            torch.save(model.state_dict(), '../lstm_models/' + str(epoch) + 'lstmModel.pt')

    # torch.save(model.state_dict(), 'model.pt')


def model_test_method(model: nn.Module):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter):
            src = batch.SRC
            tgt = batch.TGT

            output = model(input=src)
            # predict = output.argmax(dim=-1).reshape(-1, output.shape[0])
            predict = output.argmax(dim=-1).transpose(0, 1)
            target = tgt.transpose(0, 1)
            for i in range(len(predict)):
                print([tgt_idx2word[int(n)] for n in target[i]], '->', [tgt_idx2word[int(m)] for m in predict[i]])

            # if batch_idx > 10:
            #     break


if __name__ == '__main__':
    model = NegativeSurvey_LSTM(
        input_size=256,
        num_layers=2
    )
    # load_state_dict = torch.load('../trained_models/180model.pt')
    # model.load_state_dict(load_state_dict)

    model_train_method(
        model=model,
        epoch_num=1000,
        lr=0.001
    )

    # model_test_method(model=model)
    # model_test_method2(model=model)
