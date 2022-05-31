'''
author: hzj
date: 2022-4-9
file info: 搭建transformer模型
'''
import torch.nn as nn
import numpy as np
import torch
from torch import Tensor
import torch.optim as optim
import csv
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from DataGenerate.seq2Traindata import SeqDataset
from torch.utils.data import DataLoader
import pandas
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from torch.nn.utils.rnn import pack_padded_sequence


class NoamLR(object):
    def __init__(self, optimizer, warmup_steps, d_model, last_epoch=-1):
        super(NoamLR, self).__init__()

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_count = 0

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        self.last_epoch = last_epoch
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

    def get_lr(self):
        # step_num = self.step_count + 1
        # warmupLR
        # return [
        #     base_lr * self.warmup_steps ** 0.5 * min(self.step_count ** -0.5,
        #                                              self.step_count * self.warmup_steps ** -1.5)
        #     for base_lr in self.base_lrs]
        # NoamLR
        return [
            base_lr * self.d_model ** -0.5 * min(self.step_count ** -0.5, self.step_count * self.warmup_steps ** -1.5)
            for base_lr in self.base_lrs]

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


class NegativeSurvey_Transformer(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1
    ):
        super(NegativeSurvey_Transformer, self).__init__()
        # 初始化transformer模型
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # 词嵌入对象以及位置embedding
        self.src_word_embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=d_model,
            padding_idx=src_vocab[SRC.pad_token]
        )
        # self.src_position_embedding = nn.Embedding.from_pretrained()
        self.trg_word_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=d_model,
            padding_idx=tgt_vocab[TGT.pad_token]
        )
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        # 线性层
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, src: Tensor, tgt: Tensor):
        '''

        :param inputs: [S, N]
        :param outputs: [T, N]
        :param src_padding_index: 编码序列的padding符号表示下标
        :param tgt_padding_index: 解码序列的padding符号表示下标
        :return:
        '''
        src_len, N = src.shape
        tgt_len, N = tgt.shape

        # 生成输入输出编码
        src_word_emb = self.src_word_embedding(src)
        tgt_word_emb = self.trg_word_embedding(tgt)
        src_word_emb = self.dropout(src_word_emb)
        tgt_word_emb = self.dropout(tgt_word_emb)
        src_position = self.PositionEncoding(src_len, self.d_model).unsqueeze(1)
        tgt_position = self.PositionEncoding(tgt_len, self.d_model).unsqueeze(1)
        encoding_inputs = src_word_emb + src_position
        encoding_outputs = tgt_word_emb + tgt_position
        # 因为pytorch的transformer模型要求输入的src和tgt维度为[S, N, E]/[T, N, E]，todo：传统RNN的每一步要输入每个样例的一个单词的影响？
        # encoding_inputs = encoding_inputs.transpose(0, 1)
        # encoding_outputs = encoding_outputs.transpose(0, 1)
        # encoding_inputs = self.dropout(encoding_inputs)
        # encoding_outputs = self.dropout(encoding_outputs)
        # src_pad_index = src_vocab[SRC.pad_token]
        # tgt_pad_index = tgt_vocab[TGT.pad_token]
        src_key_padding_mask = self.generate_key_padding_mask(src, pad_index=src_vocab[SRC.pad_token])
        tgt_key_padding_mask = self.generate_key_padding_mask(tgt, pad_index=tgt_vocab[TGT.pad_token])
        '''注意这里memory_key_padding_mask和src_key_padding_mask值相同'''
        memory_key_padding_mask = src_key_padding_mask
        memory_mask = self.generate_memory_mask(tgt_len, src_len)
        # 生成tgt_attn_mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len)
        # 对于pytorch的transformer模型，Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        out = self.transformer(
            encoding_inputs,
            encoding_outputs,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            # memory_key_padding_mask=memory_key_padding_mask,
            memory_mask=memory_mask,
            tgt_mask=tgt_mask
        )

        out = self.projection(out)
        # 转置之后tensor属性更改，语义上不再是连续的（该情况下调用view无效），所以调用contiguous方法使其语义连续
        # return out.view(-1, out.size(-1))
        return out

    @staticmethod
    # 进行位置编码
    def PositionEncoding(positions, d_model):
        def calculate_dim(position, dim):
            return position / pow(10000, 2 * (dim // 2) / d_model)

        def calculate_dim_vector(position):
            return [calculate_dim(position, dim) for dim in range(d_model)]

        position_encoding = np.array([calculate_dim_vector(position) for position in range(positions)])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.sin(position_encoding[:, 1::2])

        return torch.FloatTensor(position_encoding)

    @staticmethod
    def generate_key_padding_mask(seq_k, pad_index):
        '''

        :param seq_k: [batch_size, seq_len]
        :param pad_index: 用于padding_mask的数据下标
        :return: 生成序列k对应的key_padding_mask
        pytorch的transformer框架内部已经实现了key_padding_mask维度拓宽，
        所以之类我们输入的长度key_padding_mask维度为[N, k_len]
        '''
        # eq(pad_index) is PAD token
        # key_padding_mask = seq_k.data.eq(pad_index).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        # key_padding_mask = seq_k.data.eq(pad_index)  # [batch_size, len_k], True is masked
        # return key_padding_mask  # [batch_size, len_k]
        return seq_k.transpose(0, 1) == pad_index

    @staticmethod
    def generate_memory_mask(tgt_len, src_len):
        '''
        这里就是为负调查特定设定的memory_mask，因为我们希望某一个位置的正调查对相同位置的负调查数据是没有关注度的；
        又因为tgt增加了开始符结束符，所以在最后预测结束符的时候可以用全局信息，是不需要mask的
        :param tgt_len: 目标序列的长度
        :param src_len: 源序列长度
        :return: memory_mask: :math:(T, S)
        '''
        mask = (torch.ones(tgt_len, src_len)) == 1
        for i in range(src_len):
            mask[i, i] = False
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# 模型训练函数
def model_train_method(
        model: NegativeSurvey_Transformer,
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    NoamScheduler = NoamLR(optimizer=optimizer, warmup_steps=50, d_model=model.d_model)

    for epoch in range(epoch_num):
        # 对于warmup策略，必须要在梯度更新前进行一次step将梯度调整到最低
        NoamScheduler.step()

        loss_list = []
        for batch_idx, batch in enumerate(train_iter):
            src = batch.SRC
            tgt = batch.TGT

            output = model(
                src=src,
                tgt=tgt[:-1]
            )
            output = output.reshape(-1, output.shape[-1])
            # item = tgt[1:].view(-1)
            loss = criterion(output, tgt[1:].view(-1))
            loss_list.append(round(loss.data.item(), 4))
            print('Epoch: %04d' % (epoch + 1) + '/' + str(epoch_num), 'loss= {:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()

            # todo: 对所有参数的梯度进行规范化
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # 将每个epoch的loss损失函数值记录到loss.csv文件中
        with open('loss.csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(loss_list)

        if epoch % 10 == 0:
            valid_loss_list = []
            for batch_idx, batch in enumerate(validation_iter):
                src_valid = batch.SRC
                tgt_valid = batch.TGT

                output = model(
                    src=src_valid,
                    tgt=tgt_valid[:-1]
                )
                output = output.reshape(-1, output.shape[-1])
                loss = criterion(output, tgt_valid[1:].view(-1))
                valid_loss_list.append(round(loss.data.item(), 4))
                print('-----valid-loss = {:.6f}-----'.format(loss))

            with open('valid_loss.csv', mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(valid_loss_list)
            torch.save(model.state_dict(), '../trained_models/' + str(epoch) + 'model_5.pt')

    # torch.save(model.state_dict(), 'model.pt')


def model_train_method2(
        model: NegativeSurvey_Transformer,
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
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    NoamScheduler = NoamLR(optimizer=optimizer, warmup_steps=50, d_model=model.d_model)

    for epoch in range(epoch_num):
        # 对于warmup策略，必须要在梯度更新前进行一次step将梯度调整到最低
        NoamScheduler.step()

        for batch_idx, batch in enumerate(train_iter):
            src = batch.SRC
            tgt = batch.TGT

            dec_input = tgt[:1]
            tgt_len = len(tgt) - 1
            global output
            for i in range(tgt_len):
                output = model(
                    src=src,
                    tgt=dec_input
                )
                # predict = output.argmax(dim=-1).reshape(-1, output.shape[0])
                predict = output.argmax(dim=-1)[-1:]
                dec_input = torch.cat((dec_input, predict), dim=0)

            output = output.reshape(-1, output.shape[-1])
            # item = tgt[1:].view(-1)
            loss = criterion(output, tgt[1:].view(-1))
            print('Epoch: %04d' % (epoch + 1) + '/' + str(epoch_num), 'loss= {:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()

            # todo:对所有参数的梯度进行规范化
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), '../trained_models/' + str(epoch) + 'model.pt')

    # torch.save(model.state_dict(), 'model.pt')


def model_test_method(model: nn.Module):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter):
            src = batch.SRC
            tgt = batch.TGT

            output = model(
                src=src,
                tgt=tgt[:-1]
            )
            # predict = output.argmax(dim=-1).reshape(-1, output.shape[0])
            predict = output.argmax(dim=-1).transpose(0, 1)
            source = src.transpose(0, 1)
            target = tgt[1:].transpose(0, 1)
            for i in range(len(predict)):
                print([src_idx2word[int(x)] for x in source[i]], '->', [tgt_idx2word[int(y)] for y in target[i]], '->',
                      [tgt_idx2word[int(z)] for z in predict[i]])

            # if batch_idx > 10:
            #     break


def model_test_method2(model: nn.Module):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter):
            src = batch.SRC
            tgt = batch.TGT

            dec_input = tgt[:1]
            tgt_len = len(tgt) - 1
            global output
            for i in range(tgt_len):
                output = model(
                    src=src,
                    tgt=dec_input
                )
                # predict = output.argmax(dim=-1).reshape(-1, output.shape[0])
                predict = output.argmax(dim=-1)[-1:]
                dec_input = torch.cat((dec_input, predict), dim=0)
            target = tgt[1:].transpose(0, 1)
            output = output.argmax(dim=-1).transpose(0, 1)
            for i in range(len(target)):
                print([tgt_idx2word[int(n)] for n in target[i]], '->', [tgt_idx2word[int(m)] for m in output[i]])


if __name__ == '__main__':
    # trainedmodel-3.pt，lr=0.01，warmup_step=50，weight_decay=0
    # model = NegativeSurvey_Transformer(
    #     d_model=256,
    #     nhead=4,
    #     num_encoder_layers=3,
    #     num_decoder_layers=3,
    #     dim_feedforward=1024
    # )

    model = NegativeSurvey_Transformer(
        d_model=512,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.3
    )
    load_state_dict = torch.load('../transformer_models/100model_5.pt')
    model.load_state_dict(load_state_dict)

    # model_train_method(
    #     model=model,
    #     epoch_num=200,
    #     lr=0.1
    # )
    # named_parameters = model.named_parameters()
    # params_0 = [p for n, p in named_parameters]
    # temp = [n for n, p in named_parameters]
    #
    model_test_method(model=model)
    # model_test_method2(model=model)
