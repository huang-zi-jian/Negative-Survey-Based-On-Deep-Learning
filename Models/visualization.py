'''
author: hzj
date: 2022-5-2
file info: 可视化数据
'''
from transformerModel.mode1 import NegativeSurvey_Transformer, tgt_vocab
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def heatmap(model_file):
    with torch.no_grad():
        model = NegativeSurvey_Transformer(
            d_model=512,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            dropout=0.8
        )
        load_state_dict = torch.load(model_file)
        model.load_state_dict(load_state_dict)

        embd = model.trg_word_embedding(torch.tensor(
            [tgt_vocab['80'], tgt_vocab['90'], tgt_vocab['250']])).numpy()

        ax = sns.heatmap(embd, yticklabels=['80', '90', '250'])
        # plt.savefig("confusion_matrix.jpg")
        plt.show()


# 绘制训练集和验证集的损失值对比图像
def train_valid_loss_visual(train_loss_file, valid_loss_file):
    train_loss = pd.read_csv(train_loss_file, header=None)
    valid_loss = pd.read_csv(valid_loss_file, header=None)

    train_loss_mean = train_loss.mean(axis=1)
    valid_loss_mean = valid_loss.mean(axis=1)
    train_loss_mean = list(train_loss_mean)
    valid_loss_mean = list(valid_loss_mean)

    train_x = [i for i in range(len(train_loss_mean))]
    valid_x = [i * 10 for i in range(len(valid_loss_mean))]
    plt.plot(train_x, train_loss_mean, label='train-loss')
    plt.plot(valid_x, valid_loss_mean, label='valid-loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(["train_loss", "valid_loss"])

    plt.show()


# 绘制训练集损失函数值
def train_loss_visual(train_loss_file):
    train_loss = pd.read_csv(train_loss_file, header=None)

    train_loss_mean = train_loss.mean(axis=1)
    train_loss_mean = list(train_loss_mean)

    train_x = [i for i in range(len(train_loss_mean))]
    plt.plot(train_x, train_loss_mean, label='train-loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()


if __name__ == '__main__':
    train_valid_loss_visual('loss-model_4.csv', 'valid_loss-model_4.csv')
    # train_loss_visual('trainedmodel3-loss.csv')
    # heatmap('../trained_models/120model.pt')
