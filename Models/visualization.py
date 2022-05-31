'''
author: hzj
date: 2022-5-2
file info: 可视化数据
'''
from Models.mode1 import NegativeSurvey_Transformer, tgt_vocab
# from Models.lstmModel import Seq2Seq, tgt_vocab
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def heatmap(
        model,
        model_file
):
    with torch.no_grad():
        # model = NegativeSurvey_Transformer(
        #     d_model=512,
        #     nhead=4,
        #     num_encoder_layers=3,
        #     num_decoder_layers=3,
        #     dim_feedforward=2048,
        #     dropout=0.3
        # )
        load_state_dict = torch.load(model_file)
        model.load_state_dict(load_state_dict)

        embd = model.trg_word_embedding(torch.tensor(
            [tgt_vocab['75'], tgt_vocab['80'], tgt_vocab['180']])).numpy()
        # embd = model.decoder.embedding(torch.tensor(
        #     [tgt_vocab['75'], tgt_vocab['80'], tgt_vocab['180']])).numpy()

        x = [i for i in range(100, 151)]
        data = {}
        for i in range(100, 151):
            data[x[i - 100]] = embd[:, i]
        data = pd.DataFrame(data)
        ax = sns.heatmap(data, yticklabels=['75', '80', '180'])
        # plt.savefig("confusion_matrix.jpg")
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        plt.title('transformer')
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
    valid_x = [i * 2 for i in range(len(valid_loss_mean))]
    plt.plot(train_x, train_loss_mean, label='train-loss')
    plt.plot(valid_x, valid_loss_mean, label='valid-loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(["train_loss", "valid_loss"])
    plt.title(label='dropout=0.4,weight_decay=0.0001')
    # 将横坐标设置为整数
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

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


def NoamUp():
    x = []
    y = []
    y_1 = []
    y_2 = []

    for i in range(1, 500):
        x.append(i)
        # d_model = 512; warmup_step = 20
        y.append(512 ** -0.5 * min(i ** -0.5, i * 20 ** -1.5))
        # d_model = 512; warmup_step = 50
        y_1.append(512 ** -0.5 * min(i ** -0.5, i * 50 ** -1.5))
        # d_model = 256; warmup_step = 50
        y_2.append(256 ** -0.5 * min(i ** -0.5, i * 50 ** -1.5))

    plt.plot(x, y, label='Noamup')
    plt.plot(x, y_1, label='Noamup')
    plt.plot(x, y_2, label='Noamup')
    # plt.xlabel('epoch')
    # plt.ylabel('lr')

    plt.grid(True, linestyle="--", color='gray', linewidth='0.5', axis='y')

    plt.legend(["d_model=512, warmup_step=20", "d_model=512, warmup_step=50", "d_model=256, warmup_step=50"])

    plt.savefig('../image/figure.png', bbox_inches='tight')

    plt.show()


def WarmUp():
    x = []
    y = []
    y_1 = []
    y_2 = []

    for i in range(1, 500):
        x.append(i)
        # warmup_step = 20; lr = 0.01
        y.append(0.01 * 20 ** 0.5 * min(i ** -0.5, i * 20 ** -1.5))
        # warmup_step = 50; lr = 0.01
        y_1.append(0.01 * 50 ** 0.5 * min(i ** -0.5, i * 50 ** -1.5))
        # warmup_step = 50; lr = 0.005
        y_2.append(0.005 * 50 ** 0.5 * min(i ** -0.5, i * 50 ** -1.5))

    plt.plot(x, y, label='WarmUp')
    plt.plot(x, y_1, label='WarmUp')
    plt.plot(x, y_2, label='WarmUp')
    # plt.xlabel('epoch')
    # plt.ylabel('lr')

    plt.grid(True, linestyle="--", color='gray', linewidth='0.5', axis='y')

    plt.legend(["lr=0.01,warmup_step=20", "lr=0.01,warmup_step=50", "lr=0.005,warmup_step=50"])

    plt.savefig('../image/figure_1.png', bbox_inches='tight')

    plt.show()


def Barchart_error():
    # NStoPS = [67, 87, 83, 99, 87]

    Original_expen = [215, 167, 34, 13, 4]
    Original_normal = [9, 99, 220, 89, 6]
    Original_uniform = [91, 79, 83, 80, 90]

    NStoPS_I_expen = [199, 178, 5, 0, 56]
    NStoPS_I_normal = [5, 94, 244, 81, 3]
    NStoPS_I_uniform = [67, 87, 83, 99, 87]

    Transformer_expon = [209, 160, 36, 9, 3]
    Transformer_normal = [9, 100, 209, 89, 7]
    Transformer_uniform = [89, 80, 86, 91, 91]

    Seq2Seq_expon = [209, 170, 40, 10, 4]
    Seq2Seq_normal = [10, 88, 214, 88, 10]
    Seq2Seq_uniform = [87, 85, 87, 88, 87]
    index = ['d=Uniform', 'd=Exponential', 'd=Normal']

    NStoPS_I_error = []
    Transformer_error = []
    Seq2Seq_error = []

    N_uniform = 0
    N_expen = 0
    N_normal = 0
    T_uniform = 0
    T_expen = 0
    T_normal = 0
    S_uniform = 0
    S_expen = 0
    S_normal = 0
    for i in range(5):
        N_uniform = N_uniform + (Original_uniform[i] - NStoPS_I_uniform[i]) ** 2
        N_expen = N_expen + (Original_expen[i] - NStoPS_I_expen[i]) ** 2
        N_normal = N_normal + (Original_normal[i] - NStoPS_I_normal[i]) ** 2
        T_uniform = T_uniform + (Original_uniform[i] - Transformer_uniform[i]) ** 2
        T_expen = T_expen + (Original_expen[i] - Transformer_expon[i]) ** 2
        T_normal = T_normal + (Original_normal[i] - Transformer_normal[i]) ** 2
        S_uniform = S_uniform + (Original_uniform[i] - Seq2Seq_uniform[i]) ** 2
        S_expen = S_expen + (Original_expen[i] - Seq2Seq_expon[i]) ** 2
        S_normal = S_normal + (Original_normal[i] - Seq2Seq_normal[i]) ** 2

    NStoPS_I_error.append(N_uniform**0.5/5)
    NStoPS_I_error.append(N_expen**0.5/5)
    NStoPS_I_error.append(N_normal**0.5/5)

    Transformer_error.append(T_uniform ** 0.5 / 5)
    Transformer_error.append(T_expen ** 0.5 / 5)
    Transformer_error.append(T_normal ** 0.5 / 5)

    Seq2Seq_error.append(S_uniform ** 0.5 / 5)
    Seq2Seq_error.append(S_expen ** 0.5 / 5)
    Seq2Seq_error.append(S_normal ** 0.5 / 5)

    df = pd.DataFrame(
        {'NStoPS-I': NStoPS_I_error, 'transformer': Transformer_error, 'Seq2Seq': Seq2Seq_error},
        index=index)
    ax = df.plot.bar(rot=0)
    plt.legend(fontsize=8, loc='best')
    # plt.xlabel('Category')
    plt.ylabel('error')
    # plt.title('Normal')
    plt.show()


def Barchart():
    # NStoPS = [67, 87, 83, 99, 87]

    # Original = [215, 167, 34, 13, 4]
    Original = [9, 99, 220, 89, 6]
    # Original = [91, 79, 83, 80, 90]

    # NStoPS_I = [199, 178, 5, 0, 56]
    NStoPS_I = [5, 94, 244, 81, 3]
    # NStoPS_I = [67, 87, 83, 99, 87]

    # Transformer = [209, 160, 36, 9, 3]
    Transformer = [9, 100, 209, 89, 7]
    # Transformer = [89, 80, 86, 91, 91]

    # Seq2Seq = [209, 170, 40, 10, 4]
    Seq2Seq = [10, 88, 214, 88, 10]
    # Seq2Seq = [87, 85, 87, 88, 87]
    index = ['1', '2', '3', '4', '5']
    df = pd.DataFrame(
        {'Original Positive Survey': Original, 'NStoPS-I': NStoPS_I, 'transformer': Transformer, 'Seq2Seq': Seq2Seq},
        index=index)
    ax = df.plot.bar(rot=0, color=['#1b9e77', '#a9f971', '#fdaa48', '#2dfa48'])
    plt.legend(fontsize=6, loc='best')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Normal')
    plt.show()

if __name__ == '__main__':
    # train_valid_loss_visual('lstm_loss-lstmModel_1.csv', 'lstm_valid_loss-lstmModel_1.csv')
    # train_loss_visual('trainedmodel3-loss.csv')
    # model = NegativeSurvey_Transformer(
    #     d_model=512,
    #     nhead=4,
    #     num_encoder_layers=3,
    #     num_decoder_layers=3,
    #     dim_feedforward=2048,
    #     dropout=0.3
    # )
    # model = Seq2Seq(
    #     enc_hidden_size=2048,
    #     dec_hidden_size=2048
    # )
    # heatmap(model, '../transformer_models/100model_5.pt')

    # NoamUp()
    # WarmUp()
    # Barchart()
    Barchart_error()
