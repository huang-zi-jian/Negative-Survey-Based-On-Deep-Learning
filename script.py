import numpy as np
import torch
import torch.nn as nn
import nltk
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from torchtext.legacy.datasets import Multi30k

# print(torch.version.cuda)
# print(torch.cuda.is_available())

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def tokenize_text(text):
    # 分词
    token_list = nltk.word_tokenize(text)
    # 去停用词
    filtered = [w for w in token_list if (w not in nltk.corpus.stopwords.words('english'))]
    # 词性标注，可以根据相应的词性进行过滤。这里不过滤先
    rfiltered = nltk.pos_tag(filtered)
    result = [i[0] for i in rfiltered]
    return result


def tokenize(text):
    # 分词
    # [1:-1]是剔除字符串的第一个和最后一个字符，这里也就是剔除列表符号
    return text.replace(' ', '')[1:-1].split(',')

    # 数字字符转int
    # input = np.array([int(i) for i in input])
    # output = np.array([int(i) for i in output])
    # 去停用词
    # filtered = [w for w in token_list if (w not in nltk.corpus.stopwords.words('english'))]
    # 词性标注，可以根据相应的词性进行过滤。这里不过滤先
    # rfiltered = nltk.pos_tag(filtered)
    # result = [i[0] for i in rfiltered]
    # return result


if __name__ == '__main__':



    # for batch_idx, batch in enumerate(train_iter):
    # item_input = batch.review
    # input = src.process(batch.review)
    # item_output = batch.label
    # output = tgt.process(batch.label)
    # print()

    # for data in train_data.examples:
    #     print(vars(data))
    #

    # print(len(src.vocab))
    # print(len(tgt.vocab))
    #
    # print(tgt.vocab.stoi)

    a = torch.zeros((3, 6, 10))
    b = torch.ones((3, 10))
    # b.transpose(0, 1)
    # b = b.unsqueeze(1)
    # c = b.expand((3, 6, 10))

    # d = a + b
    # print(b.size(-1))
    # print(b.shape[-1])
    # for i in b.squeeze():
    #     item = i.item()
    # lstm_output = torch.randint(low=2, high=20, size=(2, 3, 4))
    # print('lstm_output')
    # print(lstm_output)
    # H = torch.nn.Tanh()(lstm_output)
    # print('H')
    # print(H)
    # # b = c.argmax(dim=-1)
    # hidden = torch.randint(low=2, high=20, size=(2, 4, 1))
    # print('hidden')
    # print(hidden)

    # attn_weights = torch.bmm(H, hidden)
    # print('attn_weights')
    # print(attn_weights)
    # attn_weights = attn_weights.squeeze()
    # print('attn_weights')
    # print(attn_weights)
    # weights = torch.nn.Softmax(dim=-1)(attn_weights)
    # print('weights')
    # print(weights)
    # weights = weights.unsqueeze(dim=-1)
    # print('weights')
    # print(weights)
    # weights = weights.repeat(1, 1, 4)
    # print('weights')
    # print(weights)
    # atten_output = torch.mul(lstm_output, weights)
    # print('atten_output')
    # print(atten_output)
    # atten_output = atten_output.sum(dim=-1)
    # print('atten_output')
    # print(atten_output)
    # print()

    # a = torch.randint(low=2, high=10, size=(4, 2, 3))
    # print(a)
    # a = a.permute(1, 0, 2)
    # print(a)
    # a = a.permute(1, 2, 0)
    # print(a)
    # b = torch.randint(low=2, high=10, size=(4, 2))
    # print(b)
    # c = torch.mul(a, b)
    # print(c)
    c=5
    print(c/2)