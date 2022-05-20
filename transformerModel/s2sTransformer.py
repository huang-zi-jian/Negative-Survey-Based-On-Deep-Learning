'''
author: hzj
date: 2022-4-10
file info: 搭建序列到序列的transformer模型
'''
import torch.nn as nn
import torch.nn.functional as FC


class s2sTransformer(nn.Module):
    def __init__(self, vocab_size):
        super(s2sTransformer, self).__init__()

        # Preprocess数据预处理
        # self.pos_encode_src =

        # Encoder模块
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
        # encoder_norm = nn.LayerNorm(normalized_shape=512)
        # self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6, norm=encoder_norm)
        # 因为TransformerEncoderLayer结构中实现了norm归一化，所以不需要对最后一次encode结果进行归一化
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)

        # Decode模块
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=6)
        # TransformerDecoderLayer中不包括最后的线性层
        self.output_layer = nn.Linear(512, vocab_size)
        # 初始化模型参数
        self._reset_parameters()

    def forward(self):
        output = '...'

        # nn.Softmax(dim=)

        # position encoding

        #
        memory = self.encoder(src=, mask=, src_key_padding_mask=)

        # dim=-1表示在input张量的最后一维求softmax
        return FC.softmax(input=output, dim=-1)

    def generate_key_padding_mask(self):



    def _reset_parameters(self):
        '''
        初始化模型参数，对于多维参数使其满足uniform均匀分布
        :return:
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
