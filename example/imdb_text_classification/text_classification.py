# -*- encoding: utf-8 -*-
"""
@Author : Sophia Xu
@File : text_classification.py
@Time : 2025/03/06 16:30:25
@Desc : 门控神经网络分类
"""

import torch.nn as nn
import torch

VOCAB_SIZE = 15000

class GCNN(nn.Module):
    def __init__(self,vocab_size = VOCAB_SIZE, embedding_dim =64, num_class = 2):
        super(GCNN, self).__init__() #   对父类进行初始化

        self.embedding_table = nn.Embedding(vocab_size, embedding_dim) # 二维张量
        nn.init.xavier_uniform(self.embedding_table.weight)

        # 构造两层门卷积网络

        # 对文本进行卷积通常用一维卷积
        # 把embedding的维度当成输入通道数，把句子长度当成信号的长度
        # 参数（输入通道数，输出通道数，kernel_size，其他
        # stride 决定了卷积核每次移动的像素（或特征）数量。
        self.conv_A_1 = nn.Conv1d(embedding_dim, 64,15,stride=7)      
        self.conv_B_1 = nn.Conv1d(embedding_dim, 64,15,stride=7) 


        self.conv_A_2 = nn.Conv1d(64, 64,15,stride=7)      
        self.conv_B_2 = nn.Conv1d(64, 64,15,stride=7) 

        self.output_linear1 = nn.Linear(64,128)
        self.output_linear2 = nn.Linear(128,num_class)


    def forward(self, word_index): 
        # 定义GCN网络的算子操作流程，基于句子单词ID输入得到分类logits输出

        # 1. 通过 word_index得到word_embedding
        # word_index_shape: batch_size, max_seq_len
        # word_index = torch.tensor([[1, 2, 3, 4, 0], [6, 5, 0, 0, 0]) # 两个句子的索引序列，例如 "hello world this is <PAD>"," "test a <PAD> <PAD> <PAD>"
        word_embedding = self.embedding_table(word_index) # [batch_size, max_seq_len, embedding_dim]

        
        # 2. 第一层1D 卷积模块
        # 三维张量传入一维卷积中，shape应该为batch_Size，通道数，信号长度，所以需要transpose 
        word_embedding = word_embedding.transpose(1,2) # [batch_size, embedding_dim, max_seq_len]

        A = self.conv_A_1(word_embedding)
        B = self.conv_B_1(word_embedding)

        # 门卷积神经网络（CNN）的特征交互操作，增强模型对输入数据的表达能力
        H = A*torch.sigmoid(B)   # [batch_size, 64, max_seq_len]

        A = self.conv_A_2(H)
        B = self.conv_B_2(H)

        H = A* torch.sigmoid(B) # [batch_size, 64, max_seq_len]


        # 3. 池化并经过全连接层
        # 池化：减少特征的维度
        # 全局分类：对三维张量进行平均池化，平均池化的作用是沿着序列长度方向（sequence_length）对特征进行聚合，计算每个特征维度的平均值。
        pool_output = torch.mean(H, dim=-1)  # [batch_size, 64]
        linear1_output = self.output_linear1(pool_output)
        logits = self.output_linear2(linear1_output) # [batch_size, 2]

        return logits 
    


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size =VOCAB_SIZE, embed_dim = 64, num_class =2):
        super(TextClassificationModel, self).__init__()

        # EmbeddingBag 之后不是三维，是二维张量，shape为[batch_size, embedding_dim]，这里已经做了平均
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, token_index):
        embedded = self.embedding(token_index)
        return self.fc(embedded)
    



