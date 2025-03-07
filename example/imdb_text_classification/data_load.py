# -*- encoding: utf-8 -*-
"""
@Author : Sophia Xu
@File : data_load.py
@Time : 2025/03/06 17:03:35
@Desc : 构建IMDB DataLoader
"""

import os
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
import sys

import logging
logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


from text_classification import GCNN, TextClassificationModel  

BATCH_SIZE = 64

# 构建IMDB DataLoader 
def yield_tokens(train_data_iter, tokenizer):
    for i, sample in enumerate(train_data_iter):
        label, comment = sample
        yield tokenizer(comment)

train_data_iter = IMDB(root='data', split='train')
tokenizer = get_tokenizer('basic_english')

# min_freq=20 只考虑出现次数大于20的token，少于20次的替换成<unk>
vocab = build_vocab_from_iterator(yield_tokens(train_data_iter, tokenizer), min_freq=20, special_first=["<unk>"])
vocab.set_default_index(0) # 特殊字符的索引设为0

print(f"单词表大小: {len(vocab)}")


def collate_fn(batch):
    """
    对dataloader 生成的mini_batch进行后处理，例如对每句话进行处理
    1. 将文本转化成索引，因为dataset中返回的只是token的列表，需要转化成词典的索引
    2. 将一个mini_batch中所有句子填充到相同长度，构成一个张量
    3. 对标签数据pos, neg分别转化成1和0
    输入是一个batch，根据dataset的返回get_item()来确定，一般会是(x,y)的元组，如果有batch_size个，
    那就是batch_size个(x,y)
    输出需要和dataset中返回的格式保持一致

    """
    target = []
    token_index = []
    max_length = 0

    for i,(label, comment) in enumerate(batch):

        tokens = tokenizer(comment)
        token_index.append(vocab(tokens))
        if len(tokens) > max_length:
            max_length = len(tokens)

        if label == 'pos':
            target.append(1)
        else:
            target.append(0)
        
        
    token_index = [index + [0]*(max_length - len(index)) for index in token_index] # padding,0是unk的索引
    return (torch.tensor(target).to(torch.int64), torch.tensor(token_index).to(torch.int32)) # target为one hot 转为int64


def train(train_data_loader, eval_data_loader,
          model, optimizer, num_epoch, log_step_interval,
          save_step_interval,eval_step_interval,save_path, resume = ""):
    """
    训练函数
    train_data_loader 训练集
    eval_data_loader 验证集，此处是map style的dataset。map style的数据类型就是说在dataset中写init, len和get_item函数
    num_epoch 训练多少个周期
    log_step_interval 多少步打印一次日志
    save_step_interval 多少步保存一次模型和优化器的参数
    eval_step_interval 多少次做一次inference
    save_path 模型保存的路径
    resume 是否要导入一个已经训练好的模型


    """
    start_epoch = 0
    start_step = 0

    if resume!="":
        logging.warning(f"loading from {resume}")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']

    for epoch_index in range(start_epoch, num_epoch): 
        """
        对周期进行循环
        """
        ema_loss = 0. # 指数移动平均loss来观看loss的变化，而不是用单步的loss，因为单步的loss比较震荡 explosion moving average
        num_batches = len(train_data_loader)

        for batch_index, (target, token_index) in enumerate(train_data_loader):
            """
            对data_loader进行遍历
            """
            optimizer.zero_grad() # 梯度置0
            step = num_batches*(epoch_index) + batch_index + 1 # 当前一共训练了多少步
            logits = model(token_index)
            # 二分类的交叉熵，第一个位置是二分类的概率值，第二个位置是和第一个参数shape一样的标签值，所以要改成one hot
            bce_loss = F.binary_cross_entropy(torch.sigmoid(logits), 
            F.one_hot(target, num_classes=2).to(torch.float32))
            ema_loss = 0.9* ema_loss + 0.1*bce_loss

            bce_loss.backward() # 在 PyTorch 中，梯度是自动累积的。这意味着在每次调用 backward() 方法时，计算得到的梯度会被累加到参数的 .grad 属性中，而不是直接覆盖之前的梯度。
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # 对梯度的模进行截断校正，这样运行更稳定
            optimizer.step() # 参数更新

            if step % log_step_interval == 0:
                logging.warning(f"epoch_index:{epoch_index}, batch_indexL{batch_index}, ema_loss:{ema_loss}")

            if step % save_step_interval == 0:
                os.makedirs(save_path, exist_ok = True) 
                save_file = os.path.join(save_path, f"step_{step}.pt")
                torch.save({

                    'epoch':epoch_index,
                    'step':step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.state_dict(),
                    'loss': bce_loss,

                }, save_file)
                logging.warning(f"checkpoint has been saved in {save_file}")

            if step % eval_step_interval == 0:
                logging.warning("start inference...")
                model.eval() # 不会记录反向传播值
                ema_eval_loss = 0
                total_acc_account = 0
                total_account = 0

                for eval_batch_index, (eval_target, eval_token_index) in enumerate(eval_data_loader):
                    total_account += eval_target.shape[0]
                    eval_logits = model(eval_token_index)
                    total_acc_account += (torch.argmax(eval_logits, dim=-1) == eval_target).sum().item()
                    eval_bce_loss = F.binary_cross_entropy(torch.sigmoid(eval_logits), 
                                                           F.one_hot(eval_target, num_classes=2).to(torch.float32))
                    
                    ema_eval_loss = 0.9 * ema_eval_loss + 0.1 * eval_bce_loss

                
                logging.warning(f"eval_ema_loss:{ema_eval_loss}, eval_acc:{total_acc_account/total_account}")
                model.train() # nn.module中默认已经启动了model.train()，所以第一步中不需要调用model.train()

                

    


if __name__ == "__main__":
    model = GCNN()
    # model = TextClassificationModel()

    print("模型总参数：", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data_iter = IMDB(root='data', split='train')  # Dataset类型的对象
    
    # dataloader可以接受迭代型的数据库，也可以接受map_style型的数据库
    # to_map_style_dataset将iterable dataset转为map style dataset
    # iterable dataset的缺点在于迭代完变成空，map_style可以保存样本
    # DataLoader从dataset里面取数据，拼成一个batch，但是从dataset里面传过来的每个样本的长度是不一样的，所以需要collate_fn
    train_data_loader = torch.utils.data.DataLoader(
        to_map_style_dataset(train_data_iter), batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True 
    )

    eval_data_iter = IMDB(root='data', split='test')  # Dataset类型的对象
   
    eval_data_loader = torch.utils.data.DataLoader(
        to_map_style_dataset(eval_data_iter), batch_size=8,collate_fn=collate_fn
    )


    num_epoch = 10
    log_step_interval = 20
    save_step_interval = 500
    eval_step_interval = 500 
    save_path = 'logs_imdb_text_classification'


    train(train_data_loader, eval_data_loader,
          model, optimizer, num_epoch, log_step_interval,
          save_step_interval,eval_step_interval,save_path, resume = "")


