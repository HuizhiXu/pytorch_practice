#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@Author : Huizhi Xu
@File : test_module.py
@Time : 2024/07/25 21:12:09
@Desc : 测试module的功能
'''


import torch


class TestModule(torch.nn.Module):
    def __init__(self) -> None:
        super(TestModule, self).__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 4)
        self.batchnorm = torch.nn.BatchNorm2d(4)


if __name__ == "__main__":
    test_module = TestModule()

    #调用 __getattr__方法
    print(test_module._modules)
    print(test_module._parameters) # 空的字典，因为没有对子模块进行遍历
    print(test_module._buffers) # 空的字典，因为没有对子模块进行遍历

    # 调用parameters方法，返回module的参数
    print(list(test_module.parameters()))
    

    # 调用state_dict方法，返回module的参数
    print(test_module.state_dict())
    """
    OrderedDict([('linear1.weight', tensor([[ 0.1914,  0.4562],
        [-0.6935, -0.5657],
        [-0.6182,  0.0467]])),
          ('linear1.bias', tensor([-0.6343, -0.1399,  0.6540])),
            ('linear2.weight', tensor([[-0.4073, -0.4785,  0.0793],
        [-0.3806,  0.5653, -0.3795],
        [ 0.5382,  0.1470,  0.5596],
        [-0.0288,  0.2061, -0.4967]])),
          ('linear2.bias', tensor([-0.1930, -0.1195, -0.0375,  0.3926])), 
          ('batchnorm.weight', tensor([1., 1., 1., 1.])), 
          ('batchnorm.bias', tensor([0., 0., 0., 0.])), 
          ('batchnorm.running_mean', tensor([0., 0., 0., 0.])), 
          ('batchnorm.running_var', tensor([1., 1., 1., 1.])), 
          ('batchnorm.num_batches_tracked', tensor(0))])
    """
    print(test_module.state_dict()['linear1.weight']) # tensor([[ 0.1914,  0.4562], [-0.6935, -0.5657], [-0.6182,  0.0467]])

    # weight 
    print(test_module._modules['linear1'].weight)
    print(test_module._modules['linear2'].weight.dtype) # torch.float32
    # 调用to方法，将module的参数转换为double类型
    test_module.to(torch.double)
    print(test_module._modules['linear2'].weight.dtype) # torch.float64

    # 调用named_parameters方法，返回module的参数
    print(list(test_module.named_parameters()))

    
    # 调用named_children方法，返回module的子模块
    print(list(test_module.named_children()))