

# 所有的层的父类



# 概念


- buffer


    - buffer不是parameter，但是是一个统计的累计量，但是他是模块的一个状态


    - 例如：BatchNorm中的running_mean


    - buffer可以被保存。如果persistent为False则不保存


- Parameter 类


    - 是Tensor的子类


    - 用在module中的话，会自动加到module的parameters() 列表中





# 方法


- register_buffer(name, tensor, persistent)


    - 往模块中添加buffer


    - code


         - self.register_buffer('running_mean',torch.zeros(num_features))


- register_parameter(name, param)


    - 往模块中添加参数


    - name: name of parameter


    - param: Parameter 类or  None。注意这里不是Tensor类型


    - code:


- add_module


    - 往当前的module中添加子模块，可以.name访问到该模块


- apply(fn)


    - 递归应用到所有子模块


    - 一般应用于模块参数随机初始化


    - 例子


         - 


- bfloat16()


    - 把模块的所有的浮点类型都转换成bfloat16类型


- buffers()


    - 返回模块缓存的迭代器


         - 参数parameters会参与到梯度下降训练


         - 缓存 buffers 不会参与


    - model.buffers() ->torch.Tensor


- children()


- cpu()


- gpu() 将模块所有的参数和buffer搬到不同的设备上


- cuda(device=)


- eval()


    - 把module设置为evaluation，与train模型的区别在于dropout和batchnorm的运行逻辑


- get_parameter(target)


    - 根据字符串得到参数


    - module_path,_,param_name=target.rpartition(".)


    - mod:torch.nn.Module = self.get_submodule(module_path)


    - if hasattr(mod,param_name):  param:torch.nn.Parameter = getattr(mod,param_name)


- get_submodule(target)


    - 根据字符串得到当前模块的子模块


    - get_submodule(A.net_b.net_c)


- load_state_dict


    - 从state_dict中把当前模型的参数和buffers导入进来


- named_parameters


    - 返回参数的名称和值


- requires_grad


    - 模型是否需要参数更新，是否进行梯度下降


    - 在GAN训练中，分别训练生成器和判别器


    - 或者Bert中，只需要训练顶层，中间层或者底层不需要参数更新


- requires_grad_(requires_grad:bool)


    - 这个函数可以使module调用，也可以是Tensor调用


- zero_grad(set_to_none:bool)


    - 梯度清零


    - 因为参数的梯度计算会累积，在训练的时候，每一步都需要在优化器调用zero_grad。优化器会将模型的参数的梯度清零。


- state_dict()


    - state_dict包含模型所有的参数和所有的buffer变量


    - example


         - save and load models   保存训练好的pytorch模型


              - save


                   - torch.save({'epoch':Epoch, 'model_state_dict':net.state_dict(),'optimizer_dict':optimizer.state_dict(),'loss':LOSS,},PATH)


                        - epoch：当前训练多少周期


                        - model_state_dict：做推理的时候只需要这个，当前网络的parameters和buffers


                        - optimizer_state_dict：优化器的parameters 和Buffers


                        - loss： 当前epoch模型的loss


              - load


                   - checkpoint = torch.load(PATH)


                   - model.load_state_dict(checkpoint['model_state_dict'])


                   - optimizer.load_state_dict(checkpoint['optimizer_state_dict']  需要先初始化optimizer


         - saving and loading model weights


              - state_dict包含所有的参数和所有的buffer变量


              - 一般还需要存储优化器的状态，optimizer_state_dict，见前面


              - torch.save(model.state_dict(),'model_weights.pth')


              - torch.load_state_dict(torch.load('model_weights.pth'))


- to()


    - 转换数据类型


         - x = torch.tensor([1, 2, 3])


         - x_float = x.to(torch.float32)


    - 转换设备


         - device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


         - x = x.to(device)


- __getattr()__


    - 这个函数里面可以直接取_parameters, _buffers, _modules的值


    - moduls_init._parameters()


    - 注意，不会遍历，只获取当前模块的值。state_dict会遍历，得到所有的。


- __repr__


    - str(test_module)


         - 会打印这个模型


- dir()


    - dir(test_module)


         - 打印出所有的attribute和method


- __named_members


    - 查找函数，例如查找module以及其参数


- parameters()


    - 返回module遍历的参数


    - 和_parameters的区别：_parameters返回当前module的参数


- children()


    - 调用named_children():返回迭代器，返回每个子模块的名称和子模块本身


    - print(list(test_module.named_children()))


    - 和_modules的区别：*modules返回有序字典，named_children 返回迭代器


- modules()]


    - 调用named_modules()： 返回迭代器，返回模块


    - 和_modules的区别：_modules 只返回子模块，modules()会返回自身和子模块，所以会锁一个modul


- train(mode)


    - set the module in training mode


    - mode为True, 设置self.training为训练模式


- eval()


    - return self.train(False)


    - 设置module为推理模式


- Dropout


    - 当training设置为True的时候，他们的子模块也会设置为True和False


    - 所以Dropout这个module也会发生变化，因为它内部调用了self.training


    - 同样的，BatchNorm这个module也会发生变化


- 


