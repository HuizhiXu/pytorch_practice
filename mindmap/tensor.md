

# Tensor 张量


- tensor是一种data structure


- 用于encode inputs, outputs and parameters of a model


- tensor和numpy共享同一块内存，改变张量，它的numpy数组也会改变


- 创建


    - torch.tensor()


    - torch.rand((2,2)),torch.rand([2,2]),torch.rand([2,2,]) 这三者等价


    - torch.ones_like(a)


    - torch.zeros_like(b)


    - torch.rand_like(b)


- 属性


    - a.dtype


    - a.shape


    - a.device


- 操作


    - torch.cuda.is_available() ->torch.to('gpu')


    - torch.is_tensor(a)


    - torch.is_complex(a) 是否是复数类型 支持complex64 He complex128


    - torch.is_floating_point()


    - torch.is_nonzero(a) 非零标量


    - torch.numel() 返回Input中所有元素的数目，其实就是a.shape里面连乘的结果


    - torch.zeros(size:list| tuple, dtype=torch.int32)->torch.zeros([2,3]) 返回23全零的张量


    - torch.set_default_tensor_type(torch.DoubleTensor)


    - torch.arange(start=0, end, step=1)->torch.arange(5) 生成0-4的索引。


         - 返回(end-start)/step


    - torch.range (start=0,end, step)


         - 返回(end-start)/step + 1， 和torch.arange不同。


         - 返回类型为float32


    - torch.linspace() 创建一个线性的


    - torch.logsoace 创建对数的


    - torch.eye(n) 创建n维张量，对角线全为1，其他全为0


    - torch.full(size, fill_value)->torch.full([2,2], 5)


    - torch.cat(tensors, dim) concatenate


         - torch.cat([a,b], dim=1)


    - torch.chunk(tensor, chunk) 切割张量   默认dim=0


         - b = tensor([[1,1],[1,1],[1,1]])


         - torch.chunk(b, chunk=2, dim=0)-> (tensor([[1,1],[1,1]]), tensor([[1,1]])) b是3行的


    - torch.dstack torch.hstack 堆叠张量


    - torch.gather(input, dim,index)


         - b = torch.tensor([[1,2],[3,4]])


         - torch.gather(b,1, torch.tensor([[0,0],[1,0]))->tensor([[1,1],[4,3]]) 第1列的0，第二列的0，第1列的4，第0列的3


         - 如果output是3维的


              - dim = 0, out[i][j][k] = input[index[i][j][k]][j][k]


              - dim = 1, out[i][j][k] = input[i][index[i][j][k]][k]


              - dim = 2, out[i][j][k] = input[i][j][index[i][j][k]]


    - torch.reshape(Input, shape) 元素顺序不变


         - torch.reshape(b,(-1,))  变成一维


    - tensor.scatter(dim, index, src) 根据索引对输入张量的元素进行更新或复制


         - tensor.scatter_()  加下划线是Inplace 操作


              - dim = 0,  self[index[i][j][k]][j][k] = src[i][j][k]


              - dim = 1,  self[i][index[i][j][k]][j][k] = src[i][j][k]


              - dim = 0,  self[i][j][index[i][j][k]][j][k] = src[i][j][k]


         - src = torch.arange(1,11).reshape((2,5))->


         - torch.zeros(3,5).scatter_(0, index, src)


         - tensor.scatter_scatter_()  是+=


    - tensor.split(tensor,split_size_or_sections)


         - 和chunk的区别


              - chunk只能均分


              - 可以均分，也可以根据split_size_or_sections分


              - torch.split(a,[1,4])


    - tensor.squeeze(input, dim) 对多余的维度进行压缩


         - 将Input维度为1的维度移除掉


         - b = torch.reshape(b,[3,1,2])


         - torch.squeeze(b) 这时候b变成3乘2的张量


    - torch.stack(tensors, )


         - 将tensor堆叠起来


         - cat 直接连起来， stack的维度是有扩充的


    - torch.take(input: Tensor,index:longTensor)


         - 把input铺平，变成一维张量，然后取值


         - a = torch.Tensor([[4,3,5],[6,7,8]])


         - torch.take(a, torch.tensor([0,2,5])-> tensor([4,5,6])


    - torch.tile(input, dims)


         - 重复input里面的元素来创建新的张量。


         - dims是一个元组，指定了每个维度上要重复的数


         - 如果dims维度比Input要少，那么前面的维度默认填充1


         - x = torch.tensor([1,2,3])


         - x.tile((2,) -> tensor([1,2,3,1,2,3]) 表示第0维重复两次


         - x = torch.tensor([[1,2],[3,4]])


         - x.tile((2,1))-> tensor([[1,2],[3,4],[1,2],[3,4]])


         - x = torch.tensor([[1,2],[3,4]])


         - x.tile((1,3))-> tensor([[1,2,1,2,1,2],[3,4,3,4,3,4]])


    - torch.transpose(tensors,dim0,dim1 )


         - torch.transpose(x,0,1)-> (23)变成(32)


    - torch.unbind(input, dim)


         - removes a tensor dimension


         - a = torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])


         - torch.unbind(a, dim=0)-> (tensor([1,1,1], tensor([2,2,2],...)


         - torch.unbind(a, dim=1)-> (tensor([1,2,3,4], tensor([1,2,3,4],...)


    - torch.unsqueeze(x,1) 新增一个维度


         - x = torch.tensor([1,2,3,4]) 4维


         - torch.unsqueeze(x,0)-> tensor([[1,2,3,4]]) 变成(1,4)


         - torch.unsqueeze(x,1)-> tensor([[1],[2],[3],[4]])  变成(4,1)


    - torch.where(condition,x,y)


         - condition满足，返回x。不满足，返回y


         - x = torch.randn(3,2)  y = torch.ones(3,2)


         - torch.where(x>0,x,y)


    - torch.manuel_seed(seed) 为生成随机数设置种子


    - torch.bernoulli(input)  input 是一个概率，基于概率生成0或1


         - a = torch.empty(3,3).uniform_(0,1)


         - torch.bernoulli(a)->torch([[1,1,0],[0,0,0][1,1,1]])


    - torch.normal() 高斯分布


         - torch.normal(mean=torch.arange(1,11), std = torch.arange(1,5))


         - torch.normal(mean=5, std = 6)


         - torch.normal(mean=torch.arange(1,11), std = 6)


    - torch.rand() 0-1区间得到均匀的浮点数


    - torch.randint(low=0, high, size)  得到整数


         - torch.randint(3,5,(3,))-> tensor([4,3,4])


    - torch.randn(size) 均值为0，标准差为1的正态分布中取值。得到浮点数


    - torch.randperm(4)->tensor([2,1,0,3])


         - 得到一个4以内的索引，可以在数据里面用，例如得到一个10000以内的索引，再根据索引选择数


