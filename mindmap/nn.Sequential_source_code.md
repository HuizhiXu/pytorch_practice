
# nn.Sequantial()


- 模块添加的顺序就是前向传播的顺序


- 添加模块


    - model = nn.Sequential(OrderedDict([('conv1',nn.Conv2d(1,20,5)),('relu1',nn.ReLU()),('conv2',nn.Conv2d(20,64,5)),('relu2',nn.ReLU()))


    - model = nn.Sequential(nn.Conv2d(1,20,5),nn.ReLU(),nn.Conv2d(20,64,5),nn.ReLU())


    - 如果传入的不是字典，那么会自动创建索引


- forward()


    - s(input)


         - input依次去过s里面的每个module，得到输出


- Sequential与ModuleList, ModuleDict的区别：


    - Sequential有前向运算功能


    - ModuleList 和ModuleDict只有存放功能





# ModuleList(Module)


- 可以理解为存放模型的列表


- 本质上还是Module，可以调用module的所有的方法


- 与用python的list增加列表有很大区别


- class MyModule(nn.Module): def __init__(self)：


    - super(MyModule, self).__init__()


    - self.linears=nn.ModuleList([nn.Linear(10,10) for i in range(10)])


    - def forward(self,x):for i,l in enumerate(self.linears):x=self.linears[i//2](x)+l(x)  return x





# ModuleDict(Module)


- 是一个字典，也是一个module


- class MyModule(nn.Module): def __init__(self)：


    - super(MyModule, self).__init__()


    - self.choices=nn.ModuleDict({'conv':nn.Conv2d(10,10,3),'pool':nn.MaxPool2d(3)})


    - def forward(self,x,choice): x =self.choices[choice](x) return x




# ParameterList(Module)


- class MyModule(nn.Module): def __init__(self)：


    - super(MyModule, self).__init__()


    - self.params=nn.ParameterList([nn.Parameter(torch.randn(10,10)) for i in range(10)])


    - def forward(self,x):for i,p in enumerate(self.params):x=self.params[i//2].mm(x)+p.mm(x)  return x




# ParameterDict(Module)


- 同理


