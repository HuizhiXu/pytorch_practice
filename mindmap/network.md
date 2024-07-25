
# 打印model


- 看到子模块的构成


    - print(model)


- 看到参数量


    - torchsummary


         - pip install torchsummary or


         - summary(your_model, input_size=(channels, H, W))





# nn.Flatten(start_dim=1, end_dim=-1)


- 从start_dim到end_dim铺平


- 例子


    - input = torch.randn(32,1,5,5)


    - m = nn.Sequential( nn.Conv2d(1, 32,5,1,1), nn.Flatten())


    - output = m(input)


    - output.size() -> torch.Size([32,288])


- 解释


    - 输入张量 input 的形状是 (32, 1, 5, 5)：


         - 32 是批次大小（batch size）。


         - 1 是输入通道数（number of channels）。


         - 5x5 是输入特征图（feature map）的尺寸。


    - 卷积层 nn.Conv2d(1, 32, 5, 1, 1)：


         - 第一个参数 1 是输入通道数，与输入张量的通道数匹配。


         - 第二个参数 32 是输出通道数。


         - 第三个参数 5 是卷积核的大小（height 和 width）。


         - 第四个参数 1 是步长（stride），表示卷积核移动的像素数。


         - 第五个参数 1 是填充（padding），表示在输入特征图边缘添加的零填充数量。


         - 在卷积神经网络中，卷积层的输出尺寸（高度和宽度）可以通过以下公式计算得出：


              - outputsize= \frac{inputsize + 2 *padding - kernelsize}{step}  + 1


              - outputsize= \frac{1 + 2 *1 - 5}{1}  + 1 = 3


         - 使用上述参数，卷积层将对每个输入通道应用32个5x5的卷积核，得到32个特征图。由于填充为1，每个特征图的尺寸将从5x5变为3x3（因为每边添加了1个像素的填充）。


    - 展平层 nn.Flatten()：


         - 展平层默认将从第二个维度开始展平（start_dim=1），直到张量的最后一维。


         - 在这个例子中，卷积层的输出是 (32, 32, 3, 3)，因为每个通道的特征图尺寸是3x3。


         - 展平后，这些特征图将被展平成一个长向量。展平操作将 (32, 32, 3, 3) 转换为 (32, 32*3*3)





# nn.Linear(in_features,out_features,bias =True, device=None, dtype=None)


- 实例化时必须传入in_features,out_features


- bias = False y = w*b


- 属性


    - Linear.weight


    - Linear.bias





# nn.ReLU





# nn.Sequential


- an ordered Container of modules


- 把很多Module作为它的参数


- 还有其他的容器





# nn.Softmax


- example


    - m = nn.Softmax(dim=1)


    - input = torch.randn(2,3)


    - output = m(input)


- 解释：


    - 对第一维进行归一化，输入张量，得到结果


- Parameter


    - model.named_parameters()


    - 例子输出：


         - 打印linear_relu_stack.0 linear_relu_stack.2 linear_relu_stack.4 的参数， 因为relu层没有参数


