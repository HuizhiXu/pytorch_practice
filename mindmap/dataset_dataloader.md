
# torch.utils.data.Dataset


- 作用


    - 处理单个训练样本


- 代码


    - 自定义Dataset


         - 继承自Dataset


         - 需要实现 __init__,__len__ 和__getitem__ 这三个函数


         - example


         - self.transform() 对原始数据的预处理，例如去掉标点符号，清洗数据


         - self.target_transform同理


         - self._getitem_(idx) 根据idx取出数据样本




# torch.utils.data.Dataloader


- 作用


    - 处理多个样本，将数据处理成Mini-batch的格式


    - 在每个数据周期之后，打乱多个样本，降低模型的过拟合


    - 调用multi_process


- 代码


    - 导入Dataloader


    - 参数：


         - dataset: Dataset 类的对象


         - batch_size: default = 1， 训练的时候的大小


         - shuffle：是否打乱


              - 源码：设置了用random sampler，没有设置用sequential sampler


         - sampler： 怎么对数据进行采样，可以自定义Sampler，例如想要按照顺序取样。


              - 不能同时设置shuffle和sampler


         - batch_sampler: 与shuffle, sampler, drop_last, batch_size互斥


         - num_workers: default=0 表示用主进程记载数据，取决于cpu的个数


         - pin_memory： 把tensor保存到gpu中，它不一定会提高模型训练的效率


         - drop_last: True表示丢掉最后一个批次，如果不整除的话


         - collate_fn: 聚集函数，类似于transform ，对小批次的数据进行后处理，例如padding。它的输入输出都是Batch。


              - 


    - example:


         - train_dataloader=Dataloader(training_Data, batch_size=64, shuffle=True)


         - 通过iter方法进行调用


              - train_features, train_labels= next(iter(train_dataloader))


    - 源码


         - sampler


              - RandomSampler 和SequentialSampler


                   - RandomSampler： 返回torch.randomperm()，返回一个1-n的打乱顺序的索引列表


                   - SequentialSampler：返回iter(range(len()))，返回一个有序的索引


         - batch_sampler:


              - 如果batch_size 不为None， batch_Sample为None，会实例化一个BatchSampler


                   - 通过self.sampler 取index，达到Batch_Size就返回


         - collate_fn


              - 如果collate_fn 为None， 如果self.__auto_collation不为None :


                   - utils.collate.default_collate(batch):


                        - 什么都没干


              - 如果collate_fn为None，如果self.__auto_collation为None：


                   - utils.collate.default_convert(batch)


         - get_iterator


              - 如果实现了这个函数，可以用next(iter(train_dataloader)) 遍历dataloader


              - 如果num_workers=0


                   - 调用_SingleProcessDataLoaderIter()


                   - _next_data，可以基于batch_sampler来获取数据


                        - 这个_next_data是什么时候被调用的呢？


              - _BaseDataLoaderIter


                   - 在next中调用/*next/*data函数




# torch.utils.data.Dataset


- 类型


    - map-style datasets：从磁盘中读取


    - iterable-style datasets：适用于流式计算场景


