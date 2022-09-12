
图解，卷积神经网络（CNN可视化） 视频
https://www.bilibili.com/video/BV1x44y1P7s2

Convolution arithmetic
卷积原理图示范
https://github.com/vdumoulin/conv_arithmetic


Pytorch 中nn.Conv2d的参数用法 channel含义详解
https://www.cnblogs.com/L-shuai/p/15334221.html




参数原型
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
参数说明
dataset (Dataset) – 加载数据的数据集。
batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据(默认: False).
sampler (Sampler, optional) – 定义从数据集中提取样本的策略，即生成index的方式，可以顺序也可以乱序
num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
collate_fn (callable, optional) –将一个batch的数据和标签进行合并操作。
pin_memory (bool, optional) –设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)
timeout，是用来设置数据读取的超时时间的，但超过这个时间还没读取到数据的话就会报错。


卷积算法 gif动图 https://github.com/vdumoulin/conv_arithmetic
函数原型
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                    padding_mode='zeros', device=None, dtype=None)
参数说明
in_channels：输入图像中的通道数
out_channels：经过卷积运算产生的通道数
kernel_size：卷积核大小，整数或者元组类型
stride：卷积运算的步幅，整数或者元组类型，默认1
padding：边界处的填充大小，整数或者元组类型，默认0
padding_mode：填充方式，zeros、reflect、replicate、circular，默认是zeros
zeros：零填充，在张量边界全部填充0
reflect：镜像填充，以矩阵边缘为对称轴，将反方向的对称元素填充到最外围。
replicate：复制填充，使用输入边界的复制值填充张量
circular：循环填充，重复矩阵边界另一侧的元素
具体区别请见代码案例
dilation：控制点之间的距离，默认是1，如果大于1，则该运算又称为扩张卷积运算。
池化
函数原型
    torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
参数说明
kernel_size ：表示做最大池化的窗口大小，可以是单个值，也可以是tuple元组
stride ：步长，可以是单个值，也可以是tuple元组
padding ：填充，可以是单个值，也可以是tuple元组
dilation ：控制窗口中元素步幅
return_indices ：布尔类型，返回最大值位置索引
ceil_mode ：布尔类型，为True，用向上取整的方法，计算输出形状；默认是向下取整。
Pytorch 中nn.Conv2d的参数用法 channel含义详解
https://www.cnblogs.com/L-shuai/p/15334221.html

一般的RGB图片，channels 数量是 3 （红、绿、蓝）；而monochrome图片，channels 数量是 1

一般 channels 的含义是: 每个卷积层中卷积核的数量。

out_channels 经过卷积运算产生的通道数；


dropout() 防止过拟合，增强模型泛化能力
x = F.dropout(x, training=self.training) #以默认为0.5的概率对这一层进行dropout操作，为了防止过拟合
# 根据一定概率，随机将其中的一些神经元暂时丢弃，以此增加模型的泛化能力

net.train() net.eval()
训练阶段net.train() 会打开 dropout()

非训练阶段net.eval() 会关闭 dropout(); 网络层数值固定


