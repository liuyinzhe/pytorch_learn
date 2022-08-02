
# https://zh-v2.d2l.ai/chapter_linear-networks/linear-regression-concise.html

import numpy as np
import torch
from torch.utils import data

from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 使用iter构造Python迭代器，并使用next从迭代器中获取第一项

next(iter(data_iter))


###### 定义模型
# 这一单层被称为全连接层（fully-connected layer）， 因为它的每一个输入都通过矩阵-向量乘法得到它的每个输出。

# nn是神经网络的缩写
from torch import nn
# nn.Linear 封装好的线性回归模型， 第一个指定输入特征形状，即2，第二个指定输出特征形状
net = nn.Sequential(nn.Linear(2, 1))

# 使用weight.data和bias.data方法访问参数
# # torch.normal(mean, std, *, generator=None, out=None) → Tensor
net[0].weight.data.normal_(0, 0.01) # mean, std
# 使用替换方法normal_和fill_来重写参数值
net[0].bias.data.fill_(0)

# 计算均方误差使用的是MSELoss类，也称为平方范数。 默认情况下，它返回所有样本损失的平均值。
loss = nn.MSELoss()

# 小批量随机梯度下降算法是一种优化神经网络的标准工具， PyTorch在optim模块中实现了该算法的许多变种
# 例化一个SGD实例时，我们要指定优化的参数 （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。 小批量随机梯度下降只需要设置lr值，这里设置为0.03。

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

'''
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
'''

# 在每个迭代周期里，我们将完整遍历一次数据集（train_data）， 不停地从中获取一个小批量的输入和相应的标签
'''
通过调用net(X)生成预测并计算损失l（前向传播）。

通过进行反向传播来计算梯度。

通过调用优化器来更新模型参数。
'''
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
