# 详细请见网址 https://zhuanlan.zhihu.com/p/84812085
# 自动求导

import torch
#比如有一个函数，y=x的平方（y=x2）,在x=3的时候它的导数为6，我们通过代码来演示这样一个过程。

x=torch.tensor(3.0,requires_grad=True) # 自动可backward (过程反向求导，求计算过程的导数-梯度，提供梯度下降，确定权重和最佳取值)
y=torch.pow(x,2)

#判断x,y是否是可以求导的
print(x.requires_grad)
print(y.requires_grad)

#求导，通过backward函数来实现
y.backward()

#查看导数，也即所谓的梯度
print(x.grad)

