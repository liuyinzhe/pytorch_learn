# 内容来自：https://courses.d2l.ai/zh-v2/assets/notebooks/chapter_preliminaries/linear-algebra.slides.html#/3
# 自己根据理解适当做添改，备注自己的注释，记录学习状态

import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x**y
# 理解传播
# Out: (tensor([5.]), tensor([6.]), tensor([1.5000]), tensor([9.]))


# 自定义列表
x = torch.arange(4)
x
# Out： tensor([0, 1, 2, 3])

# 张量的索引类似列表
x[3]
# Out：tensor(3)

# 张量的长度/形状
len(x)
# Out: 4

x.shape
# Out: torch.Size([4])

# 指定两个分量m和 n来创建一个形状为m×n的矩阵
A = torch.arange(20).reshape(5, 4)
A
# Out:
'''
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])
'''

# 矩阵的转置；根据左上右下对角线进行翻转
A.T
# Out:
'''
tensor([[ 0,  4,  8, 12, 16],
        [ 1,  5,  9, 13, 17],
        [ 2,  6, 10, 14, 18],
        [ 3,  7, 11, 15, 19]])
'''

# 如矩阵的转置（翻转）前后一样，则内容变成bool 类型
# 对等矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
# Out：
'''

tensor([[1, 2, 3],
        [2, 0, 4],
        [3, 4, 5]])
'''
B.T
'''
tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
'''
