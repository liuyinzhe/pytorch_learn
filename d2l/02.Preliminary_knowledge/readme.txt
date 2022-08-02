import torch

torch.normal(mean, std, *, generator=None, out=None) → Tensor



torch.mul(a, b) # 单乘 矩阵， 1维*2维
>>> a = torch.ones(3,4)
>>> a
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
>>> b = torch.Tensor([1,2,3]).reshape((3,1))
>>> b
tensor([[1.],
        [2.],
        [3.]])
>>> torch.mul(a, b) # b的第一列与 a 第一行相乘
tensor([[1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.]])

### 
矩阵相乘有torch.mm和torch.matmul两个函数。其中前一个是针对二维矩阵，后一个是高维。当torch.mm用于大于二维时将报错

>>> a = torch.ones(3,4)
>>> b = torch.ones(4,2)
>>> torch.mm(a, b)
tensor([[4., 4.],
        [4., 4.],
        [4., 4.]])

### 

a = torch.ones(3,4)  # 2维,3行，2列
b = torch.ones(4,2) # 3维 5个(第三个维度) [4行2列的元素]

print(a)
'''
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
'''
print(b) 
'''
tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]])
'''


print(torch.matmul(a, b)) # a 的行的元素依次与b的列元素 (结果3行2列) 相乘 ，累加得4
# 第一a的 列元素总数，要大于等于b行元素总数
# 第一a的 行元素总数，要大于等于b列元素总数
# a 的行列总数乘积 要大于等于b行列总数乘积
'''
tensor([[4., 4.],
        [4., 4.],
        [4., 4.]])
'''
print(torch.matmul(b, a))
'''
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x2 and 3x4)
'''
