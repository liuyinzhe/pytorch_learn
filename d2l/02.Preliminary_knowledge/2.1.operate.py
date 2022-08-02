import torch

# 张量(tensor)表示由一个数值组成的数组，这个数组可能有多个维度
x = torch.arange(12)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])


# 行列形状
x.shape
# torch.Size([12])


# 元素个数
x.numel()
# 12


# reshape 函数,改变一个张量的形状而不改变元素数量和元素值
X = x.reshape(3, 4)
'''
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
'''


# 创建三维形状（张量），两行，每一行中有3行4列的向量 ，每个值初始化为0
torch.zeros((2, 3, 4))  # 三维，x,y,z
'''
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])
'''

# 同上，每个值初始化为1
torch.ones((2, 3, 4))
'''
tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])
'''

#  创建一个形状为（3,4）的张量。 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
torch.randn(3, 4)
'''
tensor([[ 0.2104,  1.4439, -1.3455, -0.8273],
        [ 0.8009,  0.3585, -0.2690,  1.6183],
        [-0.4611,  1.5744, -0.4882, -0.5317]])
'''

# python 列表（list）或嵌套列表，转化为张量
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
'''
tensor([[2, 1, 4, 3],
        [1, 2, 3, 4],
        [4, 3, 2, 1]])
'''


# 按元素运算,相同维度shape 的张量，相同位置的数值之间进行计算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
'''
(tensor([ 3.,  4.,  6., 10.]),
 tensor([-1.,  0.,  2.,  6.]),
 tensor([ 2.,  4.,  8., 16.]),
 tensor([0.5000, 1.0000, 2.0000, 4.0000]),
 tensor([ 1.,  4., 16., 64.]))
按按元素方式应用更多的计算
'''


# e 的x幂次方
torch.exp(x)
'''
tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])
'''

# 张量行/列连结（concatenate)

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
'''
tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]])
'''
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
'''
(tensor([[ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]])
'''

torch.cat((X, Y), dim=0) # dim 0 表示按行连结
'''
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]]),
'''

torch.cat((X, Y), dim=1) # dim 1 表示按列连结
'''
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
'''


# 逻辑判断，shape 一致的一一对应元素换成True 或False
X == Y
'''
tensor([[False,  True, False,  True],
        [False, False, False, False],
        [False, False, False, False]])
'''


# 求和
X.sum()
# tensor(66.)


# 广播机制： 就是对应位置，元素与元素进行计算操作
# 即使形状不同，我们仍然可以通过调用 广播机制 （broadcasting mechanism） 来执行按元素操作
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
'''
(tensor([[0],
         [1],
         [2]]),
 tensor([[0, 1]]))
'''

a + b
'''
tensor([[0, 1],
        [1, 2],
        [2, 3]])
'''

# 同python 列表(list)正常切片操作
# 可以用 [-1] 选择最后一个元素，可以用 [1:3] 选择第二个和第三个元素
X[-1], X[1:3]
'''
(tensor([ 8.,  9., 10., 11.]),
 tensor([[ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]]))
'''


# 指定值修改
# 通过指定索引来将元素写入矩阵
X[1, 2] = 9
print(X)
'''
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  9.,  7.],
        [ 8.,  9., 10., 11.]])
'''

# 多个元素赋值相同的值，
# 我们只需要索引所有元素，然后为它们赋值
X[0:2, :] = 12
# x[行,列] -> X[0:2#第一行-到第二行, : #全部列]

'''
tensor([[12., 12., 12., 12.],
        [12., 12., 12., 12.],
        [ 8.,  9., 10., 11.]])
'''

# 累加操作，会导致为新结果分配内存
# id(变量名) 获得变量内存地址，类似C 中的 &变量名，取地址
before = id(Y)
Y = Y + X
id(Y) == before
# False 
# 新分配地址，数组(变量)首地址就会发生变化
# torch.zeros_like() 返回一个用标量 0 填充的张量，其大小与input相同，torch.zeros_like(input) 等同于 torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)

# 解决方法用过Z[:] 保证更改发生在申请的同一段(数组)内存空间内; 使用 X[:] = X + Y 或 X += Y 来减少操作的内存开销
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
'''
id(Z): 140452400950336
id(Z): 140452400950336
'''

before = id(X)
X += Y
id(X) == before
# True


# 转换为 NumPy 张量
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)

# (numpy.ndarray, torch.Tensor)


# 将大小为1的张量转换为 Python 标量
a = torch.tensor([3.5])
a #tensor([3.5000])
a.item() # 3.5
float(a) # 3.5
int(a) # 3

