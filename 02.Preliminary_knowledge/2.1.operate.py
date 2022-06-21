import torch

# 行向量 x
x = torch.arange(12)

# 行列形状
x.shape

# 元素个数
x.numel()

# 重塑行列形状
X = x.reshape(3, 4)

# 创建三维形状（张量），两行，每一行中有3行4列的向量 ，每个值初始化为0
torch.zeros((2, 3, 4))

# 同上，每个值初始化为1
torch.ones((2, 3, 4))

#  创建一个形状为（3,4）的张量。 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
torch.randn(3, 4)


# 按元素运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算

# e 的x幂次方
torch.exp(x)


# 张量连结（concatenate
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

# 逻辑判断，shape 一致的一一对应元素换成True 或False
X == Y

# 求和
X.sum()

# 广播机制： 就是对应位置，元素与元素进行计算操作
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
a + b

# 同正常切片操作
X[-1], X[1:3]
