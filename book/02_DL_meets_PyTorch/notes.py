import torch  #导入torch包
print(torch.__version__)

# 创建n阶张量（n维度数组），最后两个数字分别为5行，3列，3维以上的成为矩阵

x = torch.rand(5, 3)  #产生一个5*3的tensor，随机取值
y = np.ones([5, 3]) #建立一个5*3全是1的二维数组（矩阵）
# 3阶张量 (2组，5行三列的)


#张量(矩阵)之间相乘
# matrix multiplication
torch.matmul(x,y)


# 转换
import numpy as np #导入numpy包
x_torch = torch.randn(2,3)
y_numpy = np.random.randn(2,3)

x_numpy = x_torch.numpy()
y_torch = torch.from_numpy(y_numpy)


#GPU运算
if torch.cuda.is_available():  #检测本机器上有无GPU可用
    x = x.cuda() #返回x的GPU上运算的版本
    y = y.cuda()
    print(x + y) #tensor可以在GPU上正常运算
 

if torch.cuda.is_available():  #检测本机器上有无GPU可用
    device = torch.device("cuda")          # 选择一个CUDA设备
    y = torch.ones_like(x, device=device)  # 在GPU上直接创建张量
    x = x.to(device)                       # 也可以直接加载``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # 转回到CPU上``.to``
