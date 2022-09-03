
import torch
import numpy as np
import matplotlib.pyplot as plt 
# linspace构建0到100均匀数字 作为 时间轴 x
# 生成0到100的100个数构成的等差数列
x = torch.linspace(0.0,100.0,steps=100).type(torch.FloatTensor)
print(x)
# 构造噪音
'''
torch.randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).

返回，均值为0，方差为10的正态分布
'''
rand = torch.randn(100)*10 # *10 10份不同的随机值
# 添加噪音
y = x + rand

# 拆分训练集，

x_train = x[:-10]

x_test = x[-10:]

y_train = y[:-10]
y_test = y[-10:]

# 数据源可视化
'''


plt.figure(figsize=(10,8)) # 设定画布大小10*8 inch

# 绘图
# tensor 变量转换为numpy变量
plt.plot(x_train.data.numpy(),y_train.data.numpy(),'o')

plt.xlabel('X')
plt.ylabel('Y')
plt.show()
'''

# 训练
# 建立两个自动微分变量
# torch.rand(*sizes, out=None)→ Tensor; [0,1)之间的均匀分布
a = torch.rand(1,requires_grad=True) 
b = torch.rand(1,requires_grad=True) 

# 学习率 ，连续导数梯度下降之间步进
learning_rate = 0.0001

# 迭代
for i in range(1000):
    # 代入公式 ax+b
    predictions = a.expand_as(x_train) * x_train + b.expand_as(x_train)
    # expand_as 作用是 将a，b 张量的维度扩充到和 x_train 一样
    
    
    # x**2  ,x的2次幂(平方)
    loss= torch.mean(predictions - y_train)**2 # 通过与标签数据y比较计算误差
    print('loss:',loss)
    
    loss.backward() # 对损失函数进行梯度反转(反向传播)
    
    # 利用上一步计算中得到的a的梯度(导数)信息，更新a中的 data 数值
    a.data.add_(- learning_rate * a.grad.data )
    # 利用上一步计算中得到的b的梯度(导数)信息，更新b中的 data 数值
    b.data.add_(- learning_rate * b.grad.data )
    
    # 清空存储在a、b中的梯度(grad，导数)信息，以免在backward的过程中反复不停的累加
    a.grad.data.zero_()
    b.grad.data.zero_()
    
    
# 预测结果可视化
'''
x_data= x_train.data.numpy()
plt.figure(figsize=(10,7))

# 将拟合直线的参数 a,b显示出来, 同时绘制实际y，以及 y 直线

str1 = str(a.data.numpy()[0]) + 'x +' + str(b.data.numpy()[0])

xplot = plt.plot(x_data,y_train.data.numpy(),label ='Data', marker='o') # 实际y
yplot = plt.plot(x_data,a.data.numpy()*x_data+b.data.numpy(),label =str1,marker='o') # 绘制拟合的y 直线图  y= a.data.numpy()*x_data+b.data.numpy()

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
'''
''
# 预测

predictions = a.expand_as(x_test) * x_test + b.expand_as(x_test) # 计算模型的预测结果
print(predictions)

x_data = x_train.data.numpy() # 获得x包裹的数据
x_pred = x_test.data.numpy()
plt.figure(figsize = (10, 7)) #设定绘图窗口大小
'''
label 指定标签
marker 指定绘图形状
https://matplotlib.org/stable/api/markers_api.html?highlight=mark#module-matplotlib.markers
'''
xplot = plt.plot(x_data, y_train.data.numpy(),label ='Data',marker='o') # 绘制训练数据
str1 = str(a.data.numpy()[0]) + 'x +' + str(b.data.numpy()[0])
yplot = plt.plot(x_pred, y_test.data.numpy(),label =str1,marker='s') # 绘制测试数据
x_data = np.r_[x_data, x_test.data.numpy()]

plt.plot(x_data, a.data.numpy() * x_data + b.data.numpy())  #绘制拟合数据
plt.plot(x_pred, a.data.numpy() * x_pred + b.data.numpy(), 'o') #绘制预测数据
plt.xlabel('X') #更改坐标轴标注
plt.ylabel('Y') #更改坐标轴标注
str1 = str(a.data.numpy()[0]) + 'x +' + str(b.data.numpy()[0]) #图例信息

plt.legend() #绘制图例
plt.show()
''