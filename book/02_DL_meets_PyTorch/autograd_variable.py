# 自动微分变量（autograd_variable）

import torch  #导入torch包

x = torch.ones((2, 2), requires_grad=True)  
# requires_grad=True # 自动微分变量，是在张量的基础上，利用grad存储每一步的计算过程，自动构建计算图 ；通过计算图，可以进行反向传播
# 反向传播，就是根据已知步骤 进行求导，确定这个步骤对结果影响的权重大小，反向传播去除影响权重较小或无关的信息
# x.data 中存储张量
# x.grad  储存 叶节点
# x.grad_fn 
'''
data：存储了Tensor，是本体的数据
grad：保存了data的梯度，本身是个Variable而非Tensor，与data形状一致
grad_fn：指向Function对象，用于反向传播的梯度计算之用
'''
# 便于 backward ,反向传播，去除对结果影响较小(权重)的计算，更加准确
print(x)

y = x + 2  #可以按照Tensor的方式进行计算
print(y.grad_fn)
#注：在新版本PyTorch中，可以用.grad_fn
# <AddBackward0 object at 0x0000019832CD3EE0>
# Add 加法操作， 反向传播也就有了对应的方法

z = y * y  #可以进行各种符合运算
print(z.grad_fn)
#<MulBackward0 object at 0x00000272447F3EE0>
# Mul 乘法操作

z = torch.mean(y * y)  #也可以进行复合运算
print(z.data) #.data属性可以返回z所包裹的tensor 张量

# backward可以实施反向传播算法，并计算所有计算图上叶子节点的导数（梯度）信息。注意，由于z和y都不是叶子节点，所以都没有梯度信息）
# 叶节点 ，计算(流程)图 最边缘的输入
z.backward() #梯度反向传播
#print(z.grad)
#print(y.grad)
print("##")
print(x.grad)



# 这里有重复了一次操作， 叶节点 其实就是计算过程中的输入
# s.mm() 这里是矩阵乘法，x参数是作为输入的

s = torch.tensor([[0.01, 0.02]], requires_grad = True) #创建一个1*2的tensor（1维向量）
x = torch.ones(2, 2, requires_grad = True) #创建一个2*2的矩阵型tensor
for i in range(10):
    s = s.mm(x)  #反复用s乘以x（矩阵乘法），注意s始终是1*2的tensor
z = torch.mean(s) #对s中的各个元素求均值，得到一个1*1的scalar（标量，即1*1张量）

z.backward() #在具有很长的依赖路径的计算图上用反向传播算法计算叶节点的梯度
print(x.grad)  #x作为叶节点可以获得这部分梯度信息
#print(s.grad)  #s不是叶节点，没有梯度信息
