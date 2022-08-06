# 视频推荐
# https://space.bilibili.com/168709400

# 函数涉及公式：  z = xy
# 导数，是对含有一个自变量的函数进行求导。
# 偏导数，是对含有两个自变量的函数中的一个自变量求导。
import torch
class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y):
        '''
        在forward函数中，接收包含输入的Tensor并返回包含输出的Tensor。
        ctx是环境变量，用于提供反向传播是需要的信息。可通过ctx.save_for_backward方法缓存数据。
        '''
        ctx.save_for_backward(x,y)
        z = x * y
        return z
    
    @staticmethod
    def backward(ctx, grad_z):
        '''
        grad_z = 损失函数L/z # 损失函数相对z 的梯度值
        https://www.cnblogs.com/peixu/p/13209105.html
        在微积分里面，对多元函数的参数求∂偏导数，把求得的各个参数的偏导数以向量的形式写出来，就是梯度。 例如函数f(x,y), 分别对x，y求偏导数，求得的梯度向量就是(∂f/∂x, ∂f/∂y)T，简称grad f(x,y)或者▽f(x,y)。
        '''
        x, y = ctx.save_tensors
        grad_x = grad_z * y
        grad_y = grad_z * x
        return grad_x,grad_y
    
    
'''
x相关 自动微分变量（autograd_variable）
data：存储了Tensor，是本体的数据
grad：保存了data的梯度，本身是个Variable而非Tensor，与data形状一致
grad_fn：指向Function对象，用于反向传播的梯度计算之用
'''
