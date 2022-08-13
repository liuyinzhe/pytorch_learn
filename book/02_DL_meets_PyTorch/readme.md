
t(x) =mean(x+2)**2


定义为自动微分变量  #requires_grad=True

 三个属性:
 
    data:存储张量
    
    grad:梯度，就是导数数值
    
    grad_fn:记录计算图的上一步骤，如x+2  结果z，z.grad_fn 内容是AddBackward0 at 0x内存地址
    
名词：

    反向传播，根据计算图中，grad_fn 记录上一步计算的形式，进行求梯度操作

    求梯度： 高等数学中的求导运算

    梯度信息：导数数值
