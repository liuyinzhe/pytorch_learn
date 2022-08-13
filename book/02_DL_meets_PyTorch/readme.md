
t(x) =m(x+2)**2

叶子节点x,自变量；t 因变量， m表示取平均值；
    t 的变化 dt; x的变化 dx;
    d 导数
    问题：当 t 发生变化 dt, 那么 dx 的变化多大
    问题可转化为 求dt/dx的导数
    
    pytoch 中提供了.backward() 自动求导
    t.backward()
    

定义为自动微分变量  #requires_grad=True

 三个属性:
 
    data:存储张量
    
    grad:梯度，就是导数数值
    
    grad_fn:记录计算图的上一步骤，如x+2  结果z，z.grad_fn 内容是AddBackward0 at 0x内存地址
    
名词：

    反向传播，根据计算图中，grad_fn 记录上一步计算的形式，进行求梯度操作

    求梯度： 高等数学中的求导运算

    梯度信息：导数数值
    
    叶子节点： 自变量x，输入

另外入门视频(主要理解过程作用，公式可以忽略)：
    5分钟学深度学习
    https://space.bilibili.com/168709400/channel/seriesdetail?sid=2430388
