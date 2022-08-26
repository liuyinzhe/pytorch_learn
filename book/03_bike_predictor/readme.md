
# 对应

* 隐藏层神经元单元个数
  - 就是矩阵乘法 中用来
  ```
  定义神经网络架构，features.shape[1]个输入层单元，10个隐含层，1个输出层
  input_size = features.shape[1] #输入层单元个数
  hidden_size = 10 #隐含层单元个数
  output_size = 1 #输出层单元个数
  batch_size = 128 #每隔batch的记录数
  weights1 = torch.randn([input_size, hidden_size], dtype = torch.double,  requires_grad = True) #第一到二层权重
  biases1 = torch.randn([hidden_size], dtype = torch.double, requires_grad = True) #隐含层偏置
  weights2 = torch.randn([hidden_size, output_size], dtype = torch.double, requires_grad = True) #隐含层到输出层权重
  def neu(x):
      #计算隐含层输出
      #x为batch_size * input_size的矩阵，weights1为input_size*hidden_size矩阵，
      #biases为hidden_size向量，输出为batch_size * hidden_size矩阵    
      hidden = x.mm(weights1) + biases1.expand(x.size()[0], hidden_size)
      hidden = torch.sigmoid(hidden)

      #输入batch_size * hidden_size矩阵，mm上weights2, hidden_size*output_size矩阵，
      #输出batch_size*output_size矩阵
      output = hidden.mm(weights2)
      return output
  ```
  
  --------------------------------------------------------
