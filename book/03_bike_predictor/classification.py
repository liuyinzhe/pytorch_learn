#https://github.com/swarmapytorch/book_DeepLearning_in_PyTorch_Source/blob/master/03_bike_predictor/FirstPyTorchNN.ipynb

# 这个v2 调用PyTorch现成的函数，构建序列化的神经网络(pytorch_function)

#导入需要使用的库
import numpy as np
import pandas as pd #读取csv文件的库
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os


#读取数据到内存中，rides为一个dataframe对象
data_path = 'hour.txt'
rides = pd.read_csv(data_path)


#对于类型变量的特殊处理
# season=1,2,3,4, weathersi=1,2,3, mnth= 1,2,...,12, hr=0,1, ...,23, weekday=0,1,...,6
# 经过下面的处理后，将会多出若干特征，例如，对于season变量就会有 season_1, season_2, season_3, season_4
# 这四种不同的特征。
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    #利用pandas对象，我们可以很方便地将一个类型变量属性进行one-hot编码，变成多个属性
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

# 把原有的类型变量对应的特征去掉，将一些不相关的特征去掉
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)


# 将所有的数据集分为测试集和训练集，我们以后21天数据一共21*24个数据点作为测试集，其它是训练集
test_data = data[-21*24:]
train_data = data[:-21*24]
print('训练数据：',len(train_data),'测试数据：',len(test_data))

# 将我们的数据列分为特征列和目标列

#目标列
target_fields = ['cnt', 'casual', 'registered']
features, targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# 将数据从pandas dataframe转换为numpy
X = features.values  # 三个信息 特征 target_fields = ['cnt', 'casual', 'registered']
Y = targets['cnt'].values # 数值

print(X)
print(Y)


# 4. 分类人工神经网络Neuc
# 重新构造用于分类的人工神经网络Neuc

input_size = features.shape[1]
hidden_size = 10
output_size = 2
batch_size = 128
neuc = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size), # 输入 > 隐藏层线性映射
    torch.nn.Sigmoid(), # Sigmoid激活函数，将变量映射到0,1之间
    torch.nn.Linear(hidden_size, output_size), # 隐藏 > 输出层线性映射
    torch.nn.Sigmoid(), # # Sigmoid激活函数，将变量映射到0,1之间
)
# 将损失函数定义为交叉熵
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(neuc.parameters(), lr = 0.1)

Y_labels = Y > np.mean(Y)  # numpy 判断取值 ，存储 bool  [False,True]
Y_labels = Y_labels.astype(int) # 全部转为整数 , 对应 [0,1]
Y_labels = Y_labels.reshape(-1)  # 自动变为 n行,1列 

print(Y_labels)

# 定义一个专门计算分类错误率的函数，它的基本思想是，对于预测向量predictions的每一行，
# 取最大的那个元素的下标，与标签labels中的元素做比较
def error_rate(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，labels是数据之中的正确答案"""
    predictions = np.argmax(predictions, 1)
    return 100.0 - (
      100.0 *
      np.sum( predictions == labels) /
      predictions.shape[0])

# 神经网络训练循环
losses = []
errors = []
for i in range(4000):
    # 每128个样本点被划分为一个撮
    batch_loss = []
    batch_errors = []
    # start,end,step
    for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X)+1, batch_size)):
        xx = torch.tensor(X[start:end], dtype = torch.float, requires_grad = True)
        yy = torch.tensor(Y_labels[start:end], dtype = torch.long)
        predict = neuc(xx)
        loss = cost(predict, yy)
        err = error_rate(predict.data.numpy(), yy.data.numpy())
        optimizer.zero_grad()
        '''
        optimizer.zero_grad()函数会遍历模型的所有参数，通过p.grad.detach_()方法截断反向传播的梯度流，再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。
        因为训练的过程通常使用mini-batch方法，所以如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前。
        # https://blog.csdn.net/qq_41468616/article/details/121244698
        '''
        loss.backward()
        '''
        PyTorch的反向传播(即tensor.backward())是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。
        具体来说，torch.tensor是autograd包的基础类，如果你设置tensor的requires_grads为True，就会开始跟踪这个tensor上面的所有运算，如果你做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。
        更具体地说，损失函数loss是由模型的所有权重w经过一系列运算得到的，若某个w的requires_grads为True，则w的所有上层参数（后面层的权重w）的.grad_fn属性中就保存了对应的运算，然后在使用loss.backward()后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。
        # https://blog.csdn.net/qq_41468616/article/details/121244698
        如果没有进行tensor.backward()的话，梯度值将会是None，因此loss.backward()要写在optimizer.step()之前。
        '''
        optimizer.step()  
        '''
        step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。
        # https://blog.csdn.net/qq_41468616/article/details/121244698
        '''
        batch_loss.append(loss.data.numpy())
        batch_errors.append(err)
    
    # 每隔100步输出一下损失值（loss）
    if i % 100==0:
        losses.append(np.mean(batch_loss))
        errors.append(np.mean(batch_errors))
        print(i, np.mean(batch_loss), np.mean(batch_errors))
        
# 打印输出损失值
plt.plot(np.arange(len(losses))*100,losses, label = 'Cross Entropy')
plt.plot(np.arange(len(losses))*100, np.array(errors) / float(100), label = 'Error Rate')
plt.xlabel('epoch')
plt.ylabel('Cross Entropy/Error rates')
plt.legend()
plt.show()

# 读取测试数据
targets = test_targets['cnt']
targets = targets.values.reshape([len(targets), 1])
Y_labels = targets > np.mean(Y)  # 筛选大于均值判断位置 bool
Y_labels = Y_labels.astype(int)  # 标签 就是 bool 的整数形式，0与1
Y_labels = Y_labels.reshape(-1)  # 自动将多维度列表转换为一列
x = torch.tensor(test_features.values, dtype = torch.float, requires_grad = True) # 测试值

# 打印神经网络预测的错误率
predict = neuc(x)
print(error_rate(predict.data.numpy(), Y_labels))

# 接下来，我们把预测正确的数据和错误的数据分别画出来，纵坐标分别是预测正确的概率和预测错误的概率
prob = predict.data.numpy()  # numpy 矩阵
print(prob)
'''
#     分类0         分类1 
[[9.9378467e-01 6.0273316e-03]
 [9.9998617e-01 1.3634504e-05]
 [1.0000000e+00 8.6561947e-12]
 ...
 [9.0901166e-01 9.0969697e-02]
 [1.0000000e+00 2.6941093e-08]
 [1.0000000e+00 9.8539177e-10]]
'''
rights = np.argmax(prob, 1) == Y_labels # 列
wrongs = np.argmax(prob, 1) != Y_labels
'''
numpy.argmax(a, axis=None, out=None)
此函数包括三个参数。
函数功能，返回最大值的索引；若axis=1，表明按行比较，输出每行中最大值的索引，若axis=0，则输出每列中最大值的索引。

[               b.max(axis=1)   b.argmax(axis=1)
    [1, 3, 5 ],     5，             2,
    [30,50,10],     50，            1,
    [25,15, 5],     25，            0,
]

     30,50,10   b.max(axis=0)
     1 , 1, 1   b.argmax(axis=0) 
'''

right_labels = Y_labels[rights]
wrong_labels = Y_labels[wrongs]
probs = prob[rights, :]
probs1 = prob[wrongs, :]
rightness = [probs[i, right_labels[i]] for i in range(len(right_labels))]
right_index = np.arange(len(targets))[rights]
wrongness = [probs1[i, wrong_labels[i]] for i in range(len(wrong_labels))]
wrong_index = np.arange(len(targets))[wrongs]
fig, ax = plt.subplots(figsize = (8, 6))
ax.plot(right_index, rightness, '.', label='Right')
ax.plot(wrong_index, wrongness,'o',label='Wrong')

ax.legend()
plt.ylabel('Probabilities')

dates = pd.to_datetime(rides.loc[test_features.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)

plt.show()
