#https://github.com/swarmapytorch/book_DeepLearning_in_PyTorch_Source/blob/master/03_bike_predictor/FirstPyTorchNN.ipynb

# 这个v2 调用PyTorch现成的函数，构建序列化的神经网络(pytorch_function)

#导入需要使用的库
import numpy as np
import pandas as pd #读取csv文件的库
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os
os.chdir(r'C:\Users\Family\Desktop\PyTorch')

#读取数据到内存中，rides为一个dataframe对象
data_path = 'hour.txt'
rides = pd.read_csv(data_path)

#看看数据长什么样子
rides.head()


# a. 对于类型变量的处理


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
data.head()


# b. 对于数值类型变量进行标准化

# 调整所有的特征，标准化处理
quant_features = ['cnt', 'temp', 'hum', 'windspeed']
#quant_features = ['temp', 'hum', 'windspeed']

# 我们将每一个变量的均值和方差都存储到scaled_features变量中。后面逆过程恢复真实数值
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


# c. 将数据集进行分割
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
X = features.values
Y = targets['cnt'].values
Y = Y.astype(float)

Y = np.reshape(Y, [len(Y),1])
losses = []



#### 2. 构建神经网络并进行训练

#b. 调用PyTorch现成的函数，构建序列化的神经网络
# 定义神经网络架构，features.shape[1]个输入层单元，10个隐含层，1个输出层
input_size = features.shape[1]
hidden_size = 10 # 隐藏层
output_size = 1 # 输出层
batch_size = 128  #应对大量数据，的多次迭代循环，一次读取太大数据量会比较慢，对数据进行批处理(batch processing); 这里只输入单个批次大小，后面需要手动拆分批次
neu = torch.nn.Sequential( # 序列化构成功能，多层神经网络
    torch.nn.Linear(input_size, hidden_size), #输入-> 隐藏层 线性映射
    torch.nn.Sigmoid(),  # 隐含层 非线性sigmoid 激活函数 映射到 (0,1)的区间
    torch.nn.Linear(hidden_size, output_size), #隐藏层-> 输出层 线性映射
)
cost = torch.nn.MSELoss() # 均方误差的损失函数
optimizer = torch.optim.SGD(neu.parameters(), lr = 0.01) # 随机梯度下降算法(stochastic gradient descent)
# neu.parameters() 参数自动调参，对应权重(a)与偏置(b)  y =ax + b ，也就是手动版本中的weights,biases.weights2
# lr learnning rate 学习率



# 神经网络训练循环
losses = []
for i in range(1000):
    # 每128个样本点被划分为一个撮，在循环的时候一批一批地读取
    batch_loss = []
    # start和end分别是提取一个batch数据的起始和终止下标
    for start in range(0, len(X), batch_size): # range(起始，终止,步长)
        # 手动拆分批次,如果end(start + batch_size) 小于总长度len(X),没问题，如果大于则end大小为len(x)
        end = start + batch_size if start + batch_size < len(X) else len(X)
        xx = torch.tensor(X[start:end], dtype = torch.float, requires_grad = True)
        yy = torch.tensor(Y[start:end], dtype = torch.float, requires_grad = True)
        predict = neu(xx)
        loss = cost(predict, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    
    # 每隔100步输出一下损失值（loss）
    if i % 100==0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))


# 打印输出损失值
fig = plt.figure(figsize=(10, 7))
plt.plot(np.arange(len(losses))*100,losses, 'o-')
plt.xlabel('epoch')
plt.ylabel('MSE')

plt.show()


# 3. 测试神经网络

# 用训练好的神经网络在测试集上进行预测
targets = test_targets['cnt'] #读取测试集的cnt数值
targets = targets.values.reshape([len(targets),1]) #将数据转换成合适的tensor形式
targets = targets.astype(float) #保证数据为实数

x = torch.tensor(test_features.values, dtype = torch.float, requires_grad = True)
y = torch.tensor(targets, dtype = torch.float, requires_grad = True)

print(x[:10])
# 用神经网络进行预测
predict = neu(x)
predict = predict.data.numpy()

print((predict * std + mean)[:10])


# 将后21天的预测数据与真实数据画在一起并比较
# 横坐标轴是不同的日期，纵坐标轴是预测或者真实数据的值
fig, ax = plt.subplots(figsize = (10, 7))

mean, std = scaled_features['cnt'] # 数值变量标准化过的数据恢复：(data[each] - mean)/std 逆过程，恢复数值
ax.plot(predict * std + mean, label='Prediction', linestyle = '--')
ax.plot(targets * std + mean, label='Data', linestyle = '-')
ax.legend()
ax.set_xlabel('Date-time')
ax.set_ylabel('Counts')
# 对横坐标轴进行标注
dates = pd.to_datetime(rides.loc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])  # start:end:step
print(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45) # rotation 角度

plt.show()



# 4. 诊断网络*

# 选出三天预测不准的日期：Dec 22，23，24
# 将这三天的数据聚集到一起，存入subset和subtargets中
bool1 = rides['dteday'] == '2012-12-22'  # bool
bool2 = rides['dteday'] == '2012-12-23'  # bool
bool3 = rides['dteday'] == '2012-12-24'  # bool

# 将三个布尔型数组求与; 求列中是是否有存在Ture的
bools = [any(tup) for tup in zip(bool1,bool2,bool3) ]

'''

a = [True, False, True]
b = [False, False, True]
c = [True, True, False]

# 作用: 求列中是是否有存在Ture的
mask = [any(tup) for tup in zip(a, b, c)]
print(mask)

for tup in zip(a,b,c):
    print(tup)
    print(any(tup)) # any(布尔型求和)

#any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True。
#元素除了是 0、空、FALSE 外都算 TRUE。

'''
''' 返回
[True, True, True] # print(mask)

(True, False, True) # print(tup)
True                # print(any(tup))
(False, False, True)
True
(True, True, False)
True
'''
# 将相应的变量取出来 ；any() 作用下，bool 相当于 三个或判断式 一起判断； subset 表示特征
subset = test_features.loc[rides[bools].index]  # rides[[rides['dteday'] == '2012-12-22' or == '2012-12-23' or == '2012-12-24']].index 返回符合标准的index
subtargets = test_targets.loc[rides[bools].index]
subtargets = subtargets['cnt']
subtargets = subtargets.values.reshape([len(subtargets),1])



def feature(X, net):
    # 定义了一个函数可以提取网络的权重信息，所有的网络参数信息全部存储在了neu的named_parameters集合中了
    X = torch.tensor(X, dtype = torch.float, requires_grad = False)
    dic = dict(net.named_parameters()) #提取出来这个集合
    weights = dic['0.weight'] #可以按照层数.名称来索引集合中的相应参数值
    biases = dic['0.bias'] #可以按照层数.名称来索引集合中的相应参数值
    h = torch.sigmoid(X.mm(weights.t()) + biases.expand([len(X), len(biases)])) # 隐含层的计算过程
    return h # 输出层的计算

# 将这几天的数据输入到神经网络中，读取出隐含层神经元的激活数值，存入results中
results = feature(subset.values, neu).data.numpy()
# 这些数据对应的预测值（输出层）
predict = neu(torch.tensor(subset.values, dtype = torch.float, requires_grad = True)).data.numpy()

#将预测值还原成原始数据的数值范围
mean, std = scaled_features['cnt']
predict = predict * std + mean
subtargets = subtargets * std + mean
# 将所有的神经元激活水平画在同一张图上，蓝色的是模型预测的数值
fig, ax = plt.subplots(figsize = (8, 6))
ax.plot(results[:,:],'.:',alpha = 0.3)
ax.plot((predict - min(predict)) / (max(predict) - min(predict)),'bs-',label='Prediction')
ax.plot((subtargets - min(predict)) / (max(predict) - min(predict)),'ro-',label='Real')
ax.plot(results[:, 3],':*',alpha=1, label='Neuro 4')

ax.set_xlim(right=len(predict)) # right #X轴上限
ax.legend()
plt.ylabel('Normalized Values')
plt.show()

fig, ax = plt.subplots(figsize = (8, 6))
dates = pd.to_datetime(rides.loc[subset.index]['dteday']) # 获取日期
dates = dates.apply(lambda d: d.strftime('%b %d')) # datas 中变量 进行 lambda 函数运算，转换为数字
ax.set_xticks(np.arange(len(dates))[12::24]) #重读中从12开始取值，间隔
_ = ax.set_xticklabels(dates[12::24], rotation=45)

# 找到了与峰值响应的神经元，把它到输入层的权重输出出来
dic = dict(neu.named_parameters())
weights = dic['2.weight']  # 输出层 1 个神经元
print('weights_all:',weights.data.numpy())
print(len(weights.data.numpy()))
print(len(weights)) # 1 个神经元
print('weights:',weights.data.numpy()[0])
'''
weights: [-4.4204097   1.4368908   3.6112711   4.535624   -0.82777774 -0.3207325
 -1.7761252  -1.2498413   2.6007059  -0.14742516]
'''
plt.plot(weights.data.numpy()[0],'o-')
plt.xlabel('Input Neurons')
plt.ylabel('Weight')

plt.show()

fig, ax = plt.subplots(figsize = (8, 6))
# for para in neu.named_parameters():
#     print(para) 

# 找到了与峰值相应的神经元，把它到输入层的权重输出出来
dic = dict(neu.named_parameters())
'''x轴长度就是输入的特征，按照顺序找到影响最大的 特征'''
weights = dic['0.weight'][7]  # 输入 10个神经元
print(weights)
plt.plot(np.arange(len(weights.data.numpy())),weights.data.numpy(),'o-')
plt.xlabel('Input Neurons')
plt.ylabel('Weight')
plt.show()


# 列出所有的features中的数据列，找到对应的编号
print(features.columns)
'''
range(len(features.columns)) 生成索引 0..10
features.columns  名字

最终输出
0 yr
1 holiday
2 temp
3 hum
'''
for (i, c) in zip(range(len(features.columns)), features.columns):
    print(i,c)

# 显示在不同日期，指定的第7个隐含层神经元细胞的激活值，以及输入层响应，看隐藏层中特定时间特征数值高对曲线影响大
fig, ax = plt.subplots(figsize = (10, 7))
ax.plot(results[:,6],label='neuron in hidden') # 6 隐含层神经元的激活数值
ax.plot(subset.values[:,33],label='neuron in input at 8am') # 33 特征值
ax.plot(subset.values[:,42],label='neuron in input at 5pm') # 42 特征值
ax.set_xlim(right=len(predict))
ax.legend()

dates = pd.to_datetime(rides.loc[subset.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.show()
