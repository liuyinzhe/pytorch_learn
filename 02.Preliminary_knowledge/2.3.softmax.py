# ref https://zhuanlan.zhihu.com/p/168562182

def softmax(x):
    return (np.exp(x) - max(x)) / np.sum(np.exp(x) - max(x), axis=0)
