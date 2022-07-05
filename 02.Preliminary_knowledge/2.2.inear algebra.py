import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x**y
# 理解传播
# Out: (tensor([5.]), tensor([6.]), tensor([1.5000]), tensor([9.]))
