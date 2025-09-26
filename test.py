import torch

input = torch.rand(24, 1)

a = torch.ones_like(input)
print(a.shape)
print(a)
