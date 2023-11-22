import torch

x = torch.randn((32,307,12))
b = torch.randn((12,307))

print(torch.mul(x,b).size())
