'''double checking tensor operations are as expected'''
import torch

first = torch.randn(64, 4, 300)
second = torch.randn(64, 300)
print(first.shape)
print(second.unsqueeze(1).shape)
print(torch.cat((first, second.unsqueeze(dim=1)), dim=1).shape)


x = torch.Tensor([[1,1,1], [0,0,0], [2,2,2],[3,3,3]]).unsqueeze(2)
y = torch.Tensor([5,5,5,5]).unsqueeze(1).unsqueeze(2)
print(x.shape)
print(y.shape)

z = torch.cat([x, y], dim=1)
print(z.flatten(1))