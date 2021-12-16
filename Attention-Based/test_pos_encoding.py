import torch
x = torch.Tensor([[0,1,0], [1,0,0], [0,0,1],[0,1,0]]).unsqueeze(0).unsqueeze(0)

print(x.shape)
B, _, H, W = x.shape


pos = torch.arange(H).float() / H

pos = pos.unsqueeze(dim=1)
pos = pos.repeat(B, 1, 1, 1)

#add and remove a dim for dimensional inference then reshape the data
x = torch.cat([x.unsqueeze(4), pos.unsqueeze(4)], dim=3).squeeze(dim=4).reshape(-1, 1, 16) #399


print("final: {}".format(x))

