"""
Attention/relational mechanism and positional encoder
"""

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

#dot product attention from Attention is All You Need, code by Yu-Hsiang Huang

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class Positional_Encoder(nn.Module):
    def __init__(self, length, device) -> None:
        super(Positional_Encoder, self).__init__()
        pos = torch.arange(length).float() / length
        self.pos = pos.unsqueeze(dim=1).to(device=device)
        
    def forward(self, x):
        batch, _, _ = x.shape
        #add and remove a dim for dimensional inference and add positional encoding
        x = torch.cat([x.unsqueeze(3), self.pos.repeat(batch, 1, 1).unsqueeze(3)], dim=2).squeeze(dim=3)
        return x




class Relational_Layer(nn.Module):
    def __init__(self, input_size, node_size, n_nodes, n_heads, residual = True ) -> None:
        super(Relational_Layer, self).__init__()
        self.n_heads = n_heads
        self.residual = residual
        self.proj_shape = (input_size, self.n_heads * node_size)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)
        
        self.k_lin = nn.Linear(node_size, n_nodes)
        self.q_lin = nn.Linear(node_size, n_nodes)
        self.a_lin = nn.Linear(n_nodes, n_nodes)

        self.node_shape = (self.n_heads, n_nodes, node_size)
        
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)

        self.linear = nn.Linear(self.n_heads * node_size, node_size)
        self.norm = nn.LayerNorm([n_nodes, node_size], elementwise_affine=False)
    

    def forward(self, x):
        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        #print("k shape {}".format(K.shape))
        K = self.k_norm(K) 
        #print("k {}".format(K.shape))
        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q) 
        #print("q {}".format(Q.shape))
        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V) 
        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K)) #D
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A,dim=3) 
        with torch.no_grad():
            self.att_map = A.clone() #E
        E = torch.einsum('bhfc,bhcd->bhfd',A,V) #F

        E = rearrange(E, 'b head n d -> b n (head d)')
        #print("e {}".format(E.shape))
        E = self.linear(E)
        E = torch.relu(E)
        if self.residual:
            E = E + x
        E = self.norm(E)
        
        #print("e2 {}".format(E.shape))
        return E