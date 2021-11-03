import sys
import os
sys.path.append('datasets')
import yaml
import torch
import time
import numpy as np

from utilities import data_loader
from utilities import plot_utils
from utilities import mol_utils

from torch.utils.tensorboard import SummaryWriter
from random import shuffle
from torch import nn

from einops import rearrange

class Relational:
    def __init__(self, input_size, output_size, node_size, n_nodes, n_heads ) -> None:
        self.node_size = 19#C

        
        self.N = 21 #D
        self.n_heads = n_heads



        self.proj_shape = (19, self.n_heads * self.node_size) #E
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)
        
        self.k_lin = nn.Linear(self.node_size,self.N) #B
        self.q_lin = nn.Linear(self.node_size,self.N)
        self.a_lin = nn.Linear(self.N,self.N)

        self.node_shape = (self.n_heads, self.N,self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True) #F
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
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
        E = self.norm(E)
        E = E + x
        #print("e2 {}".format(E.shape))
        x = E