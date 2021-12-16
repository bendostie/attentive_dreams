#!/usr/bin/env python3
"""
Example script demonstrating molecular transformations, using logP as target.
The entire cycle - training and dreaming - is involved.
"""

import sys
import os
sys.path.append('datasets')
import yaml
import torch
import time
import numpy as np



from random import shuffle
from torch import nn



class fc_model(nn.Module):

    def __init__(self, len_max_molec1Hot, num_of_neurons_layer1,
                 num_of_neurons_layer2, num_of_neurons_layer3):
        """
        Fully Connected layers for the RNN.
        """
        super(fc_model, self).__init__()

        # Reduce dimension up to second last layer of Encoder
        self.encode_4d = nn.Sequential(
            nn.Linear(len_max_molec1Hot, num_of_neurons_layer1),
            nn.ReLU(),
            nn.Linear(num_of_neurons_layer1, num_of_neurons_layer2),
            nn.ReLU(),
            nn.Linear(num_of_neurons_layer2, num_of_neurons_layer3),
            nn.ReLU(),
            nn.Linear(num_of_neurons_layer3, 1)
        )


    def forward(self, x):
        """
        Pass through the model
        """
        # Go down to dim-4
        h1 = self.encode_4d(x)

        return h1



if __name__ == '__main__':
    # import hyperparameter and training settings from yaml
    model = fc_model(21,500,500,500)
    model.load_state_dict(torch.load("model.pt")).to('cpu')
    torch.save(model.state_dict(), "cpu_model.pt")