#!/usr/bin/env python3
"""
Example script demonstrating molecular transformations, using logP as target.
The entire cycle - training and dreaming - is involved.
"""
import math
import sys
import os
from att_basic_linears import train

from utilities.relational import Positional_Encoder, Relational_Layer
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

from utilities.utils import change_str, make_dir, use_gpu
from utilities.mol_utils import edit_hot, lst_of_logP, multiple_hot_to_indices


class fc_model(nn.Module):

    def __init__(self, device, len_alphabet, largest_molecule_len, n_heads, num_of_neurons_layer1,
                 num_of_neurons_layer2, num_of_neurons_layer3):
        """
        Fully Connected layers for the RNN.
        """
        super(fc_model, self).__init__()


        self.node_size = len_alphabet + 1
        self.out_dim = 1
        self.n_nodes = largest_molecule_len
        self.n_heads = n_heads

        self.relational1 = Relational_Layer(self.node_size, self.node_size, self.n_nodes, self.n_heads)
        #self.relational2 = Relational_Layer(self.node_size, self.node_size, self.n_nodes, self.n_heads)
        #self.relational3 = Relational_Layer(self.node_size, self.node_size, self.n_nodes, self.n_heads)
        self.get_positional = Positional_Encoder(largest_molecule_len, device)
        
        
        
        self.linear1 = nn.Linear(self.node_size, num_of_neurons_layer1)
        self.linear2 = nn.Linear(num_of_neurons_layer1, num_of_neurons_layer2)
        self.linear3 =  nn.Linear(num_of_neurons_layer2, num_of_neurons_layer3)
        self.linear4 = nn.Linear(num_of_neurons_layer3, 1)


        
        

        

        
        
        
    def forward(self, x):
        """
        Pass through the model
        """ 
        
        x = self.get_positional(x)
        #print("x", x)
        x = self.relational1(x)
        #x = self.relational2(x)
        #x = self.relational3(x)
        x = x.max(dim=1)[0]
        y = self.linear1(x)
        y = torch.nn.functional.elu(y)
        y = self.linear2(y)
        y = torch.relu(y)
        y = self.linear3(y)
        y = torch.relu(y)
        y = self.linear4(y)
        
        
        return y


def load_model(file_name, args, len_alphabet, largest_molecule_len, n_heads, model_parameters):
    """Load existing model state dict from file"""

    model = fc_model(args.device, len_alphabet, largest_molecule_len, n_heads, **model_parameters).to(device=args.device)
    model.load_state_dict(torch.load(file_name))
    model.eval()
    
    return model

def eval_model(directory, args, model, data, data_prop, upperbound):
    """Calculates MAE and RMSE for model on validation set"""

    test_data = torch.tensor(data, dtype=torch.float, device=args.device)
    computed_data_prop = torch.tensor(data_prop, device=args.device)

    # add random noise to one-hot encoding with specified upperbound
    test_data_edit = edit_hot(test_data, upperbound, args.device)
    running_mae = 0
    running_mse = 0
    print(test_data_edit.size())
    # feedforward step
    for i in range(test_data_edit.size()[0]-1):
        trained_data_prop = model(test_data_edit[i].unsqueeze(0))
        trained_data_prop = trained_data_prop
        
        error = torch.abs(trained_data_prop - data_prop[i]).sum().data
        squared_error = ((trained_data_prop - data_prop[i])*(trained_data_prop - data_prop[i])).sum().data
        running_mae += error
        running_mse += squared_error


    rmse = math.sqrt(running_mse/test_data_edit.size()[0])
    mae = running_mae/test_data_edit.size()[0]
    print(mae, rmse)
    
  
    


if __name__ == '__main__':
    # import hyperparameter and training settings from yaml
    print('Start reading data file...')
    settings=yaml.load(open("settings-long.yml","r"))
    test = settings['test_model']
    plot = settings['plot_transform']
    n_heads = settings['n_heads']
    mols = settings['mols']
    file_name = settings['data_preprocess']['smiles_file']
    lr_train=settings['lr_train']
    lr_train=float(lr_train)
    lr_dream=settings['lr_dream']
    lr_dream=float(lr_dream)
    batch_size=settings['training']['batch_size']
    num_epochs = settings['training']['num_epochs']
    model_parameters = settings['model']
    #print("model params {}".format(model_parameters))
    dreaming_parameters = settings['dreaming']
    dreaming_parameters_str = '{}_{}'.format(dreaming_parameters['batch_size'],
                                             dreaming_parameters['num_epochs'])
    training_parameters = settings['training']
    training_parameters_str = '{}_{}'.format(training_parameters['num_epochs'],
                                             training_parameters['batch_size'])
    data_parameters = settings['data']
    data_parameters_str = '{}_{}'.format(data_parameters['num_train'],
                                         data_parameters['num_dream'])

    upperbound_tr = settings['upperbound_tr']
    upperbound_dr = settings['upperbound_dr']
    prop=settings['property_value']

    num_train = settings['data']['num_train']
    num_dream = settings['data']['num_dream']

    num_mol = num_train

    if num_dream > num_train:
        num_mol = num_dream

    def str_model_args(args):
        new_string = ""
        for (k) in args:
            new_string += ", " + str(args.get(k))
        return new_string
    print(type(model_parameters))

    directory = change_str('dream_results/residual_{}_{}_{}_{}/{}/{}' \
                           .format(data_parameters_str,
                                   training_parameters_str,
                                   #n_layers,
                                   str_model_args(model_parameters),
                                   n_heads,
                                   upperbound_tr,
                                   lr_train))
    make_dir(directory)

    args = use_gpu()

    # data-preprocessing
    data, prop_vals, alphabet, len_alphabet, largest_molecule_len = \
        data_loader.preprocess(10000, file_name)


    
    name = change_str(directory)+'/model.pt'
    if os.path.exists(name):
        model = load_model(name, args, len_alphabet, largest_molecule_len, n_heads, model_parameters)
        print("model: ", sum(p.numel() for p in model.parameters()))
        print('evaluating model...')
        eval_model(directory, args, model,
                   data, prop_vals, upperbound_tr)
    else:
        print("no model")
    
    







