import torch
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt

#Create PINN network

def create_pinn_network(input_dim=1, output_dim=2, hidden_dim=50, num_layers=3):
    """
    Args:
        input_dim: Dimension of input (1 for time)
        output_dim: Dimension of output (2 for S and I)
        hidden_dim: Number of neurons in hidden layers
        num_layers: Number of hidden layers
    
    Returns:
        A dictionary containing all network components
    """
    pinn_dict = {}
    
    # Known constants
    pinn_dict['N'] = 1000.0
    pinn_dict['S0'] = 999.0
    pinn_dict['I0'] = 1.0
    
    # Learnable SIR parameters
    pinn_dict['beta'] = torch.nn.Parameter(torch.tensor([0.3], dtype=torch.float32))
    pinn_dict['gamma'] = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
    
    # All parameters for optimizer
    pinn_dict['parameters'] = [pinn_dict['beta'], pinn_dict['gamma']]
    
    # Input layer
    W1 = torch.nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
    b1 = torch.nn.Parameter(torch.zeros(hidden_dim))
    pinn_dict['W1'] = W1
    pinn_dict['b1'] = b1
    pinn_dict['parameters'].extend([W1, b1]) 
    #extend() is a method used for adding multiple items to a list at once
    
    # Hidden layers
    pinn_dict['hidden_weights'] = []
    pinn_dict['hidden_biases'] = []
    for i in range(num_layers - 1):
        W = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        b = torch.nn.Parameter(torch.zeros(hidden_dim))
        pinn_dict['hidden_weights'].append(W)
        pinn_dict['hidden_biases'].append(b)
        pinn_dict['parameters'].extend([W, b])
    
    # OUTPUT LAYER 
    W_out = torch.nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)
    
    # Initialize biases at t=0:
    # S(0) ≈ 999 (sigmoid^-1(999/1000) ≈ 6.9)
    # I(0) ≈ 1 (sigmoid^-1(1/1000) ≈ -6.9)
    b_out_0 = 6.9  # For S
    b_out_1 = -6.9 # For I
    