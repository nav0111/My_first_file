
import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

#Create a PINN network
def create_pinn(input_dim=1, output_dim=2, hidden_dim=50, num_layers=3):
    """Creates a Physics_Informed Neural Network (PINN) model
    Args:
        input_dim: dimension of input layer (1 for time t)
        output_dim: dimension of output layer (S and I)
        hidden_dim: number of neurons in each hidden layer
        num_layers: number of hidden layers
    Returns: A dictionary containg all components     """

    pinn_dict = {}

    #Known initial values
    pinn_dict['N']=1000  #Total population
    pinn_dict['S0']=999  #Initial Susceptible population
    pinn_dict['I0']=1    #Initial Infected population

    #Model parameters to be learned
    pinn_dict['beta']= torch.nn.Parameter(torch.tensor([0.3], dtype=torch.float32))
    pinn_dict['gamma']= torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

    #All parameters to be optimized
    pinn_dict['params']= pinn_dict['beta'], pinn_dict['gamma']

    #Input layer
    w1 = torch.nn.Parameter(torch.randn(input_dim, hidden_dim)*0.1)
    b1 = torch.nn.Parameter(torch.zeros(hidden_dim))
    pinn_dict['w1'] = w1
    pinn_dict['b1'] = b1
    pinn_dict['params']+= (w1, b1)

    #Hidden layers
    pinn_dict['hidden_weights'] = []
    pinn_dict['hidden_biases'] = []
    for i in range(num_layers -1):
        w = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim)*0.1)
        b = torch.nn.Parameter(torch.zeros(hidden_dim))
        pinn_dict['hidden_weights'].append(w)
        pinn_dict['hidden_biases'].append(b)
        pinn_dict['params']+= (w,b)

    #Output layer
    w_out = torch.nn.Parameter(torch.randn(hidden_dim, output_dim)*0.1)
    #initialize biases at t = 0
    #S0 = 999, sigmoid^-1(999/1000) = 6.907755
    #I0 =1, sigmoid^-1(1/1000) = -6.907755
    b_out = torch.nn.Parameter(torch.tensor([6.0, -6.0]))
    pinn_dict['w_out'] = w_out
    pinn_dict['b_out'] = b_out
    pinn_dict['params'] += (w_out, b_out)
    return pinn_dict
    
    #Forward pass through the PINN
def pinn_forward(pinn_dict, t):

    """
        Forward pass with output scaled between 0 and 1 using sigmoid
        then multiplied by N to get actual S and I values
    """
    if len(t.shape) ==1:
        t = t.unsqueeze(-1)
        #changes the tensor’s shape, without changing data, 
        # #Converts a 1D tensor to a column vector

    # Input layer, matmul is matrix multiplication
    H = torch.matmul(t, pinn_dict['w1']) + pinn_dict['b1']
    H = torch.tanh(H)

    #Hidden layers
    for i in range(len(pinn_dict['hidden_weights'])):
        w = pinn_dict['hidden_weights'][i]
        b = pinn_dict['hidden_biases'][i]
        H = torch.matmul(H, w.T)+b
        H = torch.tanh(H)

    #Output layer
    output = torch.matmul(H, pinn_dict['w_out'])+ pinn_dict['b_out']
    output = torch.sigmoid(output) * pinn_dict['N'] #Scale output to populatiion N
    S = output[:,0:1]
    I = output[:,1:2]
    return S, I

#Compute derivatives dS/dt and dI/dt using autograd
def compute_derivatives(pinn_dict, t):
    """
        Args:
        pinn_dict: dictionary containing PINN components
        t: time tensor
        Returns: 
        dS_dt: derivetives of S wrt t
        dI_dt: derivatives of I wrt t
    """
    if not t.requires_grad:
        #requires enable gradients wrt time
        t = t.clone().detach().requires_grad_(True)

    S, I = pinn_forward(pinn_dict, t)

    dS_dt = grad(
        S, t,
        grad_outputs=torch.ones_like(S),
        create_graph=True
    )[0]

    dI_dt = grad(
        I, t,
        grad_outputs=torch.ones_like(I),
        create_graph=True
    )[0]

    return S, I, dS_dt, dI_dt
            
#Compute data loss
def compute_data_loss(pinn_dict, t_obs, I_obs):
                
    """
    Measures how well the network matches observed I(t) data
    Mean Squared Error (MSE) between predicted and observed I
    Args:
        pinn_dict: dictionary containing PINN parameters
        t_obs: observed time tensor
        I_obs: observed infected tensor
        Returns: data_loss: MSE loss
    """
    S_pred, I_pred = pinn_forward(pinn_dict, t_obs)
    data_loss = nn.MSELoss()(I_pred, I_obs)
    return data_loss
            
#Compute ODE loss
def compute_ode_loss(pinn_dict, t_colloc):
    """
    Loss for satisfying the SIR ODEs at collocation points
    Args:
    t_colloc: collocation time tensor
    Returns: ode_loss: MSE loss for ODE residuals
    dS_dt, dI_dt = compute_derivatives(pinn_dict, t_colloc)
    """
    S, I, dS_dt, dI_dt = compute_derivatives(pinn_dict, t_colloc)

    #get model parameters
    beta = pinn_dict['beta']
    gamma = pinn_dict['gamma']
    N = pinn_dict['N']

    #SIR ODE residuals
    res_S = dS_dt + beta * S * I /N
    res_I = dI_dt - beta * S * I/ N + gamma * I
    ode_loss = nn.MSELoss()(res_S, torch.zeros_like(res_S)) + nn.MSELoss()(res_I, torch.zeros_like(res_I))
    return ode_loss
            
#Compute ic loss
def compute_ic_loss(pinn_dict):
    """
    Loss for satisfying initial conditions at t=0
    Returns: ic_loss: MSE loss for initial conditions
    """
    S0_pred, I0_pred = pinn_forward(pinn_dict, torch.tensor([[0.0]], dtype=torch.float32))
    ic_loss = nn.MSELoss()(S0_pred, torch.tensor([[pinn_dict['S0']]], dtype=torch.float32))+ nn.MSELoss()(I0_pred, torch.tensor([[pinn_dict['I0']]], dtype=torch.float32))
    return ic_loss
            
#Compute total loss
def compute_total_loss(pinn_dict, t_obs, I_obs, t_colloc):
    """
    Combine data loss, ODE loss, and Initial conditional loss
    """
    data_loss = compute_data_loss(pinn_dict, t_obs, I_obs)
    ode_loss = compute_ode_loss(pinn_dict, t_colloc)
    ic_loss = compute_ic_loss(pinn_dict)
    total_loss = data_loss +ode_loss + ic_loss
    return total_loss, data_loss, ode_loss, ic_loss
            
#training function
def train_pinn(pinn_dict, t_obs, I_obs, t_colloc, num_epochs=5000, learning_rate=0.001):
     """
    Train the PINN model
    Args:
        pinn_dict: dictionary containing PINN parameters
        t_obs: observed time tensor
        I_obs: observed infected tensor
        t_colloc: collocation time tensor
        num_epochs: number of training epochs
        learning_rate: learning rate for optimizer
     """
     optimizer = optim.Adam(pinn_dict['params'], lr = learning_rate)
     for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss, data_loss, ode_loss, ic_loss, = compute_total_loss(pinn_dict, t_obs, I_obs, t_colloc )
        total_loss.backward()
        optimizer.step()
        if epoch % 500 ==0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4e}, Data Loss: {data_loss.item():.4e}, ODE Loss: {ode_loss.item():.4e}, IC Loss: {ic_loss.item():.4e}")

     return pinn_dict
                    
# Make predictions with trained PINN
def predict_pinn(pinn_dict, t):
    """
    Make predictions using the trained PINN model
    """ 
    S_pred, I_pred = pinn_forward(pinn_dict, t)
    return S_pred.detach().numpy(), I_pred.detach().numpy()
            
#Generate synthetic SIR data

def generate_synthetic_data(beta_true=0.3, gamma_true=0.1, days=100, noise_level=0.05):
    N = 1000
    S0, I0 = 999, 1
    
    t = np.linspace(0, days, days + 1)
    S, I = np.zeros_like(t), np.zeros_like(t)
    incidence = np.zeros_like(t)  # Create FULL array for incidence
    
    S[0], I[0] = S0, I0
    incidence[0] = 0  # No infections at t=0
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        new_infections = beta_true * S[i-1] * I[i-1] / N * dt
        
        incidence[i] = new_infections  # Store in array
        recoveries = gamma_true * I[i-1] * dt
        
        S[i] = S[i-1] - new_infections
        I[i] = I[i-1] + new_infections - recoveries
    
    # Add noise to infected
    I_noisy = I + noise_level * np.std(I) * np.random.randn(len(I))
    
    return t, S, I, I_noisy, incidence  

data= generate_synthetic_data(beta_true=0.3, gamma_true=0.1, days=100, noise_level=0.1)
t, S, I, I_noisy, incidence = data[0], data[1], data[2], data[3], data[4]

# Create PINN network
pinn = create_pinn()

trained_pinn = train_pinn(pinn, t_obs = torch.tensor(t, dtype= torch.float32).unsqueeze(-1),
                          I_obs = torch.tensor(I_noisy, dtype= torch.float32).unsqueeze(-1),
                          t_colloc = torch.tensor(t, dtype = torch.float32).unsqueeze(-1),
                          num_epochs = 5000, learning_rate = 0.001)

#Make predictions
t_test = torch.tensor(t, dtype=torch.float32).unsqueeze(-1)
S_pred, I_pred = predict_pinn(trained_pinn, t_test)

#plot results
plt.figure(figsize=(10,6))
plt.plot(t, I, 'b-', label= 'True Infected', linewidth =2)
plt.scatter(t, I_noisy, c ='r', label ='Noisy Observations', s=10)
plt.plot(t, I_pred, 'g--', label= 'PINN predicted Infected', linewidth =2)
plt.xlabel('Time (days)')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.title('PINN SIR MModel Prediction')
plt.show()

#Print learned parameters
print(f"learned beta: {trained_pinn['beta'].item():.4f}, learned gamma: {trained_pinn['gamma'].item():.4f}")

#Plot incidence curve
incidence_pred =np.zeros_like(t)
for i in range(1, len(t)):
    incidence_pred[i] = (
    trained_pinn['beta'].item()
    * S_pred[i-1, 0]
    * I_pred[i-1, 0]
    / trained_pinn['N'])
plt.figure(figsize=(10,6))
plt.plot(t, incidence, 'b-', label='True Incidence', linewidth=2)
plt.plot(t, incidence_pred, 'r--', label='PINN Predicted Incidence', linewidth=2)
plt.xlabel('Time (days)')
plt.ylabel('Number of New Infections per day')
plt.legend()
plt.title('PINN SIR Model Incidence Prediction')
plt.show()







          


                        














                




    

    