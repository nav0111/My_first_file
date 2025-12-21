import torch
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## Synthetic data generation
#The SEIR model with noise
def generate_seir_data(beta_true=0.35, sigma_true=0.2, gamma_u_true=0.1, gamma_r_true=0.1, 
                       p_true=0.6, t_span=(0, 100), t_eval=None, N=1000, noise_level=0.05):
    S0, E0, I_u0, I_r0 = 990, 5, 3, 2
    
    def seir_model(t, y):
        S, E, I_u, I_r = y
        I_total = I_u + I_r
        dSdt = -beta_true * S * I_total / N
        dEdt = beta_true * S * I_total / N - sigma_true * E
        dI_udt = (1-p_true)*sigma_true * E - gamma_u_true * I_u
        dI_rdt = p_true * sigma_true * E - gamma_r_true * I_r
        return [dSdt, dEdt, dI_udt, dI_rdt] 
    
    # Solve the ODE
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    
    sol = solve_ivp(seir_model, t_span, [S0, E0, I_u0, I_r0], 
                    t_eval=t_eval, method='RK45')
    
    S, E, I_u, I_r = sol.y
    
    # Add noise to the data
    I_r_noisy = I_r + noise_level * np.std(I_r) * np.random.randn(len(I_r))
    
    return t_eval, S, E, I_u, I_r, I_r_noisy

# Create PINN model
def create_pinn(input_dim=1, output_dim=4, hidden_layers=3, hidden_dim=20):
    pinn_dict = {}
    
    # Known initial values
    pinn_dict['N'] = 1000.0
    pinn_dict['S0'] = 990.0
    pinn_dict['E0'] = 5.0
    pinn_dict['I_u0'] = 3.0
    pinn_dict['I_r0'] = 2.0

    #EPIDEMIC PARAMETERS (with positivity constraints) 
    # Initial guesses for parameters
    beta_init = 0.3
    sigma_init = 0.2
    gamma_u_init = 0.1
    gamma_r_init = 0.1
    p_init = 0.6
    
    # Use log-transformed parameters to enforce positivity
    pinn_dict['log_beta'] = torch.nn.Parameter(torch.tensor(np.log(beta_init), dtype=torch.float32))
    pinn_dict['log_sigma'] = torch.nn.Parameter(torch.tensor(np.log(sigma_init), dtype=torch.float32))
    pinn_dict['log_gamma_u'] = torch.nn.Parameter(torch.tensor(np.log(gamma_u_init), dtype=torch.float32))
    pinn_dict['log_gamma_r'] = torch.nn.Parameter(torch.tensor(np.log(gamma_r_init), dtype=torch.float32))
    # For p, use sigmoid to keep in (0,1)
    pinn_dict['logit_p'] = torch.nn.Parameter(torch.tensor(np.log(p_init/(1-p_init)), dtype=torch.float32))
    
    # Get actual parameters
    def get_params():
        beta = torch.exp(pinn_dict['log_beta'])
        sigma = torch.exp(pinn_dict['log_sigma'])
        gamma_u = torch.exp(pinn_dict['log_gamma_u'])
        gamma_r = torch.exp(pinn_dict['log_gamma_r'])
        p = torch.sigmoid(pinn_dict['logit_p'])
        return beta, sigma, gamma_u, gamma_r, p
    
    pinn_dict['get_params'] = get_params
    
    #NEURAL NETWORK ARCHITECTURE
    # Input layer
    w1 = torch.nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
    b1 = torch.nn.Parameter(torch.zeros(hidden_dim))
    pinn_dict['w1'] = w1
    pinn_dict['b1'] = b1
    
    # Hidden layers
    pinn_dict['hidden_weights'] = []
    pinn_dict['hidden_biases'] = []
    for i in range(hidden_layers):
        w = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        b = torch.nn.Parameter(torch.zeros(hidden_dim))
        pinn_dict['hidden_weights'].append(w)
        pinn_dict['hidden_biases'].append(b)

    # Output layer
    w_out = torch.nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
    # Initialize biases for sigmoid output: sigmoid^-1(value/N)
    b_out = torch.nn.Parameter(torch.tensor([
        np.log(990/10),    # S: 990/1000 = 0.99, sigmoid^-1(0.99) ≈ 5.3
        np.log(5/995),     # E: 5/1000 = 0.005, sigmoid^-1(0.005) ≈ -5.3
        np.log(3/997),     # I_u: 3/1000 = 0.003
        np.log(2/998)      # I_r: 2/1000 = 0.002
    ], dtype=torch.float32))
    
    pinn_dict['w_out'] = w_out
    pinn_dict['b_out'] = b_out
    
    # Collect all trainable parameters
    params = [
        pinn_dict['log_beta'], pinn_dict['log_sigma'], 
        pinn_dict['log_gamma_u'], pinn_dict['log_gamma_r'], pinn_dict['logit_p'],
        w1, b1, w_out, b_out
    ]
    for w, b in zip(pinn_dict['hidden_weights'], pinn_dict['hidden_biases']):
        params.extend([w, b])
    
    pinn_dict['params'] = params
    
    return pinn_dict

# Forward pass
def pinn_forward(pinn_dict, t):
    if t.dim() == 1:
        t = t.unsqueeze(1)
    
    # Input layer
    H = torch.tanh(torch.matmul(t, pinn_dict['w1']) + pinn_dict['b1'])
    
    # Hidden layers
    #zip is a function that applies loop over the list of weights and biases, paring each
    #weight with its corresponding bias and applies the tanh function to the result
    for w, b in zip(pinn_dict['hidden_weights'], pinn_dict['hidden_biases']):
        H = torch.tanh(torch.matmul(H, w) + b)

    
    
    # Output layer
    output = torch.matmul(H, pinn_dict['w_out']) + pinn_dict['b_out']
    
    # Apply sigmoid and scale by total population
    output = torch.sigmoid(output) * pinn_dict['N']
    
    # Split into compartments
    S = output[:, 0:1]
    E = output[:, 1:2]
    I_u = output[:, 2:3]
    I_r = output[:, 3:4]
    
    return S, E, I_u, I_r

# Compute derivatives
def compute_derivatives(pinn_dict, t):
    if t.dim() == 1:
        t = t.unsqueeze(1)
    
    # Enable gradient computation
    if not t.requires_grad:
        t = t.clone().detach().requires_grad_(True)
    
    # Get predictions
    S, E, I_u, I_r = pinn_forward(pinn_dict, t)
    
    # Compute gradients
    dS_dt = grad(S, t, grad_outputs=torch.ones_like(S), create_graph=True, retain_graph=True)[0]
    dE_dt = grad(E, t, grad_outputs=torch.ones_like(E), create_graph=True, retain_graph=True)[0]
    dI_u_dt = grad(I_u, t, grad_outputs=torch.ones_like(I_u), create_graph=True, retain_graph=True)[0]
    dI_r_dt = grad(I_r, t, grad_outputs=torch.ones_like(I_r), create_graph=True)[0]
    
    return S, E, I_u, I_r, dS_dt, dE_dt, dI_u_dt, dI_r_dt

# Loss functions
def compute_data_loss(pinn_dict, t_obs, I_r_obs):
    S_pred, E_pred, I_u_pred, I_r_pred = pinn_forward(pinn_dict, t_obs)
    # Weight later points more (epidemic dynamics more important later)
    weights = torch.linspace(0.5, 1.5, len(I_r_obs)).unsqueeze(1).to(I_r_obs.device)
    return torch.mean(weights * (I_r_pred - I_r_obs) ** 2)

def compute_ode_loss(pinn_dict, t_colloc):
    S, E, I_u, I_r, dS_dt, dE_dt, dI_u_dt, dI_r_dt = compute_derivatives(pinn_dict, t_colloc)
    
    # Get parameters
    beta, sigma, gamma_u, gamma_r, p = pinn_dict['get_params']()
    
    # SEIR ODE residuals
    I_total = I_u + I_r
    force = beta * S * I_total / pinn_dict['N']
    
    res_S = dS_dt + force
    res_E = dE_dt - force + sigma * E
    res_I_u = dI_u_dt - (1 - p) * sigma * E + gamma_u * I_u  
    res_I_r = dI_r_dt - p * sigma * E + gamma_r * I_r 

    
    # Weight ODE loss more at t=0 where dynamics are crucial
    #giving more importance to the ODE constraints at early times (t ≈ 0) 
    #where epidemic dynamics are most sensitive and errors propagate.
    t_weight = 1.0 + 9.0 * torch.exp(-0.1 * t_colloc)
    ode_loss = torch.mean(t_weight * (res_S**2 + res_E**2 + res_I_u**2 + res_I_r**2))
    
    # Add positivity constraints, enforce compartment values (S, E, I_u, I_r) remain non-negative
    pos_loss = torch.mean(torch.relu(-S) + torch.relu(-E) + torch.relu(-I_u) + torch.relu(-I_r))
    
    return ode_loss + 0.1 * pos_loss

def compute_ic_loss(pinn_dict):
    t0 = torch.tensor([[0.0]], dtype=torch.float32)
    S0_pred, E0_pred, I_u0_pred, I_r0_pred = pinn_forward(pinn_dict, t0)
    
    ic_loss = (
        (S0_pred - pinn_dict['S0'])**2 +
        (E0_pred - pinn_dict['E0'])**2 +
        (I_u0_pred - pinn_dict['I_u0'])**2 +
        (I_r0_pred - pinn_dict['I_r0'])**2
    )
    
    return ic_loss.mean()

# Parameter regularization (keep parameters in reasonable ranges)
def compute_param_regularization(pinn_dict):
    beta, sigma, gamma_u, gamma_r, p = pinn_dict['get_params']()
    
    # Penalize extreme values
    reg_loss = (
        0.01 * (torch.log(beta) - torch.log(torch.tensor(0.3)))**2 +
        0.01 * (torch.log(sigma) - torch.log(torch.tensor(0.2)))**2 +
        0.01 * (torch.log(gamma_u) - torch.log(torch.tensor(0.1)))**2 +
        0.01 * (torch.log(gamma_r) - torch.log(torch.tensor(0.1)))**2 +
        0.01 * (torch.log(p/(1-p)) - torch.log(torch.tensor(0.6/0.4)))**2
    )
    
    return reg_loss

def compute_total_loss(pinn_dict, t_obs, I_r_obs, t_colloc):
    data_loss = compute_data_loss(pinn_dict, t_obs, I_r_obs)
    ode_loss = compute_ode_loss(pinn_dict, t_colloc)
    ic_loss = compute_ic_loss(pinn_dict)
    param_reg = compute_param_regularization(pinn_dict)
    
    # Adaptive weighting (emphasize physics early, data later)
    total_loss = (data_loss + 10*ode_loss + ic_loss + param_reg)
    
    return total_loss, data_loss, ode_loss, ic_loss, param_reg

#Training loop
def train_pinn(pinn_dict, t_obs, I_r_obs, t_colloc, epochs=10000, learning_rate= 0.001):

    optimizer = optim.Adam(pinn_dict['params'], lr = learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, data_loss, ode_loss, ic_loss, param_reg = compute_total_loss(pinn_dict, t_obs, I_r_obs,
                                                                                 t_colloc)
        total_loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
           print(f"Epoch {epoch}, Total Loss: {total_loss.item()}, Data Loss: {data_loss.item()}, ODE Loss: {ode_loss.item()}, IC Loss: {ic_loss.item()}, Param Reg: {param_reg.item()}")

    return pinn_dict

#Make predictions
def predict_pinn(pinn_dict, t_pred):
    S_pred, E_pred, I_u_pred, I_r_pred = pinn_forward(pinn_dict, t_pred)
    return S_pred.detach().numpy(), E_pred.detach().numpy(), I_u_pred.detach().numpy(), I_r_pred.detach().numpy()


#The syntheic data generation
data = generate_seir_data(beta_true=0.35, sigma_true=0.2, gamma_u_true=0.1, gamma_r_true=0.1,
                          p_true=0.6, t_span=(0, 100), t_eval=None, N=1000, noise_level=0.05)

t, S, E, I_u, I_r, I_r_noisy = data[0], data[1], data[2], data[3], data[4], data[5]

t_obs = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
I_r_obs = torch.tensor(I_r_noisy, dtype=torch.float32).unsqueeze(1)
t_colloc = torch.tensor(np.linspace(t[0], t[-1], 1000), dtype=torch.float32).unsqueeze(1)
t_pred = torch.tensor(np.linspace(t[0], t[-1], 1000), dtype=torch.float32).unsqueeze(1)

#Create and train the PINN
pinn_dict = create_pinn()
pinn_dict = train_pinn(pinn_dict, t_obs, I_r_obs, t_colloc, epochs = 10000, learning_rate = 0.001)

#Make predictions
S_pred, E_pred, I_u_pred, I_r_pred = predict_pinn(pinn_dict, t_pred)

#Plot the results
plt.figure(figsize= (10,6))
plt.subplot(2,2,1)
plt.plot(t, I_r, label='True I_r', linewidth=2)
plt.plot(t_pred, I_r_pred, label='Predicted I_r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('I_r')
plt.legend()
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(t, S, label='True S', linewidth=2)
plt.plot(t_pred, S_pred, label='Predicted S', linewidth=2)
plt.xlabel('Time')
plt.ylabel('S')
plt.legend()
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(t, E, label='True E', linewidth=2)
plt.plot(t_pred, E_pred, label='Predicted E', linewidth=2)
plt.xlabel('Time')
plt.ylabel('E')
plt.legend()
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(t, I_u, label='True I_u', linewidth=2)
plt.plot(t_pred, I_u_pred, label='Predicted I_u', linewidth=2)
plt.xlabel('Time')
plt.ylabel('I_u')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#Print the learned parameters
beta, sigma, gamma_u, gamma_r, p = pinn_dict['get_params']()
print(f"True Beta: {0.35}, Learned beta: {beta}")
print(f"True Sigma: {0.2}, Learned sigma: {sigma}")
print(f"True Gamma_u: {0.1}, Learned gamma_u: {gamma_u}")
print(f"True Gamma_r: {0.1}, Learned gamma_r: {gamma_r}")
print(f"True p: {0.6}, Learned p: {p}")

# Calculate and plot incidence
# Incidence = p * sigma * E (new reported cases per day)

# Get parameters from the trained PINN
beta, sigma, gamma_u, gamma_r, p = pinn_dict['get_params']()

# Convert PyTorch tensors to numpy for calculation
beta_np = beta.detach().numpy()
sigma_np = sigma.detach().numpy()
gamma_u_np = gamma_u.detach().numpy()
gamma_r_np = gamma_r.detach().numpy()
p_np = p.detach().numpy()

# Calculate TRUE incidence from the synthetic data
true_incidence = 0.6 * 0.2 * E  # p_true * sigma_true * E

# Calculate PREDICTED incidence from PINN
# Need to get E predictions at all time points
S_pred, E_pred_full, I_u_pred, I_r_pred = pinn_forward(pinn_dict, t_pred)
predicted_incidence = p_np * sigma_np * E_pred_full.detach().numpy()

# Plot
plt.figure(figsize=(10,6))
plt.plot(t, true_incidence, 'b-', label='True Incidence', linewidth=2)
plt.plot(t_pred, predicted_incidence, 'r--', label='Predicted Incidence', linewidth=2)
plt.xlabel('Time (days)')
plt.ylabel('Daily New Reported Cases')
plt.title('Daily Incidence')
plt.legend()
plt.grid(True)
plt.show()
