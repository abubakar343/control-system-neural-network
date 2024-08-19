#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from functools import partial

# Setting the device for computation (GPU if available, otherwise CPU)
dv = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(dv)

def u0(x):
    """Initial condition for the problem."""
    u = torch.sin(x) * 3 - 1
    return u

class fcn(nn.Module):
    """Basic Feedforward Neural Network (FFNN) for approximating functions."""
    def __init__(self, nn_width=30, num_hidden=2):
        super().__init__()
        
        # Initial layer
        self.layer_first = nn.Linear(2, nn_width)
        
        # Hidden layers
        layers = []
        for _ in range(num_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.layer_hidden = nn.ModuleList(layers)
        
        # Output layer
        self.layer_last = nn.Linear(nn_width, 1)
        
    def forward(self, x, t):
        """Forward pass of the network."""
        xt = torch.cat([x, t], dim=1)  # Concatenate x and t
        activation = nn.Tanh()  # Activation function
        u = activation(self.layer_first(xt))
        for hidden in self.layer_hidden:
            u = activation(hidden(u))
        u = self.layer_last(u)
        return u

# Define training data
nx_train = 125
nt_train = 100
x1d = torch.linspace(-1, 1, nx_train, requires_grad=True)
t1d = torch.linspace(-1, 1, nt_train, requires_grad=True)

xm, tm = torch.meshgrid(x1d, t1d)
x1d = x1d.reshape(-1, 1)
t1d = t1d.reshape(-1, 1)
x = xm.reshape(-1, 1)
t = tm.reshape(-1, 1)

# Initialize the model
torch.manual_seed(5678)
model = fcn(30, 2)

# Set the optimizer
lr = 0.001
opt = torch.optim.Adam(model.parameters(), lr)
n_epochs = 1500
loss_history = []

# Define the loss function
def loss_function(model, x, t, x1d, t1d):    
    """Compute the loss for boundary conditions (BC), initial conditions (IC), and residuals."""
    
    # Initial condition loss
    u_ic = u0(x1d)
    t1d0 = torch.zeros_like(x1d)
    loss_ic = model(x1d, t1d0) - u_ic

    # Boundary condition losses
    loss_bc_left = model(torch.zeros_like(t1d0), t1d0)
    loss_bc_right = model(torch.ones_like(t1d0), t1d0)  # Correct boundary condition here

    # Residuals
    u = model(x, t)   
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.zeros_like(t), create_graph=True, retain_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.zeros_like(x), create_graph=True, retain_graph=True)[0]
    loss_res = du_dt + du_dx

    # Total loss
    loss_value = loss_ic.pow(2).mean() + \
                 loss_bc_left.pow(2).mean() + \
                 loss_bc_right.pow(2).mean() + \
                 loss_res.pow(2).mean()

    return loss_value

# Training loop
for i in range(n_epochs):
    loss = loss_function(model, x, t, x1d, t1d)
    opt.zero_grad()
    loss_history.append(loss.item())
    
    loss.backward()
    opt.step()

    if i % 10 == 0:
        print(f'epoch {i}, loss = {loss}')

# Plot the loss history
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot the results against the reference
plt.plot(x1d.detach().numpy(), u0(x1d).detach().numpy(), 'gray', label="Reference")
plt.plot(x1d.detach().numpy(), model(x1d, torch.zeros_like(x1d)).detach().numpy(), label='Result')
plt.xlabel("x")
plt.ylabel("u")
plt.grid()
plt.legend()
plt.show()

# 3D Plot of the function
from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

def f(x, y):
    """Compute the function value."""
    return np.sin(3*y**2) + 2*np.cos(2*y) + 1 - np.sin(x)*3 - 1

x = np.linspace(0, 1, 30)
y = np.linspace(0, 20, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 3D contour plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(60, 50)
plt.show()

# True function plot
def u_true(x, y):
    """True function values."""
    return np.sin(3*y**2) + 2*np.cos(2*y) + 1 - np.sin(x)*3 - 1

x_train = np.random.rand(1000)
y_train = np.random.rand(1000)
u_train = u_true(x_train, y_train)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_train, y_train, u_train)
plt.show()

# Plot the reference function
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 500)
y = np.sin(3*x**2) + np.cos(2*x)*2 + 1
plt.plot(x, y, 'r')
plt.show()
