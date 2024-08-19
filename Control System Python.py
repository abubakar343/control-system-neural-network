#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from functools import partial


# In[ ]:


dv = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(dv)


# In[ ]:


def u0(x):
    """initial condition"""
    u = torch.sin(x)*3-1
    return u

class fcn(nn.Module):
    """basic FF network for approximating functions"""
    def __init__(self, nn_width=30, num_hidden=2):
        super().__init__()
        
        self.layer_first = nn.Linear(2, nn_width)
        
        layers = []
        for _ in range(num_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.layer_hidden = nn.ModuleList(layers)
        
        self.layer_last = nn.Linear(nn_width, 1)
        
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        activation = nn.Tanh()
        u = activation(self.layer_first(xt))
        for hidden in self.layer_hidden:
            u = activation(hidden(u))
        u = self.layer_last(u)
        return u


# In[ ]:


# 1. FIXME -- training data
####
nx_train = 125
nt_train = 100
x1d = torch.linspace(-1,1, nx_train, requires_grad=True)
t1d = torch.linspace(-1, 1, nt_train, requires_grad=True)

xm, tm = torch.meshgrid(x1d, t1d)
x1d = x1d.reshape(-1, 1)
t1d = t1d.reshape(-1, 1)
x = xm.reshape(-1, 1)
t = tm.reshape(-1, 1)
##### First of all, draw a background with a range(-1,1) using the code



# 2. set the model
torch.manual_seed(5678)
model = fcn(30, 2)

# 3. set the optimizer
lr = 0.001
opt = torch.optim.Adam(model.parameters(), lr)
n_epochs = 1500
loss_history = []

# 4. FIXME -- set the loss for BC, IC, and the residual
def loss_function(model,x, t, x1d, t1d):    
    ####

    #IC/BC/LOSS
    u_ic = u0(x1d)
    t1d0 = torch.zeros_like(x1d)
    loss_ic = model(x1d, t1d0) - u_ic

    #initial condition

    loss_bc_left = model(torch.zeros_like(t1d0), t1d0)
    loss_bc_right = model(torch.zeros_like(t1d0), t1d0)

    #Boundary condition

    #Residuals
    u = model(x,t)   
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.zeros_like(t), create_graph=True, retain_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.zeros_like(x), create_graph=True, retain_graph=True)[0]
    loss_res = du_dt + du_dx
    #differentiation the value 

    # total loss
    loss_value = loss_ic.pow(2).mean() +                  loss_bc_left.pow(2).mean() +                  loss_bc_right.pow(2).mean() +                  loss_res.pow(2).mean()

    #loss = lossu + lossr

    return loss_value


# In[ ]:


for i in range(n_epochs):

    loss = loss_function(model, x, t, x1d, t1d)
    opt.zero_grad()
    loss_history.append(loss.item())
    
    loss.backward()
    opt.step()

    if i % 10 == 0:
        print(f'epoch {i}, loss = {loss}')


# In[ ]:


plt.plot(loss_history)
plt.xlabel('number of ep')
plt.ylabel('loss value')


# In[ ]:


plt.plot(x1d.detach().numpy(),u0(x1d).detach().numpy(),'gray',label="Reference")
plt.plot(x1d.detach().numpy(),model(x1d,torch.zeros_like(x1d)).detach().numpy(),label='Result')
plt.xlabel("x")
plt.ylabel("u")
plt.grid()
plt.legend()


# In[1]:


from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')


# In[ ]:


def f(x, y):
    return np.sin(3*y**2)+2*np.cos(2*y)+1-np.sin(x)*3-1
    #ux-u0

x = np.linspace(0, 1, 30)
y = np.linspace(0, 20, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)


# In[ ]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');


# In[ ]:


ax.view_init(60, 50)
fig


# In[ ]:


def u_true(x, y):
    return np.sin(3*y**2)+2*np.cos(2*y)+1-np.sin(x)*3-1
def f_true(x,y):
    return torch.sin(3*y1**2)+2*torch.cos(2*y1)+1-torch.sin(x)*3-1

x_train = np.random.rand(1000)
y_train = np.random.rand(1000)
u_train = u_true(x_train, y_train)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_train, y_train, u_train)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,20,500)
y = np.sin(3*x**2)+np.cos(2*x)*2+1
plt.plot(x,y, 'r')
plt.show()

#Tracking Y-Graph


# In[ ]:




