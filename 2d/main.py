import torch
from torch import nn

import numpy as np

import matplotlib.pyplot as plt

from net import Net

def generate_function_data(count, func):
    x = torch.rand((count,2), dtype=torch.float32)*4.0-2.0
    x[:,1] = func(x[:,0])
    y = torch.ones((x.shape[0]), dtype=torch.long)
    y[count//2:] = 0
    x[y==0,1] += (0.5+torch.rand(y.shape[0]-count//2)) - (np.random.randn(y.shape[0]-count//2)<0)*2.0
    return x, y

def train(model, loss_function, optimizer, epochs):
    for epoch in range(epochs):
        if epoch%1000 == 0:
            print(f"Epoch {epoch+1}:")

        model.train()
        z_train = model(x_train)
        loss = loss_function(z_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%1000 == 0:
            print(f"\tTraining loss: {loss.item()}")

        model.eval()
        z_val = model(x_val)
        loss = loss_function(z_val, y_val)
        if epoch%1000 == 0:
            print(f"\tvalidation loss: {loss.item()}\n")

def compare(model, x, y, resolution=1000):
    model.eval()
    with torch.no_grad():
        grid_x, grid_y = torch.meshgrid(torch.linspace(-2.0, 2.0, resolution), torch.linspace(-2.5, 2.5, resolution))
        grid = torch.cat((torch.reshape(grid_x, (-1,1)), torch.reshape(grid_y, (-1,1))), 1)

        probs = nn.functional.softmax(model(grid), dim=1)
        output = torch.reshape(probs[:,0], (resolution,resolution))

        plt.contour(grid_x.detach().numpy(), grid_y.detach().numpy(), output.detach().numpy(), 1, cmap=plt.cm.bone)
        plt.contourf(grid_x.detach().numpy(), grid_y.detach().numpy(), output.detach().numpy(), 1, cmap=plt.cm.bone)
        plt.plot(x[y==0,0], x[y==0,1], 'x', label='Class 0')
        plt.plot(x[y==1,0], x[y==1,1], 'o', label='Class 1')
        plt.legend()
        plt.grid()
        plt.show()

model = Net()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

x_train, y_train = generate_function_data(800, lambda x: np.sin(5.0*x))
x_val,   y_val   = generate_function_data(200, lambda x: np.sin(5.0*x))

train(model, loss_function, optimizer, 1001)
compare(model, x_train, y_train)