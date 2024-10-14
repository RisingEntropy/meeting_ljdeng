import einops
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as sio

# $\frac{\partial\Phi}{\partial t}=\frac{\partial ^2 \Phi}{\partial x}+1-|x|;0\le t\le 1; -1\le x\le 1; t=0,\Phi=0;x=\pm1,\Phi=0$

class SineLayer(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True, is_first=False, omega_0=4):
        super(SineLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_feature = in_feature
        self.linear = nn.Linear(in_feature, out_feature, bias=bias)
        self.init_weights()

    def init_weights(self):

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_feature, 1 / self.in_feature)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_feature) / self.omega_0,
                                             np.sqrt(6 / self.in_feature) / self.omega_0)

    def forward(self, x):
        tmp = self.linear(x)*self.omega_0
        return torch.sin(tmp)


class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)*x


class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return torch.exp(x)


def f(x):
    return 1 - torch.abs(x[:, 1])


# def first_layer_sine_init(m):
#     with torch.no_grad():
#         if hasattr(m, 'weight'):
#             num_input = m.weight.size(-1)
#             m.weight.uniform_(-1 / num_input, 1 / num_input)
#             m.weight *= 4


network = nn.Sequential(
    SineLayer(2, 128, is_first=True),
    SineLayer(128, 128),
    nn.Linear(128, 1),
).cuda()
optim = optim.Adam(network.parameters(), lr=1e-4, weight_decay=1e-5)

# first_layer_sine_init(network[0])
for epoch in range(3000):
    optim.zero_grad()
    # t                                      x
    grid = torch.meshgrid(torch.linspace(0, 1, 200), torch.linspace(-1, 1, 200), indexing='ij')
    grid = torch.stack(grid, dim=-1).cuda()
    grid = grid.reshape(-1, 2)
    grid.requires_grad = True
    y = f(grid).cuda()

    pred = network(grid)
    grad = torch.autograd.grad(pred, grid, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    par_t = grad[:, 0]
    par_x = grad[:, 1]
    par_par_x = torch.autograd.grad(par_x, grid, grad_outputs=torch.ones_like(par_x), create_graph=True)[0][:, 1]



    t_zero_grid = torch.stack(torch.meshgrid(torch.zeros(200), torch.linspace(-1, 1, 200), indexing='ij'),
                              dim=-1).cuda()
    x_1_grid = torch.stack(torch.meshgrid(torch.linspace(0, 1, 200), torch.ones(200), indexing='ij'), dim=-1).cuda()
    x_m1_grid = torch.stack(torch.meshgrid(torch.linspace(0, 1, 200), -torch.ones(200), indexing='ij'), dim=-1).cuda()

    loss_eq = F.mse_loss(par_t-par_par_x, y)
    loss_edge = F.mse_loss(network(t_zero_grid.reshape(-1, 2)), torch.zeros_like(pred).cuda()) + \
                F.mse_loss(network(x_1_grid.reshape(-1, 2)), torch.zeros_like(pred).cuda()) + \
                F.mse_loss(network(x_m1_grid.reshape(-1, 2)), torch.zeros_like(pred).cuda())
    loss = loss_eq + loss_edge
    loss.backward()

    optim.step()
    print(f"epoch:{epoch}:{loss.item()}")
    if (epoch + 1) % 500 == 0:
        torch.save(network.state_dict(), "./results/Schrodinger.pth")

network.load_state_dict(torch.load('results/Schrodinger.pth'))
network.eval()
grid = torch.meshgrid(torch.linspace(0, 1, 200), torch.linspace(-1, 1,
                                                                 200), indexing='ij')
grid = torch.stack(grid, dim=-1).cuda()
pred = network(grid.reshape(-1, 2)).reshape(200, 200)
print(pred)
plt.imshow(pred.cpu().detach().numpy())
plt.show()
sio.savemat('results/Schrodinger.mat', {"grid": grid.cpu().detach().numpy(), "pred": pred.cpu().detach().numpy()})