#!/usr/bin/env python3
"""01_heat_equation_pinn.py — Vanilla PINN for the 1D Heat Equation"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from visualize_heat import visualize_heat_solution
# ── Hyperparameters ──
NU = 0.01              # thermal diffusivity
N_COLLOCATION = 10000  # interior collocation points
N_BC = 200             # boundary points per edge
N_IC = 200             # initial condition points
EPOCHS = 5000
LR = 1e-3

# ── Network Definition ──
class PINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 1]):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# ── Sampling Functions ──
def sample_collocation(n):
    x = torch.rand(n, 1, requires_grad=True)
    t = torch.rand(n, 1, requires_grad=True)
    return x, t

def sample_ic(n):
    x = torch.rand(n, 1, requires_grad=True)
    t = torch.zeros(n, 1)
    u = torch.sin(np.pi * x)
    return x, t, u

def sample_bc(n):
    t = torch.rand(n, 1)
    # x = 0 and x = 1 boundaries
    x_left = torch.zeros(n, 1)
    x_right = torch.ones(n, 1)
    x = torch.cat([x_left, x_right])
    t = torch.cat([t, t])
    u = torch.zeros(2 * n, 1)
    return x, t, u

# ── Physics Residual ──
def physics_residual(model, x, t):
    u = model(x, t)
    # Compute gradients via automatic differentiation
    u_t = torch.autograd.grad(u, t, torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                create_graph=True)[0]
    # Heat equation: u_t - ν * u_xx = 0
    residual = u_t - NU * u_xx
    return residual

# ── Training Loop ──
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # Physics loss
    x_c, t_c = sample_collocation(N_COLLOCATION)
    res = physics_residual(model, x_c, t_c)
    loss_phys = (res ** 2).mean()

    # IC loss
    x_i, t_i, u_i = sample_ic(N_IC)
    loss_ic = ((model(x_i, t_i) - u_i) ** 2).mean()

    # BC loss
    x_b, t_b, u_b = sample_bc(N_BC)
    loss_bc = ((model(x_b, t_b) - u_b) ** 2).mean()

    loss = loss_phys + 10 * loss_ic + 10 * loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.6f} "
              f"(phys={loss_phys.item():.6f}, "
              f"ic={loss_ic.item():.6f}, bc={loss_bc.item():.6f})")

print("✓ Training complete")
visualize_heat_solution(model, nu=NU, nx=200, nt=200)
print("✓ Visualization complete")