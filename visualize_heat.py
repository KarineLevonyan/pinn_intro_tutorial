#!/usr/bin/env python3
"""visualize_heat.py — Plot PINN vs analytical solution for the heat equation"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ── Assume `model` is your trained PINN from heat_pinn.py ──

def visualize_heat_solution(model, nu=0.01, nx=200, nt=200):
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)

    # ── PINN prediction ──
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).numpy().reshape(nt, nx)

    # ── Analytical solution: u(x,t) = sin(πx) * exp(-ν*π²*t) ──
    u_exact = np.sin(np.pi * X) * np.exp(-nu * np.pi**2 * T)

    # ── Absolute error ──
    error = np.abs(u_pred - u_exact)

    # ── Create figure with 3 panels ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('1D Heat Equation: PINN vs Analytical',
                 fontsize=16, fontweight='bold')

    # Panel 1: PINN prediction
    im0 = axes[0].pcolormesh(X, T, u_pred, cmap='inferno',
                              shading='auto')
    axes[0].set_title('PINN Prediction')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
    fig.colorbar(im0, ax=axes[0], label='u(x,t)')

    # Panel 2: Exact solution
    im1 = axes[1].pcolormesh(X, T, u_exact, cmap='inferno',
                              shading='auto')
    axes[1].set_title('Analytical Solution')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('t')
    fig.colorbar(im1, ax=axes[1], label='u(x,t)')

    # Panel 3: Absolute error
    im2 = axes[2].pcolormesh(X, T, error, cmap='hot',
                              shading='auto')
    axes[2].set_title(f'|Error|  (max={error.max():.2e})')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('t')
    fig.colorbar(im2, ax=axes[2], label='|u_pred - u_exact|')

    plt.tight_layout()
    plt.savefig('heat_pinn_result.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Time-slice comparison ──
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for t_val in [0.0, 0.2, 0.5, 0.8]:
        idx = int(t_val * (nt - 1))
        ax2.plot(x, u_exact[idx], '--', lw=2,
                 label=f'Exact t={t_val}')
        ax2.plot(x, u_pred[idx], 'o', markersize=3,
                 label=f'PINN  t={t_val}')
    ax2.set_xlabel('x'); ax2.set_ylabel('u(x,t)')
    ax2.set_title('Time Slices: PINN (dots) vs Exact (dashed)')
    ax2.legend(ncol=2); ax2.grid(True, alpha=0.3)
    plt.savefig('heat_time_slices.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Max error: {error.max():.2e}")
    print(f"Mean error: {error.mean():.2e}")
    print(f"L2 relative error: {np.linalg.norm(error)/np.linalg.norm(u_exact):.2e}")

#