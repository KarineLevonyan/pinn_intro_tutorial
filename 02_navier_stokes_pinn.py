#!/usr/bin/env python3

"""02_navier_stokes_pinn.py — Optimized PINN for 2D steady incompressible
Navier–Stokes: channel flow past a circular cylinder at low Re.

Companion to: https://karinelevonyan.github.io/blog/2026/navier-stokes-one-term-at-a-time/

Equations (steady, incompressible):
    u·∂u/∂x + v·∂u/∂y = -1/ρ ∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
    u·∂v/∂x + v·∂v/∂y = -1/ρ ∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
    ∂u/∂x + ∂v/∂y = 0

Accuracy improvements:
  - Fourier feature embedding: encodes x,y as sin/cos at multiple scales
  - Near-cylinder dense sampling: annulus R→4R gets its own collocation set
  - Residual-based adaptive sampling: 50% of points concentrate where error is high
  - N_CYL = 2000, cylinder BC weight raised to 50×
  - Inlet pressure constraint: soft pin p ≈ P_INLET_REF at left boundary
  - L-BFGS extended to 750 steps, physics loss logged throughout
  - Outlet Neumann BC: ∂u/∂x = 0, ∂v/∂x = 0 (zero normal velocity gradient)
  - Dedicated outlet collocation strip: dense physics points near x=X_MAX
  - Fourier sigma reduced 2.0 → 1.0 to suppress high-freq oscillations
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ── Device ──
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Physical parameters ──
RE = 40
U_INF = 1.0
D_CYL = 1.0
NU = U_INF * D_CYL / RE
RHO = 1.0

# Domain
X_MIN, X_MAX = -5.0, 15.0
Y_MIN, Y_MAX = -5.0, 5.0
CYL_X, CYL_Y = 0.0, 0.0
CYL_R = D_CYL / 2

# Pressure BCs:  p=0 at outlet, p≈P_INLET_REF at inlet
# Estimate: stagnation pressure ≈ 0.5·ρ·U² = 0.5, inlet is slightly below that
P_INLET_REF = 0.4

# ── Training hyperparameters ──
N_COLLOCATION  = 5000   # global uniform collocation
N_NEAR_CYL     = 2000   # dense collocation in annulus R→4R
N_OUTLET_STRIP = 1000   # dense collocation near outlet
N_BC           = 500
N_CYL          = 2000   # cylinder surface points (raised from 1000)
EPOCHS_ADAM    = 3000
EPOCHS_LBFGS   = 750
LR_ADAM        = 1e-3
RESAMPLE_EVERY = 100
CURRICULUM_EPOCHS = 500
ADAPTIVE_FRAC  = 0.5    # fraction of N_COLLOCATION from residual-weighted sampling

# Loss weights
W_CYL        = 50.0
W_INLET      = 10.0
W_WALL       = 10.0
W_OUTLET     = 1.0
W_P_IN       = 5.0    # inlet pressure soft constraint
W_OUTLET_NEU = 5.0    # outlet Neumann: ∂u/∂x = 0, ∂v/∂x = 0

# ── Input normalization ──
X_MID  = (X_MAX + X_MIN) / 2
Y_MID  = (Y_MAX + Y_MIN) / 2
X_HALF = (X_MAX - X_MIN) / 2
Y_HALF = (Y_MAX - Y_MIN) / 2


# ── Fourier feature embedding ──
class FourierEmbedding(nn.Module):
    """Random Fourier Features: maps (x,y) → [cos(Bz), sin(Bz)] ∈ R^{2·n_freq}."""
    def __init__(self, n_freq=32, sigma=1.0, seed=42):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        B = torch.randn(2, n_freq, generator=rng) * sigma
        self.register_buffer("B", B)      # fixed, not trained

    def forward(self, xn, yn):
        z = torch.cat([xn, yn], dim=1)    # [n, 2]
        proj = z @ self.B                  # [n, n_freq]
        return torch.cat([torch.cos(2 * np.pi * proj),
                          torch.sin(2 * np.pi * proj)], dim=1)  # [n, 2·n_freq]


# ── Network ──
class PINN(nn.Module):
    """Fourier-embedded MLP: (x,y) → (u, v, p)."""
    def __init__(self, n_freq=32, hidden=[128, 128, 128]):
        super().__init__()
        self.embed = FourierEmbedding(n_freq=n_freq, sigma=1.0)
        in_dim = n_freq * 2
        layers = [in_dim] + hidden + [3]
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x, y):
        xn = (x - X_MID) / X_HALF
        yn = (y - Y_MID) / Y_HALF
        features = self.embed(xn, yn)
        out = self.net(features)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


# ── Geometry ──
def is_inside_cylinder(x, y):
    return (x - CYL_X) ** 2 + (y - CYL_Y) ** 2 < CYL_R ** 2


# ── Sampling ──
def sample_uniform(n):
    """Uniform collocation over domain, excluding cylinder interior."""
    m = int(n * 1.5)
    x = X_MIN + (X_MAX - X_MIN) * torch.rand(m, 1, device=device)
    y = Y_MIN + (Y_MAX - Y_MIN) * torch.rand(m, 1, device=device)
    mask = (~is_inside_cylinder(x, y)).squeeze(1)
    pts = torch.cat([x[mask], y[mask]], dim=1)[:n]
    return pts[:, 0:1].requires_grad_(True), pts[:, 1:2].requires_grad_(True)


def sample_near_cylinder(n, r_max=4.0):
    """Dense collocation in annulus CYL_R → r_max around the cylinder."""
    collected = []
    while sum(p.shape[0] for p in collected) < n:
        m = n * 4
        # Sample in bounding box of annulus
        x = CYL_X + (torch.rand(m, 1, device=device) * 2 - 1) * r_max
        y = CYL_Y + (torch.rand(m, 1, device=device) * 2 - 1) * r_max
        dist2 = (x - CYL_X) ** 2 + (y - CYL_Y) ** 2
        mask = ((dist2 >= CYL_R ** 2) & (dist2 <= r_max ** 2)).squeeze(1)
        pts = torch.cat([x[mask], y[mask]], dim=1)
        collected.append(pts)
    pts = torch.cat(collected, dim=0)[:n]
    return pts[:, 0:1].requires_grad_(True), pts[:, 1:2].requires_grad_(True)


def _grad_no_cg(f, var):
    """Gradient without building higher-order graph (for residual evaluation)."""
    return torch.autograd.grad(f, var, torch.ones_like(f),
                               create_graph=False, retain_graph=True)[0]


def _grad_cg(f, var):
    """Gradient keeping graph (for training)."""
    return torch.autograd.grad(f, var, torch.ones_like(f),
                               create_graph=True)[0]


def residual_magnitude(model, x, y):
    """Physics residual magnitude for adaptive sampling (no training graph)."""
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    u, v, p = model(x, y)
    # First derivatives (create_graph=True needed to compute second derivatives)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=False, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=False, retain_graph=True)[0]
    # Second derivatives (no graph needed beyond this)
    u_xx = _grad_no_cg(u_x, x);  u_yy = _grad_no_cg(u_y, y)
    v_xx = _grad_no_cg(v_x, x);  v_yy = _grad_no_cg(v_y, y)
    with torch.no_grad():
        res_u   = u * u_x + v * u_y + (1/RHO)*p_x - NU*(u_xx + u_yy)
        res_v   = u * v_x + v * v_y + (1/RHO)*p_y - NU*(v_xx + v_yy)
        res_div = u_x + v_y
        return (res_u**2 + res_v**2 + res_div**2).detach()


def sample_adaptive(n, model):
    """Sample n points, 50% weighted by physics residual, 50% uniform."""
    n_adapt   = int(n * ADAPTIVE_FRAC)
    n_uniform = n - n_adapt

    # Uniform half
    x_u, y_u = sample_uniform(n_uniform)

    # Adaptive half: evaluate residual on candidates
    n_cand = n_adapt * 8
    x_c = X_MIN + (X_MAX - X_MIN) * torch.rand(n_cand, 1, device=device)
    y_c = Y_MIN + (Y_MAX - Y_MIN) * torch.rand(n_cand, 1, device=device)
    mask = (~is_inside_cylinder(x_c, y_c)).squeeze(1)
    x_c, y_c = x_c[mask][:n_cand//2], y_c[mask][:n_cand//2]

    res = residual_magnitude(model, x_c, y_c).squeeze()
    # Importance weights: residual + small uniform floor to retain coverage
    weights = res + res.mean() * 0.1
    idx = torch.multinomial(weights, n_adapt, replacement=False)

    x_a = x_c[idx].detach().requires_grad_(True)
    y_a = y_c[idx].detach().requires_grad_(True)

    # Concatenate uniform + adaptive
    pts_u = torch.cat([x_u.detach(), y_u.detach()], dim=1)
    pts_a = torch.cat([x_a.detach(), y_a.detach()], dim=1)
    pts   = torch.cat([pts_u, pts_a], dim=0)
    return pts[:, 0:1].requires_grad_(True), pts[:, 1:2].requires_grad_(True)


def sample_collocation(n, model=None, epoch=0):
    """Dispatch to adaptive or uniform sampling."""
    if model is not None and epoch >= CURRICULUM_EPOCHS:
        return sample_adaptive(n, model)
    return sample_uniform(n)


def sample_cylinder(n):
    theta = 2 * np.pi * torch.rand(n, 1, device=device)
    x = CYL_X + CYL_R * torch.cos(theta)
    y = CYL_Y + CYL_R * torch.sin(theta)
    return x, y


def sample_inlet(n):
    y = Y_MIN + (Y_MAX - Y_MIN) * torch.rand(n, 1, device=device)
    x = torch.full_like(y, X_MIN)
    return x, y


def sample_walls(n):
    x = X_MIN + (X_MAX - X_MIN) * torch.rand(n, 1, device=device)
    x = torch.cat([x, x])
    y = torch.cat([torch.full((n, 1), Y_MIN, device=device),
                   torch.full((n, 1), Y_MAX, device=device)])
    return x, y


def sample_outlet(n):
    """Outlet boundary — x has requires_grad so we can enforce ∂u/∂x = 0."""
    y = Y_MIN + (Y_MAX - Y_MIN) * torch.rand(n, 1, device=device)
    x = torch.full_like(y, X_MAX).requires_grad_(True)
    return x, y


def sample_outlet_strip(n, width=1.5):
    """Dense collocation in strip [X_MAX-width, X_MAX] for better outlet supervision."""
    x = (X_MAX - width) + width * torch.rand(n, 1, device=device)
    y = Y_MIN + (Y_MAX - Y_MIN) * torch.rand(n, 1, device=device)
    return x.requires_grad_(True), y.requires_grad_(True)


def sample_all_bc():
    return (sample_cylinder(N_CYL),
            sample_inlet(N_BC),
            sample_walls(N_BC),
            sample_outlet(N_BC))


# ── Physics residual (training) ──
def physics_residual(model, x, y):
    u, v, p = model(x, y)
    u_x = _grad_cg(u, x);  u_y = _grad_cg(u, y)
    v_x = _grad_cg(v, x);  v_y = _grad_cg(v, y)
    p_x = _grad_cg(p, x);  p_y = _grad_cg(p, y)
    u_xx = _grad_cg(u_x, x);  u_yy = _grad_cg(u_y, y)
    v_xx = _grad_cg(v_x, x);  v_yy = _grad_cg(v_y, y)
    res_u   = u*u_x + v*u_y + (1/RHO)*p_x - NU*(u_xx + u_yy)
    res_v   = u*v_x + v*v_y + (1/RHO)*p_y - NU*(v_xx + v_yy)
    res_div = u_x + v_y
    return res_u, res_v, res_div


# ── Loss ──
def compute_loss(model, x_c, y_c, x_nc, y_nc, x_os, y_os, bc_points, include_physics=True):
    (x_cyl, y_cyl), (x_in, y_in), (x_w, y_w), (x_out, y_out) = bc_points

    # Cylinder no-slip (higher weight)
    u_cyl, v_cyl, _ = model(x_cyl, y_cyl)
    loss_cyl = (u_cyl**2).mean() + (v_cyl**2).mean()

    # Inlet velocity
    u_in, v_in, p_in = model(x_in, y_in)
    loss_inlet_vel = ((u_in - U_INF)**2).mean() + (v_in**2).mean()

    # Inlet pressure soft constraint
    loss_inlet_p = ((p_in - P_INLET_REF)**2).mean()

    # Walls no-slip
    u_w, v_w, _ = model(x_w, y_w)
    loss_wall = (u_w**2).mean() + (v_w**2).mean()

    # Outlet: pressure = 0 + Neumann ∂u/∂x = 0, ∂v/∂x = 0
    u_out, v_out, p_out = model(x_out, y_out)
    loss_outlet_p = (p_out**2).mean()
    u_out_x = _grad_cg(u_out, x_out)
    v_out_x = _grad_cg(v_out, x_out)
    loss_outlet_neu = (u_out_x**2).mean() + (v_out_x**2).mean()

    loss_bc = (W_CYL        * loss_cyl
             + W_INLET      * loss_inlet_vel
             + W_P_IN       * loss_inlet_p
             + W_WALL       * loss_wall
             + W_OUTLET     * loss_outlet_p
             + W_OUTLET_NEU * loss_outlet_neu)

    if not include_physics:
        return loss_bc, torch.zeros(1, device=device), loss_cyl, loss_inlet_vel

    # Physics on global + near-cylinder + outlet-strip collocation
    ru, rv, rd = physics_residual(model, x_c, y_c)
    loss_phys = (ru**2).mean() + (rv**2).mean() + (rd**2).mean()

    ru_nc, rv_nc, rd_nc = physics_residual(model, x_nc, y_nc)
    loss_phys_nc = (ru_nc**2).mean() + (rv_nc**2).mean() + (rd_nc**2).mean()

    ru_os, rv_os, rd_os = physics_residual(model, x_os, y_os)
    loss_phys_os = (ru_os**2).mean() + (rv_os**2).mean() + (rd_os**2).mean()

    # Near-cylinder 3×, outlet strip 2×
    total = loss_bc + loss_phys + 3.0 * loss_phys_nc + 2.0 * loss_phys_os
    return total, loss_phys + loss_phys_nc + loss_phys_os, loss_cyl, loss_inlet_vel


# ── Phase 1: Adam ──
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS_ADAM, eta_min=1e-5)

x_c,  y_c  = sample_collocation(N_COLLOCATION)
x_nc, y_nc = sample_near_cylinder(N_NEAR_CYL)
x_os, y_os = sample_outlet_strip(N_OUTLET_STRIP)
bc_points  = sample_all_bc()

print(f"\nPhase 1: Adam  ({EPOCHS_ADAM} epochs, BC-only warmup for first {CURRICULUM_EPOCHS})")
for epoch in range(EPOCHS_ADAM):
    if epoch % RESAMPLE_EVERY == 0 and epoch > 0:
        x_c,  y_c  = sample_collocation(N_COLLOCATION, model, epoch)
        x_nc, y_nc = sample_near_cylinder(N_NEAR_CYL)
        x_os, y_os = sample_outlet_strip(N_OUTLET_STRIP)
        bc_points  = sample_all_bc()

    optimizer.zero_grad()
    include_physics = (epoch >= CURRICULUM_EPOCHS)
    loss, loss_phys, loss_cyl, loss_inlet = compute_loss(
        model, x_c, y_c, x_nc, y_nc, x_os, y_os, bc_points, include_physics)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        tag = "BC-only" if not include_physics else "full"
        print(f"  [{tag}] epoch {epoch:>5d}: loss={loss.item():.6f}  "
              f"phys={loss_phys.item():.6f}  "
              f"cyl={loss_cyl.item():.6f}  "
              f"inlet={loss_inlet.item():.6f}")

# ── Phase 2: L-BFGS ──
print(f"\nPhase 2: L-BFGS  ({EPOCHS_LBFGS} steps)")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=0.1, max_iter=20,
    history_size=50, line_search_fn="strong_wolfe")

# Fix all points for L-BFGS so closure is deterministic
x_c,  y_c  = sample_collocation(N_COLLOCATION)
x_nc, y_nc = sample_near_cylinder(N_NEAR_CYL)
x_os, y_os = sample_outlet_strip(N_OUTLET_STRIP)
bc_points  = sample_all_bc()

for epoch in range(EPOCHS_LBFGS):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss, _, _, _ = compute_loss(model, x_c, y_c, x_nc, y_nc, x_os, y_os,
                                     bc_points, include_physics=True)
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)

    if epoch % 50 == 0:
        loss, loss_phys, loss_cyl, loss_inlet = compute_loss(
            model, x_c, y_c, x_nc, y_nc, x_os, y_os, bc_points, include_physics=True)
        print(f"  epoch {epoch:>4d}: loss={loss.item():.6f}  "
              f"phys={loss_phys.item():.6f}  "
              f"cyl={loss_cyl.item():.6f}  "
              f"inlet={loss_inlet.item():.6f}")

print("\n✓ Training complete")

# ── Visualize ──
nx, ny = 300, 150
xs = torch.linspace(X_MIN, X_MAX, nx)
ys = torch.linspace(Y_MIN, Y_MAX, ny)
X, Y = torch.meshgrid(xs, ys, indexing="ij")
xf = X.reshape(-1, 1).to(device)
yf = Y.reshape(-1, 1).to(device)

with torch.no_grad():
    u, v, p = model(xf, yf)

speed = torch.sqrt(u**2 + v**2).reshape(nx, ny).cpu().numpy()
p_np  = p.reshape(nx, ny).cpu().numpy()
X_np, Y_np = X.numpy(), Y.numpy()

dist = (X_np - CYL_X)**2 + (Y_np - CYL_Y)**2
speed[dist < CYL_R**2] = np.nan
p_np [dist < CYL_R**2] = np.nan

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

im0 = axes[0].pcolormesh(X_np, Y_np, speed, cmap="viridis", shading="auto")
axes[0].set_title(f"Velocity magnitude  |  Re = {RE}")
axes[0].set_ylabel("y")
axes[0].set_aspect("equal")
fig.colorbar(im0, ax=axes[0], label="|u|")

im1 = axes[1].pcolormesh(X_np, Y_np, p_np, cmap="RdBu_r", shading="auto")
axes[1].set_title("Pressure field")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_aspect("equal")
fig.colorbar(im1, ax=axes[1], label="p")

for ax in axes:
    circle = plt.Circle((CYL_X, CYL_Y), CYL_R, color="white",
                         ec="black", lw=1.5, zorder=5)
    ax.add_patch(circle)

plt.tight_layout()
plt.savefig("ns_pinn_cylinder_re40.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Visualization saved to ns_pinn_cylinder_re40.png")
