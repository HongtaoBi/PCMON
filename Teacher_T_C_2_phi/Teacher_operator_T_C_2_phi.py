import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import time
import os
import argparse

from PDE_Solver import physics_params, solve_2d_T_C_2_phi

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    torch.cuda.empty_cache()

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 2

PHI_SCALE = 2.5e7


class ParallelPINOs(nn.Module):
    """
    Parallel Physics-Informed Neural Operators with per-instance weights.
    The full parameter bank lives on the GPU; forward uses batch_indices
    to slice only the active instances for the current mini-batch.
    """
    def __init__(self, num_instances=10000, hidden_dim=64, L=2.0, H=1.0):
        super(ParallelPINOs, self).__init__()
        self.num_instances = num_instances
        self.L = L
        self.H = H

        self.main_layers = [2, 32, 64, 128, hidden_dim]
        self.high_layers = [3, 32, hidden_dim]
        self.out_layers = [hidden_dim, 1]

        self.main_weights = nn.ParameterList()
        self.main_biases = nn.ParameterList()
        self.high_weights = nn.ParameterList()
        self.high_biases = nn.ParameterList()
        self.out_weights = nn.ParameterList()
        self.out_biases = nn.ParameterList()

        for layers, weights, biases in [
            (self.main_layers, self.main_weights, self.main_biases),
            (self.high_layers, self.high_weights, self.high_biases),
            (self.out_layers, self.out_weights, self.out_biases),
        ]:
            for i in range(len(layers) - 1):
                in_dim = layers[i]
                out_dim = layers[i + 1]
                w = torch.empty(num_instances, in_dim, out_dim)
                b = torch.empty(num_instances, 1, out_dim)
                bound = 1 / (in_dim ** 0.5) if in_dim > 0 else 0
                nn.init.uniform_(w, -bound, bound)
                nn.init.uniform_(b, -bound, bound)
                weights.append(nn.Parameter(w))
                biases.append(nn.Parameter(b))

    def _apply_net(self, x, weights, biases, batch_indices):
        """
        x: (B, num_points, in_dim)
        weights[i]: (num_instances, in_dim, out_dim)
        batch_indices: (B,) LongTensor on the same device as weights
        Returns: (B, num_points, out_dim)
        """
        for i in range(len(weights)):
            w = weights[i].index_select(0, batch_indices)
            b = biases[i].index_select(0, batch_indices)
            x = torch.tanh(torch.bmm(x, w) + b)
        return x

    def forward(self, x, batch_indices):
        """
        x: (B, num_points, 2) coordinates, already expanded per instance
        batch_indices: (B,) LongTensor
        Returns: (B, num_points, 1)
        """
        x_raw = x[:, :, 0:1]
        y_raw = x[:, :, 1:2]

        X_main = torch.cat([x_raw, y_raw], dim=-1)
        H_main = self._apply_net(X_main, self.main_weights, self.main_biases, batch_indices)

        cos_1 = torch.cos(1.0 * np.pi * y_raw / self.H)
        cos_2 = torch.cos(2.0 * np.pi * y_raw / self.H)
        cos_4 = torch.cos(4.0 * np.pi * y_raw / self.H)
        X_high = torch.cat([cos_1, cos_2, cos_4], dim=-1)
        H_high = self._apply_net(X_high, self.high_weights, self.high_biases, batch_indices)

        H_fuse = H_main * H_high
        output = torch.bmm(
            H_fuse,
            self.out_weights[0].index_select(0, batch_indices)
        ) + self.out_biases[0].index_select(0, batch_indices)

        # phi has vacuum BC on all four boundaries: phi = 0 at x=0, x=L, y=0, y=H
        sigma_x = 0.05
        sigma_y = 0.05
        coeff_x = 1.0 - torch.exp(-x_raw / sigma_x) - torch.exp(-(self.L - x_raw) / sigma_x)
        coeff_y = 1.0 - torch.exp(-y_raw / sigma_y) - torch.exp(-(self.H - y_raw) / sigma_y)
        coeff = coeff_x * coeff_y
        return coeff * output


def apply_zcs_to_points(base, z_x, z_y):
    """
    base: (1, num_points, 2) precomputed fixed point coordinates on GPU
    z_x, z_y: (B, N) per-instance per-point shift tensors
    Returns: (B, num_points, 2)
    """
    shift = torch.stack([z_x, z_y], dim=-1)
    return base + shift


def model_batch_forward(model, base, z_x, z_y, batch_indices, scale=PHI_SCALE):
    """
    Utility: compute scaled phi_pred from base + shifts.
    Returns: (B, N) scaled prediction.
    """
    X = apply_zcs_to_points(base, z_x, z_y)
    return (model(X, batch_indices) * scale).squeeze(-1)


def PDE_loss_fd(model, base_points, TC_input, z_x, z_y, batch_indices,
                eps_x=1e-5, eps_y=1e-5, profile=False):
    """
    Parallel 2D steady neutron diffusion residual with temperature feedback.
    PDE: D*(phi_xx + phi_yy) - removal(T)*phi + lambda*C + S_ext = 0

    base_points: (1, num_points, 2) fixed PDE collocation points on GPU
    TC_input: (B, num_points, 3) -> [T, C, S_ext]
    z_x, z_y: (B, num_points) shift variables with requires_grad=True
    batch_indices: (B,)
    """
    def sync():
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    D = physics_params['D']
    Sigma_a0 = physics_params['Sigma_a0']
    alpha_T = physics_params['alpha_T']
    T0 = physics_params['T0']
    beta = physics_params['beta']
    nu_Sigma_f = physics_params['nu_Sigma_f']
    lam = physics_params['lambda']

    T = TC_input[:, :, 0]
    C = TC_input[:, :, 1]
    S_ext = TC_input[:, :, 2]

    t0 = time.perf_counter()
    phi_pred = model_batch_forward(model, base_points, z_x, z_y, batch_indices)
    sync()
    t_forward = time.perf_counter() - t0

    t0 = time.perf_counter()
    phi_x, phi_y = torch.autograd.grad(
        phi_pred.sum(), [z_x, z_y], create_graph=True, retain_graph=True
    )
    sync()
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    z_x_plus = z_x + eps_x
    z_x_minus = z_x - eps_x
    z_y_plus = z_y + eps_y
    z_y_minus = z_y - eps_y

    phi_xp = model_batch_forward(model, base_points, z_x_plus, z_y, batch_indices)
    phi_xm = model_batch_forward(model, base_points, z_x_minus, z_y, batch_indices)
    phi_yp = model_batch_forward(model, base_points, z_x, z_y_plus, batch_indices)
    phi_ym = model_batch_forward(model, base_points, z_x, z_y_minus, batch_indices)

    phi_xx = (phi_xp - 2.0 * phi_pred + phi_xm) / (eps_x ** 2)
    phi_yy = (phi_yp - 2.0 * phi_pred + phi_ym) / (eps_y ** 2)
    sync()
    t_second = time.perf_counter() - t0

    removal = Sigma_a0 * (1.0 + alpha_T * (T - T0)) - (1.0 - beta) * nu_Sigma_f
    residual = (D * (phi_xx + phi_yy) - removal * phi_pred + lam * C + S_ext) / PHI_SCALE
    loss = torch.mean(residual ** 2)

    if profile:
        timings = {
            'pde_forward_ms': t_forward * 1000,
            'pde_first_grad_ms': t_first * 1000,
            'pde_second_fd_ms': t_second * 1000,
        }
        return loss, timings
    return loss


def combined_BC_loss(model, left_zcs, right_zcs, bottom_zcs, top_zcs,
                     batch_indices):
    """
    Vacuum boundary conditions: phi = 0 on all four boundaries.
    Compute all four boundary losses with a single model forward.
    """
    bc_zcs = torch.cat([left_zcs, right_zcs, bottom_zcs, top_zcs], dim=1)
    phi_pred_all = (model(bc_zcs, batch_indices) * PHI_SCALE).squeeze(-1)

    left_phi = phi_pred_all[:, :25]
    right_phi = phi_pred_all[:, 25:50]
    bottom_phi = phi_pred_all[:, 50:100]
    top_phi = phi_pred_all[:, 100:150]

    left_loss = torch.mean(left_phi ** 2) / (PHI_SCALE ** 2)
    right_loss = torch.mean(right_phi ** 2) / (PHI_SCALE ** 2)
    bottom_loss = torch.mean(bottom_phi ** 2) / (PHI_SCALE ** 2)
    top_loss = torch.mean(top_phi ** 2) / (PHI_SCALE ** 2)

    return left_loss, right_loss, bottom_loss, top_loss


def sample_collocation_points(x_min, x_max, y_min, y_max, num_points):
    """Sample fixed PDE and boundary collocation points once."""
    x_values = np.random.uniform(x_min + 1e-8, x_max - 1e-8, num_points)
    y_values = np.random.uniform(y_min + 1e-8, y_max - 1e-8, num_points)
    pde_points = np.column_stack((x_values, y_values))

    left_y = np.linspace(y_min, y_max, 25)
    left_points = np.column_stack([np.full_like(left_y, x_min), left_y])

    right_y = np.linspace(y_min, y_max, 25)
    right_points = np.column_stack([np.full_like(right_y, x_max), right_y])

    bottom_x = np.linspace(x_min + 0.01, x_max - 0.01, 50)
    bottom_points = np.column_stack([bottom_x, np.full_like(bottom_x, y_min)])

    top_x = np.linspace(x_min + 0.01, x_max - 0.01, 50)
    top_points = np.column_stack([top_x, np.full_like(top_x, y_max)])

    return pde_points, left_points, right_points, bottom_points, top_points


def precompute_TC_input(pde_points, T_dataset, C_dataset, y_all, x_all):
    """
    Precompute T, C, and S_ext for ALL instances on the fixed PDE collocation points.
    """
    S0 = physics_params['S0']
    x0 = physics_params['x0']
    y0 = physics_params['y0']
    sigma_x = physics_params['sigma_x']
    sigma_y = physics_params['sigma_y']

    N = T_dataset.shape[0]
    num_points = pde_points.shape[0]

    query_points = pde_points[:, [1, 0]]

    T_vals = np.empty((N, num_points), dtype=np.float64)
    C_vals = np.empty((N, num_points), dtype=np.float64)
    for i in range(N):
        interp_T = RegularGridInterpolator(
            (y_all, x_all), T_dataset[i],
            bounds_error=False, fill_value=None
        )
        interp_C = RegularGridInterpolator(
            (y_all, x_all), C_dataset[i],
            bounds_error=False, fill_value=None
        )
        T_vals[i] = interp_T(query_points)
        C_vals[i] = interp_C(query_points)

    X = pde_points[:, 0].reshape(1, -1)
    Y = pde_points[:, 1].reshape(1, -1)
    S_ext = S0 * np.exp(-((X - x0) ** 2) / (2.0 * sigma_x ** 2)
                        - ((Y - y0) ** 2) / (2.0 * sigma_y ** 2))
    S_ext = np.broadcast_to(S_ext, (N, num_points))

    return np.stack((T_vals, C_vals, S_ext), axis=-1)


def precompute_all_data(T_dataset, C_dataset, x_all, y_all, num_points,
                        x_min, x_max, y_min, y_max):
    """
    Precompute fixed collocation points and TC_input for all instances.
    Returns a dict of CPU tensors/arrays to be reused every epoch.
    """
    pde_points, left_points, right_points, bottom_points, top_points = \
        sample_collocation_points(x_min, x_max, y_min, y_max, num_points)

    TC_input_pde = precompute_TC_input(pde_points, T_dataset, C_dataset, y_all, x_all)

    return {
        'pde_points': pde_points,
        'pde_base': torch.tensor(pde_points, dtype=torch.float64, device='cuda').unsqueeze(0),
        'left_base': torch.tensor(left_points, dtype=torch.float64, device='cuda').unsqueeze(0),
        'right_base': torch.tensor(right_points, dtype=torch.float64, device='cuda').unsqueeze(0),
        'bottom_base': torch.tensor(bottom_points, dtype=torch.float64, device='cuda').unsqueeze(0),
        'top_base': torch.tensor(top_points, dtype=torch.float64, device='cuda').unsqueeze(0),
        'TC_input_pde': torch.tensor(TC_input_pde, dtype=torch.float64, device='cuda'),
    }


def prepare_batch_data(batch_idx, precomputed):
    """
    Build all tensors needed for one mini-batch by slicing precomputed data.
    """
    B = len(batch_idx)
    num_points = precomputed['pde_points'].shape[0]

    TC_input_pde = precomputed['TC_input_pde'][batch_idx]
    batch_indices = torch.tensor(batch_idx, dtype=torch.long, device='cuda')

    z_x_pde = torch.zeros((B, num_points), dtype=torch.float64, requires_grad=True, device='cuda')
    z_y_pde = torch.zeros((B, num_points), dtype=torch.float64, requires_grad=True, device='cuda')

    pde_zcs = apply_zcs_to_points(precomputed['pde_base'], z_x_pde, z_y_pde)
    left_zcs = apply_zcs_to_points(precomputed['left_base'], z_x_pde[:, :25], z_y_pde[:, :25])
    right_zcs = apply_zcs_to_points(precomputed['right_base'], z_x_pde[:, :25], z_y_pde[:, :25])
    bottom_zcs = apply_zcs_to_points(precomputed['bottom_base'], z_x_pde[:, :50], z_y_pde[:, :50])
    top_zcs = apply_zcs_to_points(precomputed['top_base'], z_x_pde[:, :50], z_y_pde[:, :50])

    return (pde_zcs, left_zcs, right_zcs, bottom_zcs, top_zcs,
            TC_input_pde, z_x_pde, z_y_pde, batch_indices)


def compute_batch_loss(model, data, loss_weights, profile=False):
    """
    Compute weighted total loss for one batch (with gradients).
    """
    def sync():
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    (pde_zcs, left_zcs, right_zcs, bottom_zcs, top_zcs,
     TC_input_pde, z_x_pde, z_y_pde, batch_indices) = data

    w1, w2, w3, w4, w5 = loss_weights

    if profile:
        pde_loss, pde_timings = PDE_loss_fd(
            model, precomputed_global['pde_base'], TC_input_pde,
            z_x_pde, z_y_pde, batch_indices, profile=True
        )
    else:
        pde_loss = PDE_loss_fd(
            model, precomputed_global['pde_base'], TC_input_pde,
            z_x_pde, z_y_pde, batch_indices
        )
        pde_timings = {}

    t0 = time.perf_counter()
    left_loss, right_loss, bottom_loss, top_loss = combined_BC_loss(
        model, left_zcs, right_zcs, bottom_zcs, top_zcs,
        batch_indices
    )
    sync()
    t_bc = time.perf_counter() - t0

    total_loss = w1 * pde_loss + w2 * left_loss + w3 * right_loss + w4 * bottom_loss + w5 * top_loss

    losses = (pde_loss, left_loss, right_loss, bottom_loss, top_loss)
    if profile:
        timings = {
            **pde_timings,
            'bc_combined_ms': t_bc * 1000,
        }
        return total_loss, losses, timings
    return total_loss, losses


# Global reference to precomputed data so PDE_loss_fd can receive base_points
precomputed_global = None


def train_epoch_adam(model, optimizer, precomputed, N,
                     batch_size, loss_weights, verbose=False):
    """Run one Adam epoch over all instances in shuffled mini-batches.
    Returns the loss tuple (pde, left, right, bottom, top) from the last mini-batch.
    """
    global precomputed_global
    precomputed_global = precomputed

    indices = np.random.permutation(N)
    last_losses = None

    for start in range(0, N, batch_size):
        batch_idx = indices[start:start + batch_size]

        t0 = time.perf_counter()
        data = prepare_batch_data(batch_idx, precomputed)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_data = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.zero_grad()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_zero = time.perf_counter() - t0

        t0 = time.perf_counter()
        total_loss, losses, step_timings = compute_batch_loss(model, data, loss_weights, profile=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_forward = time.perf_counter() - t0

        t0 = time.perf_counter()
        total_loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_backward = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_step = time.perf_counter() - t0

        last_losses = losses

        if verbose:
            print(
                f"  [Step timing] data={t_data*1000:.2f}ms | "
                f"zero_grad={t_zero*1000:.2f}ms | "
                f"forward+loss={t_forward*1000:.2f}ms | "
                f"backward={t_backward*1000:.2f}ms | "
                f"optimizer.step={t_step*1000:.2f}ms\n"
                f"    details PDE_fwd={step_timings['pde_forward_ms']:.2f}ms | "
                f"PDE_1st={step_timings['pde_first_grad_ms']:.2f}ms | "
                f"PDE_2nd_FD={step_timings['pde_second_fd_ms']:.2f}ms | "
                f"BC_combined={step_timings['bc_combined_ms']:.2f}ms"
            )

    return last_losses


def train_epoch_lbfgs(model, optimizer, precomputed, N,
                      batch_size, loss_weights):
    """Run one LBFGS step over ALL instances."""
    global precomputed_global
    precomputed_global = precomputed

    all_indices = np.arange(N)

    def closure():
        with torch.enable_grad():
            optimizer.zero_grad()
            total_loss_full = 0.0
            for start in range(0, N, batch_size):
                batch_idx = all_indices[start:start + batch_size]
                data = prepare_batch_data(batch_idx, precomputed)
                total_loss, _ = compute_batch_loss(model, data, loss_weights)
                total_loss.backward()
                total_loss_full += total_loss.item()
        return torch.tensor(total_loss_full, dtype=torch.float64, device='cuda')

    optimizer.step(closure)

    with torch.enable_grad():
        total_loss_full = 0.0
        last_losses = None
        for start in range(0, N, batch_size):
            batch_idx = all_indices[start:start + batch_size]
            data = prepare_batch_data(batch_idx, precomputed)
            total_loss, losses = compute_batch_loss(model, data, loss_weights)
            total_loss_full += total_loss.item()
            last_losses = losses
    return last_losses


def predict_all(model, grid_points, num_instances, batch_size=200):
    """Predict for all instances in batches."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, num_instances, batch_size):
            end = min(start + batch_size, num_instances)
            batch_idx = np.arange(start, end)
            batch_indices = torch.tensor(batch_idx, dtype=torch.long, device='cuda')
            B = len(batch_idx)
            N = grid_points.shape[0]
            z_x = torch.zeros((B, N), dtype=torch.float64, device='cuda')
            z_y = torch.zeros((B, N), dtype=torch.float64, device='cuda')
            X = apply_zcs_to_points(grid_points, z_x, z_y)
            pred = (model(X, batch_indices) * PHI_SCALE).squeeze(-1)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_P_PINOs(T_dataset, C_dataset, x_all, y_all, nx=201, ny=101,
                num_epochs_adam=40000, num_epochs_lbfgs=0,
                batch_size=50, num_points=5000,
                model_path_adam="Teacher_operator_T_C_2_phi_adam_fd.pth",
                model_path_lbfgs="Teacher_operator_T_C_2_phi_LBFGS_fd.pth",
                log_interval=500):
    """Train ParallelPINOs for T, C -> phi using AD for first derivatives and FD for second."""
    N = T_dataset.shape[0]
    assert T_dataset.shape[1:] == (ny, nx), \
        f"T_dataset shape {T_dataset.shape} does not match (N, ny={ny}, nx={nx})"
    assert C_dataset.shape[1:] == (ny, nx), \
        f"C_dataset shape {C_dataset.shape} does not match (N, ny={ny}, nx={nx})"

    L = physics_params['L']
    H = physics_params['H']
    x_min, x_max = 0.0, L
    y_min, y_max = 0.0, H

    model = ParallelPINOs(num_instances=N, hidden_dim=64, L=L, H=H).double().cuda()
    print(f"Model created with {N} instances, batch_size={batch_size}, num_points={num_points}")

    print("Running in eager mode (torch.compile disabled).")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.3 * num_epochs_adam), int(0.8 * num_epochs_adam)], gamma=0.1
    )

    loss_weights = (1e4, 1, 1, 1, 1)
    w1, w2, w3, w4, w5 = loss_weights
    best_loss = float('inf')

    print("Precomputing collocation points and TC_input for all instances...")
    precomputed = precompute_all_data(
        T_dataset, C_dataset, x_all, y_all, num_points,
        x_min, x_max, y_min, y_max
    )
    print("Precomputation done.")

    start_time = time.time()

    for epoch in range(num_epochs_adam):
        t_epoch = time.perf_counter()
        losses = train_epoch_adam(
            model, optimizer, precomputed, N,
            batch_size, loss_weights, verbose=False
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_epoch = time.perf_counter() - t_epoch

        t0 = time.perf_counter()
        scheduler.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_sched = time.perf_counter() - t0

        pde_loss, left_loss, right_loss, bottom_loss, top_loss = losses
        total_loss = (w1 * pde_loss + w2 * left_loss + w3 * right_loss +
                      w4 * bottom_loss + w5 * top_loss)

        if epoch % log_interval == 0 or epoch == num_epochs_adam - 1:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[Adam] Epoch {epoch:6d}, LR: {current_lr:.2e}, "
                f"Total: {total_loss.item():.4e}, "
                f"PDE: {(w1 * pde_loss).item():.4e}, "
                f"Left: {(w2 * left_loss).item():.4e}, "
                f"Right: {(w3 * right_loss).item():.4e}, "
                f"Bottom: {(w4 * bottom_loss).item():.4e}, "
                f"Top: {(w5 * top_loss).item():.4e}, "
                f"Epoch: {t_epoch*1000:.2f}ms, Sched: {t_sched*1000:.2f}ms, "
                f"Time: {elapsed:.1f}s"
            )

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save(model.state_dict(), model_path_adam)

    if num_epochs_lbfgs > 0:
        print("L-BFGS Starts!")
        model.load_state_dict(torch.load(model_path_adam))
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=0.001,
            max_iter=20, history_size=50,
            tolerance_grad=1e-9, tolerance_change=1e-9
        )

        for epoch in range(num_epochs_lbfgs):
            losses = train_epoch_lbfgs(
                model, optimizer, precomputed, N,
                batch_size, loss_weights
            )

            pde_loss, left_loss, right_loss, bottom_loss, top_loss = losses
            total_loss = (w1 * pde_loss + w2 * left_loss + w3 * right_loss +
                          w4 * bottom_loss + w5 * top_loss)

            if epoch % 20 == 0 or epoch == num_epochs_lbfgs - 1:
                elapsed = time.time() - start_time
                print(
                    f"[LBFGS] Epoch {epoch + num_epochs_adam:6d}, "
                    f"Total: {total_loss.item():.4e}, "
                    f"PDE: {pde_loss.item():.4e}, "
                    f"Left: {left_loss.item():.4e}, "
                    f"Right: {right_loss.item():.4e}, "
                    f"Bottom: {bottom_loss.item():.4e}, "
                    f"Top: {top_loss.item():.4e}, "
                    f"Time: {elapsed:.1f}s"
                )

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                torch.save(model.state_dict(), model_path_lbfgs)

    final_model_path = model_path_lbfgs if (num_epochs_lbfgs > 0 and os.path.exists(model_path_lbfgs)) else model_path_adam
    model.load_state_dict(torch.load(final_model_path))
    print(f"Final model loaded from {final_model_path}")

    return final_model_path


def parse_args():
    parser = argparse.ArgumentParser(description='Train P-PINOs T, C -> phi in chunks.')
    parser.add_argument('--chunk_id', type=int, default=0, help='Chunk index (0,1,2,...).')
    parser.add_argument('--start_idx', type=int, default=None, help='First instance index in the full dataset.')
    parser.add_argument('--end_idx', type=int, default=None, help='Last+1 instance index in the full dataset.')
    parser.add_argument('--num_epochs_adam', type=int, default=40000, help='Number of Adam epochs.')
    parser.add_argument('--num_epochs_lbfgs', type=int, default=0, help='Number of LBFGS epochs (0 to skip).')
    parser.add_argument('--batch_size', type=int, default=50, help='Mini-batch size.')
    parser.add_argument('--num_points', type=int, default=5000, help='Number of PDE collocation points.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for collocation sampling and initialization.')
    parser.add_argument('--log_interval', type=int, default=1000, help='How often to print loss.')
    parser.add_argument('--skip_plot', action='store_true', help='Skip final prediction/plot.')
    parser.add_argument('--T_file', type=str, default='1_FFT_T_10000_101_201.npy',
                        help='Path to the full T dataset.')
    parser.add_argument('--C_file', type=str, default='1_FFT_C_10000_101_201.npy',
                        help='Path to the full C dataset.')
    return parser.parse_args()


def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    set_seed(args.seed)

    chunk_id = args.chunk_id
    start_idx = args.start_idx if args.start_idx is not None else chunk_id * 1000
    end_idx = args.end_idx if args.end_idx is not None else start_idx + 1000

    L = physics_params['L']
    H = physics_params['H']
    nx, ny = 201, 101

    x_all = np.linspace(0, L, nx)
    y_all = np.linspace(0, H, ny)

    T_dataset = np.load(args.T_file)
    C_dataset = np.load(args.C_file)
    print(f"Loaded T: {T_dataset.shape}, C: {C_dataset.shape}")

    N_full = T_dataset.shape[0]
    if end_idx > N_full:
        end_idx = N_full
    T_dataset = T_dataset[start_idx:end_idx]
    C_dataset = C_dataset[start_idx:end_idx]
    N = T_dataset.shape[0]
    print(f"Chunk {chunk_id}: using instances [{start_idx}, {end_idx}), count = {N}")

    model_path_adam = f"Teacher_operator_T_C_2_phi_adam_fd_chunk_{chunk_id}.pth"
    model_path_lbfgs = f"Teacher_operator_T_C_2_phi_LBFGS_fd_chunk_{chunk_id}.pth"

    final_model_path = run_P_PINOs(
        T_dataset, C_dataset, x_all, y_all, nx=nx, ny=ny,
        num_epochs_adam=args.num_epochs_adam,
        num_epochs_lbfgs=args.num_epochs_lbfgs,
        batch_size=args.batch_size,
        num_points=args.num_points,
        model_path_adam=model_path_adam,
        model_path_lbfgs=model_path_lbfgs,
        log_interval=args.log_interval
    )

    if not args.skip_plot:
        print("Running final predictions and saving plot...")
        x = np.linspace(0, L, nx)
        y = np.linspace(0, H, ny)
        X, Y = np.meshgrid(x, y)
        grid_points_single = np.column_stack([X.flatten(), Y.flatten()])
        grid_points = torch.tensor(grid_points_single, dtype=torch.float64, device='cuda')

        model = ParallelPINOs(num_instances=N, hidden_dim=64, L=L, H=H).double().cuda()
        model.load_state_dict(torch.load(final_model_path))
        phi_preds = predict_all(model, grid_points, N, batch_size=N)
        phi_preds = phi_preds.reshape(N, ny, nx)

        phi_true_file = 'phi_true_10000_201_101.npy'
        if os.path.exists(phi_true_file):
            phi_true = np.load(phi_true_file).transpose(0, 2, 1)[start_idx:end_idx]
            rel_l2_list = []
            for i in range(min(N, 4)):
                rel_l2 = np.linalg.norm(phi_preds[i] - phi_true[i]) / np.linalg.norm(phi_true[i])
                rel_l2_list.append(rel_l2)
                print(f"Instance {start_idx + i}: Relative L2 = {rel_l2 * 100:.4f}%")

            fig, axes = plt.subplots(min(N, 4), 3, figsize=(12, 4 * min(N, 4)))
            if min(N, 4) == 1:
                axes = axes.reshape(1, -1)
            for i in range(min(N, 4)):
                vmin = min(phi_preds[i].min(), phi_true[i].min())
                vmax = max(phi_preds[i].max(), phi_true[i].max())

                ax = axes[i, 0]
                c = ax.contourf(X, Y, phi_preds[i], levels=20, vmin=vmin, vmax=vmax)
                ax.set_title(f"Instance {start_idx + i} - Predicted")
                plt.colorbar(c, ax=ax)

                ax = axes[i, 1]
                c = ax.contourf(X, Y, phi_true[i], levels=20, vmin=vmin, vmax=vmax)
                ax.set_title(f"Instance {start_idx + i} - Reference")
                plt.colorbar(c, ax=ax)

                ax = axes[i, 2]
                err = np.abs(phi_preds[i] - phi_true[i])
                c = ax.contourf(X, Y, err, levels=20)
                ax.set_title(f"Instance {start_idx + i} - Abs Error")
                plt.colorbar(c, ax=ax)

            plt.tight_layout()
            plot_path = f"Teacher_T_C_2_phi_chunk_{chunk_id}.png"
            plt.savefig(plot_path, dpi=150)
            print(f"Saved figure to {plot_path}")
        else:
            print(f"Reference file {phi_true_file} not found, skipping plot.")

    print(f"Chunk {chunk_id} training complete. Checkpoint saved to {final_model_path}")


if __name__ == "__main__":
    main()
