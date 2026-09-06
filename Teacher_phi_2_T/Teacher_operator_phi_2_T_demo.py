import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import time

from PDE_Solver import physics_params, solve_2d_phi_2_T

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    torch.cuda.empty_cache()

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 2

# Use CUDA if available, otherwise CPU (useful for local demo)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class ParallelPINOs(nn.Module):
    """
    Parallel Physics-Informed Neural Operators with per-instance weights.
    """
    def __init__(self, num_instances=4, hidden_dim=64, H=1.0):
        super(ParallelPINOs, self).__init__()
        self.num_instances = num_instances
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
        for i in range(len(weights)):
            w = weights[i].index_select(0, batch_indices)
            b = biases[i].index_select(0, batch_indices)
            x = torch.tanh(torch.bmm(x, w) + b)
        return x

    def forward(self, x, batch_indices):
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

        T_inlet = 1.0
        sigma = 0.05
        coeff = 1.0 - torch.exp(-x_raw / sigma)
        return coeff * output + T_inlet


def apply_zcs_to_points(base, z_x, z_y):
    shift = torch.stack([z_x, z_y], dim=-1)
    return base + shift


def model_batch_forward(model, base, z_x, z_y, batch_indices, scale=600.0):
    X = apply_zcs_to_points(base, z_x, z_y)
    return (model(X, batch_indices) * scale).squeeze(-1)


def PDE_loss_fd(model, base_points, phi_input, z_x, z_y, batch_indices,
                eps_x=1e-5, eps_y=1e-5):
    rho_Cp = physics_params['rho_Cp']
    k_th = physics_params['k_th']

    q = phi_input[:, :, 0]
    u = phi_input[:, :, 1]

    T_pred = model_batch_forward(model, base_points, z_x, z_y, batch_indices)
    T_x, T_y = torch.autograd.grad(
        T_pred.sum(), [z_x, z_y], create_graph=True, retain_graph=True
    )

    z_x_plus = z_x + eps_x
    z_x_minus = z_x - eps_x
    z_y_plus = z_y + eps_y
    z_y_minus = z_y - eps_y

    T_xp = model_batch_forward(model, base_points, z_x_plus, z_y, batch_indices)
    T_xm = model_batch_forward(model, base_points, z_x_minus, z_y, batch_indices)
    T_yp = model_batch_forward(model, base_points, z_x, z_y_plus, batch_indices)
    T_ym = model_batch_forward(model, base_points, z_x, z_y_minus, batch_indices)

    T_xx = (T_xp - 2.0 * T_pred + T_xm) / (eps_x ** 2)
    T_yy = (T_yp - 2.0 * T_pred + T_ym) / (eps_y ** 2)

    residual = (rho_Cp * u * T_x - k_th * (T_xx + T_yy) - q) / 1e3
    return torch.mean(residual ** 2)


def combined_BC_loss(model, left_zcs, right_zcs, bottom_zcs, top_zcs,
                     z_x_pde, z_y_pde, z_x_right, z_y_bottom, z_y_top,
                     batch_indices):
    bc_zcs = torch.cat([left_zcs, right_zcs, bottom_zcs, top_zcs], dim=1)
    T_pred_all = (model(bc_zcs, batch_indices) * 600).squeeze(-1)

    left_T = T_pred_all[:, :25]
    right_T = T_pred_all[:, 25:50]
    bottom_T = T_pred_all[:, 50:100]
    top_T = T_pred_all[:, 100:150]

    T_inlet = 600.0
    left_loss = torch.mean((left_T - T_inlet) ** 2)

    right_T_x = torch.autograd.grad(right_T.sum(), z_x_right, create_graph=True)[0]
    right_loss = torch.mean(right_T_x ** 2)

    bottom_T_y = torch.autograd.grad(bottom_T.sum(), z_y_bottom, create_graph=True)[0]
    bottom_loss = torch.mean(bottom_T_y ** 2)

    top_T_y = torch.autograd.grad(top_T.sum(), z_y_top, create_graph=True)[0]
    top_loss = torch.mean(top_T_y ** 2)

    return left_loss, right_loss, bottom_loss, top_loss


def sample_collocation_points(x_min, x_max, y_min, y_max, num_points):
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


def precompute_phi_input(pde_points, phi_dataset, y_all, x_all):
    epsilon_f = physics_params['epsilon_f']
    Sigma_f = physics_params['Sigma_f']
    U_max = physics_params['U_max']
    H = physics_params['H']

    N = phi_dataset.shape[0]
    num_points = pde_points.shape[0]
    y = pde_points[:, 1]

    query_points = pde_points[:, [1, 0]]

    phi_vals = np.empty((N, num_points), dtype=np.float64)
    for i in range(N):
        interp = RegularGridInterpolator(
            (y_all, x_all), phi_dataset[i],
            bounds_error=False, fill_value=None
        )
        phi_vals[i] = interp(query_points)

    q_pde = epsilon_f * Sigma_f * phi_vals
    u_pde = (U_max * 4.0 * y * (H - y)) / H ** 2
    u_pde = np.broadcast_to(u_pde, q_pde.shape)
    return np.stack((q_pde, u_pde), axis=-1)


def precompute_all_data(phi_dataset, x_all, y_all, num_points,
                        x_min, x_max, y_min, y_max):
    pde_points, left_points, right_points, bottom_points, top_points = \
        sample_collocation_points(x_min, x_max, y_min, y_max, num_points)

    phi_input_pde = precompute_phi_input(pde_points, phi_dataset, y_all, x_all)

    return {
        'pde_points': pde_points,
        'pde_base': torch.tensor(pde_points, dtype=torch.float64, device=device).unsqueeze(0),
        'left_base': torch.tensor(left_points, dtype=torch.float64, device=device).unsqueeze(0),
        'right_base': torch.tensor(right_points, dtype=torch.float64, device=device).unsqueeze(0),
        'bottom_base': torch.tensor(bottom_points, dtype=torch.float64, device=device).unsqueeze(0),
        'top_base': torch.tensor(top_points, dtype=torch.float64, device=device).unsqueeze(0),
        'phi_input_pde': torch.tensor(phi_input_pde, dtype=torch.float64, device=device),
    }


def prepare_batch_data(batch_idx, precomputed):
    B = len(batch_idx)
    num_points = precomputed['pde_points'].shape[0]

    phi_input_pde = precomputed['phi_input_pde'][batch_idx]
    batch_indices = torch.tensor(batch_idx, dtype=torch.long, device=device)

    z_x_pde = torch.zeros((B, num_points), dtype=torch.float64, requires_grad=True, device=device)
    z_y_pde = torch.zeros((B, num_points), dtype=torch.float64, requires_grad=True, device=device)
    z_x_right = torch.zeros((B, 25), dtype=torch.float64, requires_grad=True, device=device)
    z_y_bottom = torch.zeros((B, 50), dtype=torch.float64, requires_grad=True, device=device)
    z_y_top = torch.zeros((B, 50), dtype=torch.float64, requires_grad=True, device=device)

    pde_zcs = apply_zcs_to_points(precomputed['pde_base'], z_x_pde, z_y_pde)
    left_zcs = apply_zcs_to_points(precomputed['left_base'], z_x_pde[:, :25], z_y_pde[:, :25])
    right_zcs = apply_zcs_to_points(precomputed['right_base'], z_x_right, z_y_pde[:, :25])
    bottom_zcs = apply_zcs_to_points(precomputed['bottom_base'], z_x_pde[:, :50], z_y_bottom)
    top_zcs = apply_zcs_to_points(precomputed['top_base'], z_x_pde[:, :50], z_y_top)

    return (pde_zcs, left_zcs, right_zcs, bottom_zcs, top_zcs,
            phi_input_pde, z_x_pde, z_y_pde, z_x_right, z_y_bottom, z_y_top, batch_indices)


def compute_batch_loss(model, precomputed, data, loss_weights):
    (pde_zcs, left_zcs, right_zcs, bottom_zcs, top_zcs,
     phi_input_pde, z_x_pde, z_y_pde, z_x_right, z_y_bottom, z_y_top, batch_indices) = data

    w1, w2, w3, w4, w5 = loss_weights

    pde_loss = PDE_loss_fd(
        model, precomputed['pde_base'], phi_input_pde,
        z_x_pde, z_y_pde, batch_indices
    )

    left_loss, right_loss, bottom_loss, top_loss = combined_BC_loss(
        model, left_zcs, right_zcs, bottom_zcs, top_zcs,
        z_x_pde, z_y_pde, z_x_right, z_y_bottom, z_y_top, batch_indices
    )

    total_loss = w1 * pde_loss + w2 * left_loss + w3 * right_loss + w4 * bottom_loss + w5 * top_loss
    losses = (pde_loss, left_loss, right_loss, bottom_loss, top_loss)
    return total_loss, losses


def train_epoch_adam(model, optimizer, precomputed, N, batch_size, loss_weights):
    indices = np.random.permutation(N)
    last_losses = None

    for start in range(0, N, batch_size):
        batch_idx = indices[start:start + batch_size]
        data = prepare_batch_data(batch_idx, precomputed)

        optimizer.zero_grad()
        total_loss, losses = compute_batch_loss(model, precomputed, data, loss_weights)
        total_loss.backward()
        optimizer.step()

        last_losses = losses

    return last_losses


def predict_all(model, grid_points, num_instances, batch_size=200):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, num_instances, batch_size):
            end = min(start + batch_size, num_instances)
            batch_idx = np.arange(start, end)
            batch_indices = torch.tensor(batch_idx, dtype=torch.long, device=device)
            B = len(batch_idx)
            N = grid_points.shape[0]
            z_x = torch.zeros((B, N), dtype=torch.float64, device=device)
            z_y = torch.zeros((B, N), dtype=torch.float64, device=device)
            X = apply_zcs_to_points(grid_points, z_x, z_y)
            pred = (model(X, batch_indices) * 600).squeeze(-1)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_demo(phi_dataset, x_all, y_all,
               num_instances=4, num_epochs=2000, batch_size=2, num_points=1000,
               nx=201, ny=101):
    """Small-scale local demo training for phi -> T."""
    N = phi_dataset.shape[0]
    assert N == num_instances
    assert phi_dataset.shape[1:] == (ny, nx)

    L = physics_params['L']
    H = physics_params['H']

    model = ParallelPINOs(num_instances=N, hidden_dim=64, H=H).double().to(device)
    print(f"Model created: {N} instances, batch_size={batch_size}, num_points={num_points}, device={device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.5 * num_epochs), int(0.85 * num_epochs)], gamma=0.1
    )

    loss_weights = (1, 1, 1, 1, 1)
    w1, w2, w3, w4, w5 = loss_weights
    best_loss = float('inf')

    print("Precomputing collocation data...")
    precomputed = precompute_all_data(
        phi_dataset, x_all, y_all, num_points,
        0.0, L, 0.0, H
    )
    print("Precomputation done.")

    model_path = "Teacher_operator_phi_2_T_demo.pth"
    start_time = time.time()

    for epoch in range(num_epochs):
        losses = train_epoch_adam(model, optimizer, precomputed, N, batch_size, loss_weights)
        scheduler.step()

        pde_loss, left_loss, right_loss, bottom_loss, top_loss = losses
        total_loss = (w1 * pde_loss + w2 * left_loss + w3 * right_loss +
                      w4 * bottom_loss + w5 * top_loss)

        if epoch % 200 == 0 or epoch == num_epochs - 1:
            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch:5d}] Total: {total_loss.item():.4e}, "
                f"PDE: {pde_loss.item():.4e}, "
                f"Left: {left_loss.item():.4e}, "
                f"Right: {right_loss.item():.4e}, "
                f"Bottom: {bottom_loss.item():.4e}, "
                f"Top: {top_loss.item():.4e}, "
                f"Time: {elapsed:.1f}s"
            )

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Training complete. Best model saved to {model_path}")
    return model


def evaluate_and_plot(model, phi_dataset, x_all, y_all, nx=201, ny=101,
                       T_true_path='T_true_10000_101_201.npy'):
    L = physics_params['L']
    H = physics_params['H']
    N = phi_dataset.shape[0]

    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y)
    grid_points_single = np.column_stack([X.flatten(), Y.flatten()])
    grid_points = torch.tensor(grid_points_single, dtype=torch.float64, device=device)

    print("Running predictions...")
    T_preds = predict_all(model, grid_points, N, batch_size=N)
    T_preds = T_preds.reshape(N, ny, nx)

    print(f"Loading reference T from {T_true_path}...")
    T_true = np.load(T_true_path)[:N]  # (N, ny, nx)

    rel_l2_list = []
    fig, axes = plt.subplots(N, 3, figsize=(12, 4 * N))
    if N == 1:
        axes = axes.reshape(1, -1)

    for i in range(N):
        T_ref = T_true[i]
        rel_l2 = np.linalg.norm(T_preds[i] - T_ref) / np.linalg.norm(T_ref)
        rel_l2_list.append(rel_l2)

        vmin = min(T_preds[i].min(), T_ref.min())
        vmax = max(T_preds[i].max(), T_ref.max())

        ax = axes[i, 0]
        c = ax.contourf(X, Y, T_preds[i], levels=20, vmin=vmin, vmax=vmax)
        ax.set_title(f"Instance {i} - Predicted")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(c, ax=ax)

        ax = axes[i, 1]
        c = ax.contourf(X, Y, T_ref, levels=20, vmin=vmin, vmax=vmax)
        ax.set_title(f"Instance {i} - Reference")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(c, ax=ax)

        ax = axes[i, 2]
        err = np.abs(T_preds[i] - T_ref)
        c = ax.contourf(X, Y, err, levels=20)
        ax.set_title(f"Instance {i} - Abs Error (RelL2={rel_l2*100:.3f}%)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(c, ax=ax)

    plt.tight_layout()
    plt.savefig("Teacher_phi_2_T_demo.png", dpi=150)
    print(f"Saved figure to Teacher_phi_2_T_demo.png")

    rel_l2_arr = np.array(rel_l2_list)
    print(f"Relative L2: mean={rel_l2_arr.mean()*100:.4f}%, "
          f"min={rel_l2_arr.min()*100:.4f}%, max={rel_l2_arr.max()*100:.4f}%")

    return T_preds


def main():
    # ---- Demo hyperparameters (tuned for local machines) ----
    num_instances = 2
    batch_size = 1
    num_points = 1000
    num_epochs = 40000
    nx, ny = 201, 101

    L = physics_params['L']
    H = physics_params['H']
    x_all = np.linspace(0, L, nx)
    y_all = np.linspace(0, H, ny)

    phi_file = '1_FFT_phi_10000_201_101.npy'
    print(f"Loading {phi_file}...")
    phi_dataset = np.load(phi_file)
    phi_dataset = phi_dataset.transpose(0, 2, 1)
    phi_dataset = phi_dataset[:num_instances]
    print(f"Using first {num_instances} instances, shape {phi_dataset.shape}")

    model = train_demo(
        phi_dataset, x_all, y_all,
        num_instances=num_instances,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_points=num_points,
        nx=nx, ny=ny
    )

    evaluate_and_plot(model, phi_dataset, x_all, y_all, nx=nx, ny=ny)
    print("Demo finished.")


if __name__ == "__main__":
    main()
