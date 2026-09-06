"""
Evaluate a trained FNO student model for (T, C) -> phi.

Compares FNO predictions against:
  1. Teacher labels (phi_pred_10000_201_101.npy)
  2. True reference solutions (phi_true_10000_201_101.npy)

Usage:
    python evaluate_FNO_T_C_2_phi.py
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from Student_FNO_T_C_2_phi import FNO2d, L, H, PHI_SCALE


def _match_spatial_shape(arr, target_shape):
    """Ensure arr has the same (ny, nx) spatial shape as target_shape."""
    if arr.shape[1:] == target_shape[1:]:
        return arr
    if arr.shape[1] == target_shape[2] and arr.shape[2] == target_shape[1]:
        return arr.transpose(0, 2, 1)
    raise ValueError(
        f"Cannot reconcile spatial shape {arr.shape[1:]} with target {target_shape[1:]}"
    )


def load_data(T_file, C_file, teacher_file, true_file):
    """Load T, C, teacher labels, and true reference solutions."""
    T = np.load(T_file)                 # (N, ny, nx)
    C = np.load(C_file)                 # (N, ny, nx)
    phi_teacher = np.load(teacher_file) # (N, ny, nx) or (N, nx, ny)
    phi_teacher = _match_spatial_shape(phi_teacher, T.shape)
    phi_true = np.load(true_file)       # (N, ny, nx) or (N, nx, ny)
    phi_true = _match_spatial_shape(phi_true, T.shape)

    assert T.shape == C.shape == phi_teacher.shape == phi_true.shape, \
        f"T {T.shape}, C {C.shape}, teacher {phi_teacher.shape}, true {phi_true.shape} must match"
    print(f"Loaded T: {T.shape}, C: {C.shape}, phi_teacher: {phi_teacher.shape}, phi_true: {phi_true.shape}")
    return T, C, phi_teacher, phi_true


def split_train_val(T, C, phi_teacher, phi_true, train_ratio=0.9, seed=42):
    """Return only the validation split (same seed as training)."""
    N = T.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(N * train_ratio)
    return (T[perm[n_train:]], C[perm[n_train:]],
            phi_teacher[perm[n_train:]], phi_true[perm[n_train:]])


def relative_l2(pred, true, axis=None):
    """Compute relative L2 error."""
    if axis is None:
        return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)
    return np.linalg.norm(pred - true, axis=axis) / (np.linalg.norm(true, axis=axis) + 1e-12)


def predict_fno(model, T_val, C_val, T_mean, T_std, C_mean, C_std, device, batch_size=32):
    """Run FNO inference on validation T/C fields and return original-scale phi."""
    N_val = T_val.shape[0]

    T_norm = (T_val - T_mean) / (T_std + 1e-12)
    C_norm = (C_val - C_mean) / (C_std + 1e-12)
    inputs = np.stack([T_norm, C_norm], axis=1)  # (N, 2, ny, nx)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, N_val, batch_size):
            end = min(start + batch_size, N_val)
            pred = model(inputs_tensor[start:end])  # (B, 1, ny, nx), normalized phi
            all_preds.append((pred.squeeze(1) * PHI_SCALE).cpu().numpy())  # original scale

    return np.concatenate(all_preds, axis=0)


def plot_comparison(X, Y, T_i, C_i, phi_pred, phi_teacher, phi_true, idx, save_path):
    """Plot T, C, FNO prediction, teacher, true reference, and error maps."""
    rel_l2_teacher = relative_l2(phi_pred, phi_teacher)
    rel_l2_true = relative_l2(phi_pred, phi_true)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    c = ax.contourf(X, Y, T_i, levels=20, cmap='viridis')
    ax.set_title(f"Instance {idx}: T")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[0, 1]
    c = ax.contourf(X, Y, C_i, levels=20, cmap='plasma')
    ax.set_title(f"Instance {idx}: C")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    vmin = min(phi_pred.min(), phi_teacher.min(), phi_true.min())
    vmax = max(phi_pred.max(), phi_teacher.max(), phi_true.max())

    ax = axes[0, 2]
    c = ax.contourf(X, Y, phi_pred, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("FNO Prediction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 0]
    c = ax.contourf(X, Y, phi_teacher, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("Teacher Label")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 1]
    c = ax.contourf(X, Y, phi_true, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("True Reference Solution")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 2]
    err_true = np.abs(phi_pred - phi_true)
    c = ax.contourf(X, Y, err_true, levels=20, cmap='coolwarm')
    ax.set_title(f"|FNO - True| (RelL2={rel_l2_true*100:.3f}%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained FNO student for (T, C) -> phi.')
    parser.add_argument('--T_file', type=str, default='1_FFT_T_10000_101_201.npy')
    parser.add_argument('--C_file', type=str, default='1_FFT_C_10000_101_201.npy')
    parser.add_argument('--teacher_file', type=str, default='phi_pred_10000_201_101.npy')
    parser.add_argument('--true_file', type=str, default='phi_true_10000_201_101.npy')
    parser.add_argument('--model', type=str, default='Student_FNO_T_C_2_phi_1.pth')
    parser.add_argument('--norm_file', type=str, default='T_C_2_phi_normalization.npz')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--modes1', type=int, default=16)
    parser.add_argument('--modes2', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--num_plot', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load normalization params (same as training)
    norm = np.load(args.norm_file)
    T_mean = norm['T_mean']
    T_std = norm['T_std']
    C_mean = norm['C_mean']
    C_std = norm['C_std']
    phi_scale = norm['phi_scale']
    print(f"Loaded normalization: T_mean={T_mean:.6f}, T_std={T_std:.6f}, "
          f"C_mean={C_mean:.6f}, C_std={C_std:.6f}, phi_scale={phi_scale:.6e}")

    # Load data and validation split
    T, C, phi_teacher, phi_true = load_data(
        args.T_file, args.C_file, args.teacher_file, args.true_file
    )
    T_val, C_val, phi_teacher_val, phi_true_val = split_train_val(
        T, C, phi_teacher, phi_true, seed=args.seed
    )
    print(f"Validation: {T_val.shape[0]} instances")

    # Load model
    model = FNO2d(width=args.width, modes1=args.modes1, modes2=args.modes2,
                  n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")

    # FNO predictions
    print("Running FNO inference...")
    phi_fno_val = predict_fno(
        model, T_val, C_val, T_mean, T_std, C_mean, C_std, device, batch_size=args.batch_size
    )
    print(f"FNO predictions: {phi_fno_val.shape}")

    # Relative L2 vs teacher labels
    rel_l2_teacher = relative_l2(
        phi_fno_val.reshape(phi_fno_val.shape[0], -1),
        phi_teacher_val.reshape(phi_teacher_val.shape[0], -1),
        axis=1
    )
    print(f"\nFNO vs Teacher Relative L2:")
    print(f"  mean = {rel_l2_teacher.mean()*100:.4f}%")
    print(f"  min  = {rel_l2_teacher.min()*100:.4f}%")
    print(f"  max  = {rel_l2_teacher.max()*100:.4f}%")

    # Relative L2 vs true reference
    rel_l2_true = relative_l2(
        phi_fno_val.reshape(phi_fno_val.shape[0], -1),
        phi_true_val.reshape(phi_true_val.shape[0], -1),
        axis=1
    )
    print(f"\nFNO vs True Reference Relative L2:")
    print(f"  mean = {rel_l2_true.mean()*100:.4f}%")
    print(f"  min  = {rel_l2_true.min()*100:.4f}%")
    print(f"  max  = {rel_l2_true.max()*100:.4f}%")

    # Sanity check: teacher vs true reference
    rel_l2_teacher_true = relative_l2(
        phi_teacher_val.reshape(phi_teacher_val.shape[0], -1),
        phi_true_val.reshape(phi_true_val.shape[0], -1),
        axis=1
    )
    print(f"\nTeacher vs True Reference Relative L2:")
    print(f"  mean = {rel_l2_teacher_true.mean()*100:.4f}%")
    print(f"  min  = {rel_l2_teacher_true.min()*100:.4f}%")
    print(f"  max  = {rel_l2_teacher_true.max()*100:.4f}%")

    # Plot a few random validation samples
    rng = np.random.default_rng(args.seed)
    ny, nx = T_val.shape[1], T_val.shape[2]
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    plot_idx = rng.choice(T_val.shape[0], size=min(args.num_plot, T_val.shape[0]), replace=False)
    for idx in plot_idx:
        save_path = f"FNO_T_C_2_phi_eval_instance_{idx}.png"
        plot_comparison(
            X, Y, T_val[idx], C_val[idx], phi_fno_val[idx],
            phi_teacher_val[idx], phi_true_val[idx], idx, save_path
        )

    print("Done.")

# "C:\python\Program Files\Python310\python.exe" "C:\python\PythonFiles\CNRS Benchmark\2D_PCMON\Student_T_C_2_phi\evaluate_FNO_T_C_2_phi.py"
# CUDA available: True
# Using device: cuda
# Loaded normalization: T_mean=789.214798, T_std=162.382917, C_mean=198724.758330, C_std=145147.574705, phi_scale=2.500000e+07
# Loaded T: (10000, 101, 201), C: (10000, 101, 201), phi_teacher: (10000, 101, 201), phi_true: (10000, 101, 201)
# Validation: 1000 instances
# Loaded model from Student_FNO_T_C_2_phi_1.pth
# Running FNO inference...
# FNO predictions: (1000, 101, 201)
#
# FNO vs Teacher Relative L2:
#   mean = 0.1086%
#   min  = 0.0532%
#   max  = 0.5745%
#
# FNO vs True Reference Relative L2:
#   mean = 0.0934%
#   min  = 0.0555%
#   max  = 0.5699%
#
# Teacher vs True Reference Relative L2:
#   mean = 0.0779%
#   min  = 0.0177%
#   max  = 0.5741%
# Saved figure to FNO_T_C_2_phi_eval_instance_654.png
# Saved figure to FNO_T_C_2_phi_eval_instance_89.png
# Saved figure to FNO_T_C_2_phi_eval_instance_773.png
# Done.
#
# Process finished with exit code 0



if __name__ == "__main__":
    main()
