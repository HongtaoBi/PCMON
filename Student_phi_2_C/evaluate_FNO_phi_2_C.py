"""
Evaluate a trained FNO student model for phi -> C.

Compares FNO predictions against:
  1. Teacher labels (C_pred_10000_101_201.npy)
  2. True reference solutions (C_true_10000_101_201.npy)

Usage:
    python evaluate_FNO_phi_2_C.py
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from Student_FNO_phi_2_C import FNO2d, L, H, C_SCALE


def load_data(phi_file, teacher_file, true_file):
    """Load phi, teacher labels, and true reference solutions."""
    phi = np.load(phi_file)              # (N, nx, ny)
    phi = phi.transpose(0, 2, 1)         # -> (N, ny, nx)

    C_teacher = np.load(teacher_file)    # (N, ny, nx)
    C_true = np.load(true_file)          # (N, ny, nx)

    assert phi.shape == C_teacher.shape == C_true.shape, \
        f"phi {phi.shape}, teacher {C_teacher.shape}, true {C_true.shape} must match"
    print(f"Loaded phi: {phi.shape}, C_teacher: {C_teacher.shape}, C_true: {C_true.shape}")
    return phi, C_teacher, C_true


def split_train_val(phi, C_teacher, C_true, train_ratio=0.9, seed=42):
    """Return only the validation split (same seed as training)."""
    N = phi.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(N * train_ratio)
    return phi[perm[n_train:]], C_teacher[perm[n_train:]], C_true[perm[n_train:]]


def relative_l2(pred, true, axis=None):
    """Compute relative L2 error."""
    if axis is None:
        return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)
    return np.linalg.norm(pred - true, axis=axis) / (np.linalg.norm(true, axis=axis) + 1e-12)


def predict_fno(model, phi_val, phi_mean, phi_std, device, batch_size=32):
    """Run FNO inference on validation phi fields and return original-scale C."""
    N_val = phi_val.shape[0]

    # Normalize phi the same way as training
    phi_norm = (phi_val - phi_mean) / (phi_std + 1e-12)
    phi_tensor = torch.tensor(phi_norm, dtype=torch.float32, device=device).unsqueeze(1)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, N_val, batch_size):
            end = min(start + batch_size, N_val)
            pred = model(phi_tensor[start:end])  # normalized C / C_SCALE
            all_preds.append((pred.squeeze(1) * C_SCALE).cpu().numpy())  # back to original scale

    return np.concatenate(all_preds, axis=0)


def plot_comparison(X, Y, phi_i, C_pred, C_teacher, C_true, idx, save_path):
    """Plot phi, FNO prediction, teacher, true reference, and error maps."""
    rel_l2_teacher = relative_l2(C_pred, C_teacher)
    rel_l2_true = relative_l2(C_pred, C_true)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Row 1: phi, FNO prediction, teacher
    ax = axes[0, 0]
    c = ax.contourf(X, Y, phi_i, levels=20, cmap='viridis')
    ax.set_title(f"Instance {idx}: phi")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    vmin = min(C_pred.min(), C_teacher.min(), C_true.min())
    vmax = max(C_pred.max(), C_teacher.max(), C_true.max())

    ax = axes[0, 1]
    c = ax.contourf(X, Y, C_pred, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("FNO Prediction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[0, 2]
    c = ax.contourf(X, Y, C_teacher, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("Teacher Label")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    # Row 2: true reference, error vs teacher, error vs true
    ax = axes[1, 0]
    c = ax.contourf(X, Y, C_true, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("True Reference Solution")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 1]
    err_teacher = np.abs(C_pred - C_teacher)
    c = ax.contourf(X, Y, err_teacher, levels=20, cmap='coolwarm')
    ax.set_title(f"|FNO - Teacher| (RelL2={rel_l2_teacher*100:.3f}%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 2]
    err_true = np.abs(C_pred - C_true)
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
    parser = argparse.ArgumentParser(description='Evaluate trained FNO student for phi -> C.')
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--teacher_file', type=str, default='C_pred_10000_101_201.npy')
    parser.add_argument('--true_file', type=str, default='C_true_10000_101_201.npy')
    parser.add_argument('--model', type=str, default='Student_FNO_phi_2_C_1.pth')
    parser.add_argument('--norm_file', type=str, default='phi_normalization.npz')
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
    phi_mean = norm['mean']
    phi_std = norm['std']
    print(f"Loaded phi normalization: mean={phi_mean:.6e}, std={phi_std:.6e}")

    # Load data and validation split
    phi, C_teacher, C_true = load_data(args.phi_file, args.teacher_file, args.true_file)
    phi_val, C_teacher_val, C_true_val = split_train_val(phi, C_teacher, C_true, seed=args.seed)
    print(f"Validation: {phi_val.shape[0]} instances")

    # Load model
    model = FNO2d(width=args.width, modes1=args.modes1, modes2=args.modes2,
                  n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")

    # FNO predictions
    print("Running FNO inference...")
    C_fno_val = predict_fno(model, phi_val, phi_mean, phi_std, device, batch_size=args.batch_size)
    print(f"FNO predictions: {C_fno_val.shape}")

    # Relative L2 vs teacher labels
    rel_l2_teacher = relative_l2(C_fno_val.reshape(C_fno_val.shape[0], -1),
                                  C_teacher_val.reshape(C_teacher_val.shape[0], -1),
                                  axis=1)
    print(f"\nFNO vs Teacher Relative L2:")
    print(f"  mean = {rel_l2_teacher.mean()*100:.4f}%")
    print(f"  min  = {rel_l2_teacher.min()*100:.4f}%")
    print(f"  max  = {rel_l2_teacher.max()*100:.4f}%")

    # Relative L2 vs true reference
    rel_l2_true = relative_l2(C_fno_val.reshape(C_fno_val.shape[0], -1),
                               C_true_val.reshape(C_true_val.shape[0], -1),
                               axis=1)
    print(f"\nFNO vs True Reference Relative L2:")
    print(f"  mean = {rel_l2_true.mean()*100:.4f}%")
    print(f"  min  = {rel_l2_true.min()*100:.4f}%")
    print(f"  max  = {rel_l2_true.max()*100:.4f}%")

    # Sanity check: teacher vs true reference
    rel_l2_teacher_true = relative_l2(C_teacher_val.reshape(C_teacher_val.shape[0], -1),
                                       C_true_val.reshape(C_true_val.shape[0], -1),
                                       axis=1)
    print(f"\nTeacher vs True Reference Relative L2:")
    print(f"  mean = {rel_l2_teacher_true.mean()*100:.4f}%")
    print(f"  min  = {rel_l2_teacher_true.min()*100:.4f}%")
    print(f"  max  = {rel_l2_teacher_true.max()*100:.4f}%")

    # Plot a few random validation samples
    rng = np.random.default_rng(args.seed)
    ny, nx = phi_val.shape[1], phi_val.shape[2]
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    plot_idx = rng.choice(phi_val.shape[0], size=min(args.num_plot, phi_val.shape[0]), replace=False)
    for idx in plot_idx:
        save_path = f"FNO_phi2C_eval_instance_{idx}.png"
        plot_comparison(X, Y, phi_val[idx], C_fno_val[idx],
                        C_teacher_val[idx], C_true_val[idx], idx, save_path)

    print("Done.")


# "C:\python\Program Files\Python310\python.exe" "C:\python\PythonFiles\CNRS Benchmark\2D_PCMON\Student_phi_2_C\evaluate_FNO_phi_2_C.py"
# CUDA available: True
# Using device: cuda
# Loaded phi normalization: mean=5.227624e+06, std=5.561947e+06
# Loaded phi: (10000, 101, 201), C_teacher: (10000, 101, 201), C_true: (10000, 101, 201)
# Validation: 1000 instances
# Loaded model from Student_FNO_phi_2_C_1.pth
# Running FNO inference...
# FNO predictions: (1000, 101, 201)
#
# FNO vs Teacher Relative L2:
#   mean = 0.5086%
#   min  = 0.1360%
#   max  = 3.0736%
#
# FNO vs True Reference Relative L2:
#   mean = 0.5916%
#   min  = 0.2695%
#   max  = 2.2323%
#
# Teacher vs True Reference Relative L2:
#   mean = 0.6236%
#   min  = 0.2148%
#   max  = 3.5720%
# Saved figure to FNO_phi2C_eval_instance_654.png
# Saved figure to FNO_phi2C_eval_instance_89.png
# Saved figure to FNO_phi2C_eval_instance_773.png
# Done.
#
# Process finished with exit code 0


if __name__ == "__main__":
    main()
