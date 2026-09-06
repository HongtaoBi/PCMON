"""
Analyze the worst-outlier validation samples for the FNO student (phi -> C).

Prints the indices with largest errors and compares:
  - FNO vs Teacher error
  - FNO vs True Reference error
  - phi statistics of the worst cases

Also plots the single worst outlier.

Usage:
    python analyze_FNO_outliers.py
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from Student_FNO_phi_2_C import FNO2d, L, H, C_SCALE


def load_data(phi_file, teacher_file, true_file):
    phi = np.load(phi_file).transpose(0, 2, 1)       # (N, ny, nx)
    C_teacher = np.load(teacher_file)                 # (N, ny, nx)
    C_true = np.load(true_file)                       # (N, ny, nx)
    return phi, C_teacher, C_true


def split_train_val(phi, C_teacher, C_true, train_ratio=0.9, seed=42):
    N = phi.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(N * train_ratio)
    return phi[perm[n_train:]], C_teacher[perm[n_train:]], C_true[perm[n_train:]], perm[n_train:]


def relative_l2(pred, true, axis=None):
    if axis is None:
        return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)
    return np.linalg.norm(pred - true, axis=axis) / (np.linalg.norm(true, axis=axis) + 1e-12)


def predict_fno(model, phi_val, phi_mean, phi_std, device, batch_size=32):
    """Run FNO inference and return original-scale C."""
    phi_norm = (phi_val - phi_mean) / (phi_std + 1e-12)
    phi_tensor = torch.tensor(phi_norm, dtype=torch.float32, device=device).unsqueeze(1)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, phi_val.shape[0], batch_size):
            end = min(start + batch_size, phi_val.shape[0])
            pred = model(phi_tensor[start:end])  # normalized C / C_SCALE
            all_preds.append((pred.squeeze(1) * C_SCALE).cpu().numpy())  # back to original scale
    return np.concatenate(all_preds, axis=0)


def phi_statistics(phi_i):
    """Return key statistics for a single phi field."""
    return {
        'min': phi_i.min(),
        'max': phi_i.max(),
        'mean': phi_i.mean(),
        'std': phi_i.std(),
        'range': phi_i.max() - phi_i.min(),
        'peak_to_mean': phi_i.max() / (phi_i.mean() + 1e-12),
    }


def print_outlier_table(phi_val, C_fno_val, C_teacher_val, C_true_val, val_perm, top_k=10):
    rel_l2_teacher = relative_l2(C_fno_val.reshape(C_fno_val.shape[0], -1),
                                  C_teacher_val.reshape(C_teacher_val.shape[0], -1),
                                  axis=1)
    rel_l2_true = relative_l2(C_fno_val.reshape(C_fno_val.shape[0], -1),
                               C_true_val.reshape(C_true_val.shape[0], -1),
                               axis=1)
    rel_l2_teacher_true = relative_l2(C_teacher_val.reshape(C_teacher_val.shape[0], -1),
                                       C_true_val.reshape(C_true_val.shape[0], -1),
                                       axis=1)

    # Indices in validation array
    worst_teacher_idx = np.argsort(rel_l2_teacher)[-top_k:][::-1]
    worst_true_idx = np.argsort(rel_l2_true)[-top_k:][::-1]
    worst_teacher_true_idx = np.argsort(rel_l2_teacher_true)[-top_k:][::-1]

    print("\n" + "="*80)
    print(f"Top {top_k} worst cases: FNO vs Teacher")
    print("="*80)
    print(f"{'val_idx':>8} {'orig_idx':>10} {'FNO_vs_Teacher':>16} {'FNO_vs_True':>14} {'Teacher_vs_True':>18} {'phi_max':>12} {'phi_range':>14}")
    for idx in worst_teacher_idx:
        orig_idx = val_perm[idx]
        stats = phi_statistics(phi_val[idx])
        print(f"{idx:8d} {orig_idx:10d} {rel_l2_teacher[idx]*100:15.4f}% {rel_l2_true[idx]*100:13.4f}% {rel_l2_teacher_true[idx]*100:17.4f}% {stats['max']:12.4e} {stats['range']:14.4e}")

    print("\n" + "="*80)
    print(f"Top {top_k} worst cases: FNO vs True Reference")
    print("="*80)
    print(f"{'val_idx':>8} {'orig_idx':>10} {'FNO_vs_Teacher':>16} {'FNO_vs_True':>14} {'Teacher_vs_True':>18} {'phi_max':>12} {'phi_range':>14}")
    for idx in worst_true_idx:
        orig_idx = val_perm[idx]
        stats = phi_statistics(phi_val[idx])
        print(f"{idx:8d} {orig_idx:10d} {rel_l2_teacher[idx]*100:15.4f}% {rel_l2_true[idx]*100:13.4f}% {rel_l2_teacher_true[idx]*100:17.4f}% {stats['max']:12.4e} {stats['range']:14.4e}")

    print("\n" + "="*80)
    print(f"Top {top_k} worst cases: Teacher vs True Reference")
    print("="*80)
    print(f"{'val_idx':>8} {'orig_idx':>10} {'FNO_vs_Teacher':>16} {'FNO_vs_True':>14} {'Teacher_vs_True':>18} {'phi_max':>12} {'phi_range':>14}")
    for idx in worst_teacher_true_idx:
        orig_idx = val_perm[idx]
        stats = phi_statistics(phi_val[idx])
        print(f"{idx:8d} {orig_idx:10d} {rel_l2_teacher[idx]*100:15.4f}% {rel_l2_true[idx]*100:13.4f}% {rel_l2_teacher_true[idx]*100:17.4f}% {stats['max']:12.4e} {stats['range']:14.4e}")

    return worst_teacher_idx, worst_true_idx, worst_teacher_true_idx


def plot_worst(X, Y, phi_i, C_fno, C_teacher, C_true, idx, orig_idx, save_path):
    rel_l2_teacher = relative_l2(C_fno, C_teacher)
    rel_l2_true = relative_l2(C_fno, C_true)
    rel_l2_teacher_true = relative_l2(C_teacher, C_true)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    c = ax.contourf(X, Y, phi_i, levels=20, cmap='viridis')
    ax.set_title(f"Instance {orig_idx} (val idx {idx}): phi")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    vmin = min(C_fno.min(), C_teacher.min(), C_true.min())
    vmax = max(C_fno.max(), C_teacher.max(), C_true.max())

    ax = axes[0, 1]
    c = ax.contourf(X, Y, C_fno, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("FNO Prediction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[0, 2]
    c = ax.contourf(X, Y, C_teacher, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title(f"Teacher (vs True {rel_l2_teacher_true*100:.3f}%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 0]
    c = ax.contourf(X, Y, C_true, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("True Reference")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 1]
    err_teacher = np.abs(C_fno - C_teacher)
    c = ax.contourf(X, Y, err_teacher, levels=20, cmap='coolwarm')
    ax.set_title(f"|FNO - Teacher| (RelL2={rel_l2_teacher*100:.3f}%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 2]
    err_true = np.abs(C_fno - C_true)
    c = ax.contourf(X, Y, err_true, levels=20, cmap='coolwarm')
    ax.set_title(f"|FNO - True| (RelL2={rel_l2_true*100:.3f}%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved worst-outlier figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze FNO worst outliers for phi -> C.')
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--teacher_file', type=str, default='C_pred_10000_101_201.npy')
    parser.add_argument('--true_file', type=str, default='C_true_10000_101_201.npy')
    parser.add_argument('--model', type=str, default='Student_FNO_phi_2_C_1.pth')
    parser.add_argument('--norm_file', type=str, default='phi_normalization.npz')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--modes1', type=int, default=16)
    parser.add_argument('--modes2', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    norm = np.load(args.norm_file)
    phi_mean = norm['mean']
    phi_std = norm['std']
    print(f"Loaded phi normalization: mean={phi_mean:.6e}, std={phi_std:.6e}")

    phi, C_teacher, C_true = load_data(args.phi_file, args.teacher_file, args.true_file)
    phi_val, C_teacher_val, C_true_val, val_perm = split_train_val(phi, C_teacher, C_true, seed=args.seed)
    print(f"Validation: {phi_val.shape[0]} instances")

    model = FNO2d(width=args.width, modes1=args.modes1, modes2=args.modes2,
                  n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")

    print("Running FNO inference...")
    C_fno_val = predict_fno(model, phi_val, phi_mean, phi_std, device, batch_size=args.batch_size)

    worst_teacher, worst_true, worst_teacher_true = print_outlier_table(
        phi_val, C_fno_val, C_teacher_val, C_true_val, val_perm, top_k=args.top_k
    )

    # Plot the single worst case by Teacher-vs-True error
    worst_idx = worst_teacher_true[0]
    orig_idx = val_perm[worst_idx]
    ny, nx = phi_val.shape[1], phi_val.shape[2]
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    plot_worst(X, Y, phi_val[worst_idx], C_fno_val[worst_idx],
               C_teacher_val[worst_idx], C_true_val[worst_idx],
               worst_idx, orig_idx, "FNO_phi2C_worst_outlier.png")

    # Also plot the worst case by FNO-vs-Teacher error
    worst_fno_teacher_idx = worst_teacher[0]
    orig_idx2 = val_perm[worst_fno_teacher_idx]
    plot_worst(X, Y, phi_val[worst_fno_teacher_idx], C_fno_val[worst_fno_teacher_idx],
               C_teacher_val[worst_fno_teacher_idx], C_true_val[worst_fno_teacher_idx],
               worst_fno_teacher_idx, orig_idx2, "FNO_phi2C_worst_vs_teacher.png")

    print("Done.")


if __name__ == "__main__":
    main()
