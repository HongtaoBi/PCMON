"""
Analyze the worst-outlier validation samples for the FNO student.

Prints the indices with largest errors and compares:
  - FNO vs Teacher error
  - Teacher vs True Reference error
  - phi statistics of the worst cases

Also plots the single worst outlier.

Usage:
    python analyze_FNO_outliers.py
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from Student_FNO_phi_2_T import FNO2d, L, H


def load_data(phi_file, teacher_file, true_file):
    phi = np.load(phi_file).transpose(0, 2, 1)       # (N, ny, nx)
    T_teacher = np.load(teacher_file)                 # (N, ny, nx)
    T_true = np.load(true_file)                       # (N, ny, nx)
    return phi, T_teacher, T_true


def split_train_val(phi, T_teacher, T_true, train_ratio=0.9, seed=42):
    N = phi.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(N * train_ratio)
    return phi[perm[n_train:]], T_teacher[perm[n_train:]], T_true[perm[n_train:]], perm[n_train:]


def relative_l2(pred, true, axis=None):
    if axis is None:
        return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)
    return np.linalg.norm(pred - true, axis=axis) / (np.linalg.norm(true, axis=axis) + 1e-12)


def predict_fno(model, phi_val, phi_mean, phi_std, device, batch_size=32):
    phi_norm = (phi_val - phi_mean) / (phi_std + 1e-12)
    phi_tensor = torch.tensor(phi_norm, dtype=torch.float32, device=device).unsqueeze(1)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, phi_val.shape[0], batch_size):
            end = min(start + batch_size, phi_val.shape[0])
            pred = model(phi_tensor[start:end])
            all_preds.append(pred.squeeze(1).cpu().numpy())
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


def print_outlier_table(phi_val, T_fno_val, T_teacher_val, T_true_val, val_perm, top_k=10):
    rel_l2_teacher = relative_l2(T_fno_val.reshape(T_fno_val.shape[0], -1),
                                  T_teacher_val.reshape(T_teacher_val.shape[0], -1),
                                  axis=1)
    rel_l2_true = relative_l2(T_fno_val.reshape(T_fno_val.shape[0], -1),
                               T_true_val.reshape(T_true_val.shape[0], -1),
                               axis=1)
    rel_l2_teacher_true = relative_l2(T_teacher_val.reshape(T_teacher_val.shape[0], -1),
                                       T_true_val.reshape(T_true_val.shape[0], -1),
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


def plot_worst(X, Y, phi_i, T_fno, T_teacher, T_true, idx, orig_idx, save_path):
    rel_l2_teacher = relative_l2(T_fno, T_teacher)
    rel_l2_true = relative_l2(T_fno, T_true)
    rel_l2_teacher_true = relative_l2(T_teacher, T_true)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    c = ax.contourf(X, Y, phi_i, levels=20, cmap='viridis')
    ax.set_title(f"Instance {orig_idx} (val idx {idx}): phi")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    vmin = min(T_fno.min(), T_teacher.min(), T_true.min())
    vmax = max(T_fno.max(), T_teacher.max(), T_true.max())

    ax = axes[0, 1]
    c = ax.contourf(X, Y, T_fno, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("FNO Prediction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[0, 2]
    c = ax.contourf(X, Y, T_teacher, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title(f"Teacher (vs True {rel_l2_teacher_true*100:.3f}%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 0]
    c = ax.contourf(X, Y, T_true, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("True Reference")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 1]
    err_teacher = np.abs(T_fno - T_teacher)
    c = ax.contourf(X, Y, err_teacher, levels=20, cmap='coolwarm')
    ax.set_title(f"|FNO - Teacher| (RelL2={rel_l2_teacher*100:.3f}%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 2]
    err_true = np.abs(T_fno - T_true)
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
    parser = argparse.ArgumentParser(description='Analyze FNO worst outliers.')
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--teacher_file', type=str, default='T_pred_10000_101_201.npy')
    parser.add_argument('--true_file', type=str, default='T_true_10000_101_201.npy')
    parser.add_argument('--model', type=str, default='Student_FNO_phi_2_T_1.pth')
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

    phi, T_teacher, T_true = load_data(args.phi_file, args.teacher_file, args.true_file)
    phi_val, T_teacher_val, T_true_val, val_perm = split_train_val(phi, T_teacher, T_true, seed=args.seed)
    print(f"Validation: {phi_val.shape[0]} instances")

    model = FNO2d(width=args.width, modes1=args.modes1, modes2=args.modes2,
                  n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")

    print("Running FNO inference...")
    T_fno_val = predict_fno(model, phi_val, phi_mean, phi_std, device, batch_size=args.batch_size)

    worst_teacher, worst_true, worst_teacher_true = print_outlier_table(
        phi_val, T_fno_val, T_teacher_val, T_true_val, val_perm, top_k=args.top_k
    )

    # Plot the single worst case by Teacher-vs-True error (the 4% outlier)
    worst_idx = worst_teacher_true[0]
    orig_idx = val_perm[worst_idx]
    ny, nx = phi_val.shape[1], phi_val.shape[2]
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    plot_worst(X, Y, phi_val[worst_idx], T_fno_val[worst_idx],
               T_teacher_val[worst_idx], T_true_val[worst_idx],
               worst_idx, orig_idx, "FNO_worst_outlier.png")

    # Also plot the worst case by FNO-vs-Teacher error
    worst_fno_teacher_idx = worst_teacher[0]
    orig_idx2 = val_perm[worst_fno_teacher_idx]
    plot_worst(X, Y, phi_val[worst_fno_teacher_idx], T_fno_val[worst_fno_teacher_idx],
               T_teacher_val[worst_fno_teacher_idx], T_true_val[worst_fno_teacher_idx],
               worst_fno_teacher_idx, orig_idx2, "FNO_worst_vs_teacher.png")

    print("Done.")

# "C:\python\Program Files\Python310\python.exe" "C:\python\PythonFiles\CNRS Benchmark\2D_PCMON\Student_phi_2_T\analyze_FNO_outliers.py"
# CUDA available: True
# Using device: cuda
# Loaded phi normalization: mean=5.227624e+06, std=5.561947e+06
# Validation: 1000 instances
# Loaded model from Student_FNO_phi_2_T_1.pth
# Running FNO inference...
#
# ================================================================================
# Top 10 worst cases: FNO vs Teacher
# ================================================================================
#  val_idx   orig_idx   FNO_vs_Teacher    FNO_vs_True    Teacher_vs_True      phi_max      phi_range
#      779       6502          6.1404%        0.8931%            6.4969%   4.3561e+07     4.3561e+07
#      156       6657          2.4848%        0.0975%            2.4446%   1.3240e+07     1.3240e+07
#      897       6183          1.7109%        0.1206%            1.7210%   1.9440e+07     1.9440e+07
#      912       9386          1.6580%        0.2556%            1.7092%   1.7370e+07     1.7370e+07
#      374       6021          1.6506%        0.1298%            1.6296%   2.1640e+07     2.1640e+07
#      980       6193          1.3270%        0.1073%            1.3052%   1.5070e+07     1.5070e+07
#       43       6277          1.2874%        0.1520%            1.3525%   2.2125e+07     2.2125e+07
#      422       9564          1.2240%        0.2167%            1.2123%   4.0914e+07     4.0914e+07
#      594       6395          1.1022%        0.1643%            1.0922%   2.1810e+07     2.1810e+07
#      793        838          1.0851%        0.8379%            0.4829%   3.9429e+07     3.9429e+07
#
# ================================================================================
# Top 10 worst cases: FNO vs True Reference
# ================================================================================
#  val_idx   orig_idx   FNO_vs_Teacher    FNO_vs_True    Teacher_vs_True      phi_max      phi_range
#      779       6502          6.1404%        0.8931%            6.4969%   4.3561e+07     4.3561e+07
#      760       4185          0.8441%        0.8437%            0.2121%   3.0048e+07     3.0048e+07
#      793        838          1.0851%        0.8379%            0.4829%   3.9429e+07     3.9429e+07
#      708        485          0.5564%        0.5783%            0.1535%   1.7876e+07     1.7876e+07
#      963       1815          0.4104%        0.4483%            0.1782%   4.8846e+07     4.8846e+07
#      918       2035          0.4193%        0.4453%            0.2069%   3.5505e+07     3.5505e+07
#      611       8352          0.5273%        0.4277%            0.2224%   3.7765e+07     3.7765e+07
#       64       4925          0.4258%        0.4231%            0.1864%   2.5978e+07     2.5978e+07
#      445       9459          0.5969%        0.4008%            0.4977%   4.1731e+07     4.1731e+07
#      405       7927          0.6394%        0.3845%            0.7506%   2.5375e+07     2.5375e+07
#
# ================================================================================
# Top 10 worst cases: Teacher vs True Reference
# ================================================================================
#  val_idx   orig_idx   FNO_vs_Teacher    FNO_vs_True    Teacher_vs_True      phi_max      phi_range
#      779       6502          6.1404%        0.8931%            6.4969%   4.3561e+07     4.3561e+07
#      156       6657          2.4848%        0.0975%            2.4446%   1.3240e+07     1.3240e+07
#      897       6183          1.7109%        0.1206%            1.7210%   1.9440e+07     1.9440e+07
#      912       9386          1.6580%        0.2556%            1.7092%   1.7370e+07     1.7370e+07
#      374       6021          1.6506%        0.1298%            1.6296%   2.1640e+07     2.1640e+07
#       43       6277          1.2874%        0.1520%            1.3525%   2.2125e+07     2.2125e+07
#      980       6193          1.3270%        0.1073%            1.3052%   1.5070e+07     1.5070e+07
#      422       9564          1.2240%        0.2167%            1.2123%   4.0914e+07     4.0914e+07
#      594       6395          1.1022%        0.1643%            1.0922%   2.1810e+07     2.1810e+07
#      344       9328          0.9893%        0.1255%            0.9873%   2.3170e+07     2.3170e+07
# Saved worst-outlier figure to FNO_worst_outlier.png
# Saved worst-outlier figure to FNO_worst_vs_teacher.png
# Done.


if __name__ == "__main__":
    main()
