"""
Analyze the worst-outlier validation samples for the FNO student (T, C) -> phi.

Prints the indices with largest errors and compares:
  - FNO vs Teacher error
  - FNO vs True Reference error
  - Teacher vs True Reference error
  - Input T/C statistics of the worst cases

Also plots the single worst outlier.

Usage:
    python analyze_FNO_outliers.py
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
    T = np.load(T_file)                 # (N, ny, nx)
    C = np.load(C_file)                 # (N, ny, nx)
    phi_teacher = np.load(teacher_file) # (N, ny, nx) or (N, nx, ny)
    phi_teacher = _match_spatial_shape(phi_teacher, T.shape)
    phi_true = np.load(true_file)       # (N, ny, nx) or (N, nx, ny)
    phi_true = _match_spatial_shape(phi_true, T.shape)
    return T, C, phi_teacher, phi_true


def split_train_val(T, C, phi_teacher, phi_true, train_ratio=0.9, seed=42):
    N = T.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(N * train_ratio)
    return (T[perm[n_train:]], C[perm[n_train:]],
            phi_teacher[perm[n_train:]], phi_true[perm[n_train:]],
            perm[n_train:])


def relative_l2(pred, true, axis=None):
    if axis is None:
        return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)
    return np.linalg.norm(pred - true, axis=axis) / (np.linalg.norm(true, axis=axis) + 1e-12)


def predict_fno(model, T_val, C_val, T_mean, T_std, C_mean, C_std, device, batch_size=32):
    """Run FNO inference and return original-scale phi."""
    T_norm = (T_val - T_mean) / (T_std + 1e-12)
    C_norm = (C_val - C_mean) / (C_std + 1e-12)
    inputs = np.stack([T_norm, C_norm], axis=1)  # (N, 2, ny, nx)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, T_val.shape[0], batch_size):
            end = min(start + batch_size, T_val.shape[0])
            pred = model(inputs_tensor[start:end])  # (B, 1, ny, nx)
            all_preds.append((pred.squeeze(1) * PHI_SCALE).cpu().numpy())  # original scale
    return np.concatenate(all_preds, axis=0)


def input_statistics(T_i, C_i):
    """Return key statistics for a single input T/C field."""
    return {
        'T_min': T_i.min(),
        'T_max': T_i.max(),
        'T_range': T_i.max() - T_i.min(),
        'C_min': C_i.min(),
        'C_max': C_i.max(),
        'C_range': C_i.max() - C_i.min(),
    }


def print_outlier_table(T_val, C_val, phi_fno_val, phi_teacher_val, phi_true_val,
                        val_perm, top_k=10):
    rel_l2_teacher = relative_l2(
        phi_fno_val.reshape(phi_fno_val.shape[0], -1),
        phi_teacher_val.reshape(phi_teacher_val.shape[0], -1),
        axis=1
    )
    rel_l2_true = relative_l2(
        phi_fno_val.reshape(phi_fno_val.shape[0], -1),
        phi_true_val.reshape(phi_true_val.shape[0], -1),
        axis=1
    )
    rel_l2_teacher_true = relative_l2(
        phi_teacher_val.reshape(phi_teacher_val.shape[0], -1),
        phi_true_val.reshape(phi_true_val.shape[0], -1),
        axis=1
    )

    worst_teacher_idx = np.argsort(rel_l2_teacher)[-top_k:][::-1]
    worst_true_idx = np.argsort(rel_l2_true)[-top_k:][::-1]
    worst_teacher_true_idx = np.argsort(rel_l2_teacher_true)[-top_k:][::-1]

    header = (
        f"{'val_idx':>8} {'orig_idx':>10} {'FNO_vs_Teacher':>16} "
        f"{'FNO_vs_True':>14} {'Teacher_vs_True':>18} "
        f"{'T_range':>12} {'C_max':>12}"
    )

    print("\n" + "="*100)
    print(f"Top {top_k} worst cases: FNO vs Teacher")
    print("="*100)
    print(header)
    for idx in worst_teacher_idx:
        orig_idx = val_perm[idx]
        stats = input_statistics(T_val[idx], C_val[idx])
        print(
            f"{idx:8d} {orig_idx:10d} {rel_l2_teacher[idx]*100:15.4f}% "
            f"{rel_l2_true[idx]*100:13.4f}% {rel_l2_teacher_true[idx]*100:17.4f}% "
            f"{stats['T_range']:12.4e} {stats['C_max']:12.4e}"
        )

    print("\n" + "="*100)
    print(f"Top {top_k} worst cases: FNO vs True Reference")
    print("="*100)
    print(header)
    for idx in worst_true_idx:
        orig_idx = val_perm[idx]
        stats = input_statistics(T_val[idx], C_val[idx])
        print(
            f"{idx:8d} {orig_idx:10d} {rel_l2_teacher[idx]*100:15.4f}% "
            f"{rel_l2_true[idx]*100:13.4f}% {rel_l2_teacher_true[idx]*100:17.4f}% "
            f"{stats['T_range']:12.4e} {stats['C_max']:12.4e}"
        )

    print("\n" + "="*100)
    print(f"Top {top_k} worst cases: Teacher vs True Reference")
    print("="*100)
    print(header)
    for idx in worst_teacher_true_idx:
        orig_idx = val_perm[idx]
        stats = input_statistics(T_val[idx], C_val[idx])
        print(
            f"{idx:8d} {orig_idx:10d} {rel_l2_teacher[idx]*100:15.4f}% "
            f"{rel_l2_true[idx]*100:13.4f}% {rel_l2_teacher_true[idx]*100:17.4f}% "
            f"{stats['T_range']:12.4e} {stats['C_max']:12.4e}"
        )

    return worst_teacher_idx, worst_true_idx, worst_teacher_true_idx


def plot_worst(X, Y, T_i, C_i, phi_fno, phi_teacher, phi_true, idx, orig_idx, save_path):
    rel_l2_teacher = relative_l2(phi_fno, phi_teacher)
    rel_l2_true = relative_l2(phi_fno, phi_true)
    rel_l2_teacher_true = relative_l2(phi_teacher, phi_true)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    c = ax.contourf(X, Y, T_i, levels=20, cmap='viridis')
    ax.set_title(f"Instance {orig_idx} (val idx {idx}): T")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[0, 1]
    c = ax.contourf(X, Y, C_i, levels=20, cmap='plasma')
    ax.set_title(f"Instance {orig_idx} (val idx {idx}): C")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    vmin = min(phi_fno.min(), phi_teacher.min(), phi_true.min())
    vmax = max(phi_fno.max(), phi_teacher.max(), phi_true.max())

    ax = axes[0, 2]
    c = ax.contourf(X, Y, phi_fno, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("FNO Prediction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 0]
    c = ax.contourf(X, Y, phi_teacher, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title(f"Teacher (vs True {rel_l2_teacher_true*100:.3f}%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 1]
    c = ax.contourf(X, Y, phi_true, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("True Reference")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 2]
    err_true = np.abs(phi_fno - phi_true)
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
    parser = argparse.ArgumentParser(description='Analyze FNO worst outliers for (T, C) -> phi.')
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
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    norm = np.load(args.norm_file)
    T_mean = norm['T_mean']
    T_std = norm['T_std']
    C_mean = norm['C_mean']
    C_std = norm['C_std']
    phi_scale = norm['phi_scale']
    print(f"Loaded normalization: T_mean={T_mean:.6f}, T_std={T_std:.6f}, "
          f"C_mean={C_mean:.6f}, C_std={C_std:.6f}, phi_scale={phi_scale:.6e}")

    T, C, phi_teacher, phi_true = load_data(
        args.T_file, args.C_file, args.teacher_file, args.true_file
    )
    T_val, C_val, phi_teacher_val, phi_true_val, val_perm = split_train_val(
        T, C, phi_teacher, phi_true, seed=args.seed
    )
    print(f"Validation: {T_val.shape[0]} instances")

    model = FNO2d(width=args.width, modes1=args.modes1, modes2=args.modes2,
                  n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")

    print("Running FNO inference...")
    phi_fno_val = predict_fno(
        model, T_val, C_val, T_mean, T_std, C_mean, C_std, device, batch_size=args.batch_size
    )

    worst_teacher, worst_true, worst_teacher_true = print_outlier_table(
        T_val, C_val, phi_fno_val, phi_teacher_val, phi_true_val,
        val_perm, top_k=args.top_k
    )

    # Plot the single worst case by Teacher-vs-True error
    worst_idx = worst_teacher_true[0]
    orig_idx = val_perm[worst_idx]
    ny, nx = T_val.shape[1], T_val.shape[2]
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    plot_worst(
        X, Y, T_val[worst_idx], C_val[worst_idx],
        phi_fno_val[worst_idx], phi_teacher_val[worst_idx], phi_true_val[worst_idx],
        worst_idx, orig_idx, "FNO_T_C_2_phi_worst_teacher_true.png"
    )

    # Also plot the worst case by FNO-vs-Teacher error
    worst_fno_teacher_idx = worst_teacher[0]
    orig_idx2 = val_perm[worst_fno_teacher_idx]
    plot_worst(
        X, Y, T_val[worst_fno_teacher_idx], C_val[worst_fno_teacher_idx],
        phi_fno_val[worst_fno_teacher_idx], phi_teacher_val[worst_fno_teacher_idx],
        phi_true_val[worst_fno_teacher_idx], worst_fno_teacher_idx, orig_idx2,
        "FNO_T_C_2_phi_worst_vs_teacher.png"
    )

    print("Done.")


# "C:\python\Program Files\Python310\python.exe" "C:\python\PythonFiles\CNRS Benchmark\2D_PCMON\Student_T_C_2_phi\analyze_FNO_outliers.py"
# CUDA available: True
# Using device: cuda
# Loaded normalization: T_mean=789.214798, T_std=162.382917, C_mean=198724.758330, C_std=145147.574705, phi_scale=2.500000e+07
# Validation: 1000 instances
# Loaded model from Student_FNO_T_C_2_phi_1.pth
# Running FNO inference...
#
# ====================================================================================================
# Top 10 worst cases: FNO vs Teacher
# ====================================================================================================
#  val_idx   orig_idx   FNO_vs_Teacher    FNO_vs_True    Teacher_vs_True      T_range        C_max
#      634        314          0.5745%        0.0731%            0.5741%   6.8204e+02   4.8131e+05
#      101       3101          0.5487%        0.5638%            0.0394%   1.3563e+03   5.0530e+05
#      542       6190          0.5405%        0.5452%            0.0442%   1.1046e+03   3.7189e+05
#      705       8606          0.4982%        0.5699%            0.0906%   1.0740e+03   4.3835e+05
#      644       9569          0.4927%        0.5056%            0.1122%   1.1646e+03   3.8726e+05
#      124       9042          0.4512%        0.0881%            0.5027%   4.4078e+02   3.5140e+05
#      109       8198          0.4495%        0.1238%            0.4848%   4.4928e+02   5.8278e+05
#      330       5411          0.4252%        0.4182%            0.0311%   1.1809e+03   3.5107e+05
#      484       7338          0.4017%        0.0947%            0.4535%   3.2320e+02   3.2213e+05
#      314        958          0.3498%        0.3740%            0.0486%   9.8407e+02   4.7516e+05
#
# ====================================================================================================
# Top 10 worst cases: FNO vs True Reference
# ====================================================================================================
#  val_idx   orig_idx   FNO_vs_Teacher    FNO_vs_True    Teacher_vs_True      T_range        C_max
#      705       8606          0.4982%        0.5699%            0.0906%   1.0740e+03   4.3835e+05
#      101       3101          0.5487%        0.5638%            0.0394%   1.3563e+03   5.0530e+05
#      542       6190          0.5405%        0.5452%            0.0442%   1.1046e+03   3.7189e+05
#      644       9569          0.4927%        0.5056%            0.1122%   1.1646e+03   3.8726e+05
#      330       5411          0.4252%        0.4182%            0.0311%   1.1809e+03   3.5107e+05
#      314        958          0.3498%        0.3740%            0.0486%   9.8407e+02   4.7516e+05
#      753       8002          0.3097%        0.3621%            0.0807%   2.4515e+02   6.6201e+05
#       88       2116          0.3055%        0.3285%            0.0378%   8.7729e+02   4.9540e+05
#      909       7958          0.2310%        0.2495%            0.0379%   8.4148e+02   3.5602e+05
#      812       7703          0.2281%        0.2243%            0.0326%   9.2042e+02   3.5489e+05
#
# ====================================================================================================
# Top 10 worst cases: Teacher vs True Reference
# ====================================================================================================
#  val_idx   orig_idx   FNO_vs_Teacher    FNO_vs_True    Teacher_vs_True      T_range        C_max
#      634        314          0.5745%        0.0731%            0.5741%   6.8204e+02   4.8131e+05
#      124       9042          0.4512%        0.0881%            0.5027%   4.4078e+02   3.5140e+05
#      109       8198          0.4495%        0.1238%            0.4848%   4.4928e+02   5.8278e+05
#      484       7338          0.4017%        0.0947%            0.4535%   3.2320e+02   3.2213e+05
#      501       9432          0.2869%        0.1146%            0.3716%   2.5633e+02   5.7452e+05
#      766       6813          0.3128%        0.0985%            0.3604%   2.9951e+02   3.6687e+05
#      144       9178          0.2644%        0.1028%            0.3357%   4.4480e+02   3.7984e+05
#      838       2440          0.2759%        0.0676%            0.2949%   5.9015e+02   5.5232e+05
#      475       6640          0.2633%        0.0822%            0.2897%   2.1238e+02   3.1143e+05
#      817       2894          0.2680%        0.1298%            0.2853%   4.4431e+02   4.3014e+05
# Saved worst-outlier figure to FNO_T_C_2_phi_worst_teacher_true.png
# Saved worst-outlier figure to FNO_T_C_2_phi_worst_vs_teacher.png
# Done.
#
# Process finished with exit code 0


if __name__ == "__main__":
    main()
