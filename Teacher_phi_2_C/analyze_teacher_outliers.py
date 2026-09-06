"""
Analyze worst-outlier samples from the trained Teacher phi -> C predictions.

Prints per-chunk top-k worst cases and compares:
  - Teacher prediction vs True Reference error for C
  - Input phi statistics of the worst cases

Also plots the single worst outlier per specified chunk.

Usage:
    python analyze_teacher_outliers.py --chunk_id 6 --top_k 10
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


L = 2.0
H = 1.0


def relative_l2(pred, true, axis=None):
    if axis is None:
        return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)
    return np.linalg.norm(pred - true, axis=axis) / (np.linalg.norm(true, axis=axis) + 1e-12)


def phi_statistics(phi_i):
    return {
        'min': phi_i.min(),
        'max': phi_i.max(),
        'mean': phi_i.mean(),
        'std': phi_i.std(),
        'range': phi_i.max() - phi_i.min(),
        'peak_to_mean': phi_i.max() / (phi_i.mean() + 1e-12),
    }


def print_chunk_outlier_table(phi_chunk, C_pred_chunk, C_true_chunk,
                              chunk_id, start_idx, top_k=10):
    N = C_pred_chunk.shape[0]
    rel_l2 = relative_l2(
        C_pred_chunk.reshape(N, -1),
        C_true_chunk.reshape(N, -1),
        axis=1
    )

    worst_idx = np.argsort(rel_l2)[-top_k:][::-1]

    print("\n" + "=" * 110)
    print(f"Chunk {chunk_id}: top {top_k} worst Teacher-vs-True cases "
          f"(global indices [{start_idx}, {start_idx + N}))")
    print("=" * 110)
    print(f"{'global_idx':>12} {'rel_l2_%':>12} "
          f"{'C_pred_min':>14} {'C_pred_max':>14} {'C_true_min':>14} {'C_true_max':>14} "
          f"{'phi_min':>14} {'phi_max':>14} {'phi_mean':>14} {'peak/mean':>12}")

    for idx in worst_idx:
        global_idx = start_idx + idx
        pstats = phi_statistics(phi_chunk[idx])
        print(f"{global_idx:12d} {rel_l2[idx]*100:11.4f}% "
              f"{C_pred_chunk[idx].min():14.4e} {C_pred_chunk[idx].max():14.4e} "
              f"{C_true_chunk[idx].min():14.4e} {C_true_chunk[idx].max():14.4e} "
              f"{pstats['min']:14.4e} {pstats['max']:14.4e} {pstats['mean']:14.4e} {pstats['peak_to_mean']:12.2f}")

    return worst_idx, rel_l2


def plot_worst(X, Y, phi_i, C_pred, C_true, global_idx, save_path):
    rel_l2 = relative_l2(C_pred, C_true)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    c = ax.contourf(X, Y, phi_i, levels=20, cmap='viridis')
    ax.set_title(f"Instance {global_idx}: phi")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    vmin = min(C_pred.min(), C_true.min())
    vmax = max(C_pred.max(), C_true.max())

    ax = axes[0, 1]
    c = ax.contourf(X, Y, C_pred, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("Teacher Prediction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[0, 2]
    c = ax.contourf(X, Y, C_true, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("True Reference")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 0]
    err = np.abs(C_pred - C_true)
    c = ax.contourf(X, Y, err, levels=20, cmap='coolwarm')
    ax.set_title(f"|Teacher - True| (RelL2={rel_l2*100:.3f}%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 1]
    rel_err = np.abs(C_pred - C_true) / (np.abs(C_true) + 1e-12)
    c = ax.contourf(X, Y, rel_err * 100, levels=20, cmap='coolwarm')
    ax.set_title("Relative Error (%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 2]
    mid_j = C_true.shape[0] // 2
    ax.plot(X[mid_j, :], C_true[mid_j, :], 'b-', label='True', linewidth=2)
    ax.plot(X[mid_j, :], C_pred[mid_j, :], 'r--', label='Teacher', linewidth=2)
    ax.set_title(f"Midline y={Y[mid_j, 0]:.2f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("C")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Teacher worst outliers for phi -> C.')
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--C_pred_file', type=str, default='C_pred_10000_101_201.npy')
    parser.add_argument('--C_true_file', type=str, default='C_true_10000_101_201.npy')
    parser.add_argument('--chunk_id', type=int, default=None,
                        help='Chunk to analyze (0-9). If None, analyze all chunks.')
    parser.add_argument('--instances_per_chunk', type=int, default=1000)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    phi = np.load(args.phi_file)           # (N, nx, ny) or (N, ny, nx)
    C_pred = np.load(args.C_pred_file)     # (N, ny, nx)
    C_true = np.load(args.C_true_file)     # (N, ny, nx)

    # Ensure phi matches C spatial shape (ny, nx)
    if phi.shape[1:] != C_pred.shape[1:]:
        phi = phi.transpose(0, 2, 1)

    assert phi.shape == C_pred.shape == C_true.shape, \
        f"phi {phi.shape}, C_pred {C_pred.shape}, C_true {C_true.shape} must match"

    N = phi.shape[0]
    n_chunks = (N + args.instances_per_chunk - 1) // args.instances_per_chunk

    ny, nx = phi.shape[1], phi.shape[2]
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    chunks_to_analyze = [args.chunk_id] if args.chunk_id is not None else list(range(n_chunks))

    for chunk_id in chunks_to_analyze:
        start = chunk_id * args.instances_per_chunk
        end = min(start + args.instances_per_chunk, N)

        phi_chunk = phi[start:end]
        C_pred_chunk = C_pred[start:end]
        C_true_chunk = C_true[start:end]

        worst_idx, rel_l2 = print_chunk_outlier_table(
            phi_chunk, C_pred_chunk, C_true_chunk,
            chunk_id, start, top_k=args.top_k
        )

        # Plot the single worst case for this chunk
        worst_local = worst_idx[0]
        global_idx = start + worst_local
        save_path = f"teacher_phi2C_outlier_chunk_{chunk_id}_inst_{global_idx}.png"
        plot_worst(X, Y,
                   phi_chunk[worst_local],
                   C_pred_chunk[worst_local], C_true_chunk[worst_local],
                   global_idx, save_path)

# "C:\python\Program Files\Python310\python.exe" "C:\python\PythonFiles\CNRS Benchmark\2D_PCMON\Teacher_phi_2_C\analyze_teacher_outliers.py"
#
# ==============================================================================================================
# Chunk 0: top 10 worst Teacher-vs-True cases (global indices [0, 1000))
# ==============================================================================================================
#   global_idx     rel_l2_%     C_pred_min     C_pred_max     C_true_min     C_true_max        phi_min        phi_max       phi_mean    peak/mean
#           11      3.5024%     0.0000e+00     8.7578e+05     0.0000e+00     7.6005e+05     0.0000e+00     2.8425e+07     7.4450e+06         3.82
#          323      2.9949%     0.0000e+00     6.4639e+05     0.0000e+00     6.3842e+05     0.0000e+00     2.5446e+07     5.7098e+06         4.46
#          910      2.7033%     0.0000e+00     8.5079e+05     0.0000e+00     7.9174e+05     0.0000e+00     2.8447e+07     7.6746e+06         3.71
#          327      2.5460%     0.0000e+00     5.4976e+05     0.0000e+00     5.4651e+05     0.0000e+00     3.4978e+07     5.0063e+06         6.99
#          231      2.5310%     0.0000e+00     7.3724e+05     0.0000e+00     7.1972e+05     0.0000e+00     2.6173e+07     6.1817e+06         4.23
#          556      2.4714%     0.0000e+00     5.8161e+05     0.0000e+00     5.6170e+05     0.0000e+00     3.0165e+07     5.0199e+06         6.01
#          662      2.4048%     0.0000e+00     7.8858e+05     0.0000e+00     7.8567e+05     0.0000e+00     3.3906e+07     7.3435e+06         4.62
#           64      2.3856%     0.0000e+00     6.9420e+05     0.0000e+00     6.4769e+05     0.0000e+00     2.4906e+07     6.1565e+06         4.05
#          826      2.3619%     0.0000e+00     3.4103e+05     0.0000e+00     3.4276e+05     0.0000e+00     1.5699e+07     3.6551e+06         4.30
#          541      2.3563%     0.0000e+00     6.4144e+05     0.0000e+00     6.3988e+05     0.0000e+00     3.0897e+07     6.3150e+06         4.89
# Saved figure to teacher_phi2C_outlier_chunk_0_inst_11.png
#
# ==============================================================================================================
# Chunk 1: top 10 worst Teacher-vs-True cases (global indices [1000, 2000))
# ==============================================================================================================
#   global_idx     rel_l2_%     C_pred_min     C_pred_max     C_true_min     C_true_max        phi_min        phi_max       phi_mean    peak/mean
#         1527      3.7342%     0.0000e+00     4.4603e+05     0.0000e+00     4.3317e+05     0.0000e+00     1.4614e+07     3.5086e+06         4.17
#         1413      2.8843%     0.0000e+00     6.0938e+05     0.0000e+00     6.0606e+05     0.0000e+00     2.0305e+07     6.8145e+06         2.98
#         1343      2.7799%     0.0000e+00     4.7290e+05     0.0000e+00     4.7077e+05     0.0000e+00     1.8892e+07     4.6111e+06         4.10
#         1843      2.7087%     0.0000e+00     8.6379e+05     0.0000e+00     8.2725e+05     0.0000e+00     2.9000e+07     7.9833e+06         3.63
#         1081      2.6610%     0.0000e+00     8.0800e+05     0.0000e+00     8.0400e+05     0.0000e+00     2.5147e+07     8.8901e+06         2.83
#         1398      2.6589%     0.0000e+00     3.3606e+05     0.0000e+00     3.3443e+05     0.0000e+00     1.1513e+07     2.9429e+06         3.91
#         1990      2.6383%     0.0000e+00     5.5513e+05     0.0000e+00     5.5341e+05     0.0000e+00     2.0030e+07     4.6420e+06         4.31
#         1989      2.5707%     0.0000e+00     7.3019e+05     0.0000e+00     7.2787e+05     0.0000e+00     3.0387e+07     6.0622e+06         5.01
#         1265      2.5690%     0.0000e+00     4.1393e+05     0.0000e+00     4.1232e+05     0.0000e+00     1.6618e+07     4.2033e+06         3.95
#         1498      2.5502%     0.0000e+00     7.3980e+05     0.0000e+00     7.3803e+05     0.0000e+00     3.2601e+07     6.7081e+06         4.86
# Saved figure to teacher_phi2C_outlier_chunk_1_inst_1527.png
#
# ==============================================================================================================
# Chunk 2: top 10 worst Teacher-vs-True cases (global indices [2000, 3000))
# ==============================================================================================================
#   global_idx     rel_l2_%     C_pred_min     C_pred_max     C_true_min     C_true_max        phi_min        phi_max       phi_mean    peak/mean
#         2357      1.7856%     0.0000e+00     3.7655e+05     0.0000e+00     3.6842e+05     0.0000e+00     1.9656e+07     3.3024e+06         5.95
#         2946      1.6455%    -5.1776e+02     5.4963e+05     0.0000e+00     5.4916e+05     0.0000e+00     3.0793e+07     4.0244e+06         7.65
#         2007      1.3868%     0.0000e+00     9.9178e+05     0.0000e+00     9.8966e+05     0.0000e+00     3.0078e+07     8.4063e+06         3.58
#         2349      1.2874%     0.0000e+00     3.8117e+05     0.0000e+00     3.7809e+05     0.0000e+00     1.4949e+07     3.7640e+06         3.97
#         2375      1.2469%     0.0000e+00     6.2672e+05     0.0000e+00     6.0904e+05     0.0000e+00     3.1208e+07     6.2255e+06         5.01
#         2484      1.2436%     0.0000e+00     4.5969e+05     0.0000e+00     4.5852e+05     0.0000e+00     1.7199e+07     4.4289e+06         3.88
#         2984      1.1331%     0.0000e+00     6.1901e+05     0.0000e+00     6.1747e+05     0.0000e+00     2.4694e+07     5.4667e+06         4.52
#         2992      1.1186%     0.0000e+00     6.4975e+05     0.0000e+00     6.4807e+05     0.0000e+00     2.3978e+07     5.0977e+06         4.70
#         2927      1.0705%     0.0000e+00     5.1846e+05     0.0000e+00     5.1710e+05     0.0000e+00     2.1586e+07     5.4500e+06         3.96
#         2924      0.9791%     0.0000e+00     5.2823e+05     0.0000e+00     5.2765e+05     0.0000e+00     2.2419e+07     5.2719e+06         4.25
# Saved figure to teacher_phi2C_outlier_chunk_2_inst_2357.png
#
# ==============================================================================================================
# Chunk 3: top 10 worst Teacher-vs-True cases (global indices [3000, 4000))
# ==============================================================================================================
#   global_idx     rel_l2_%     C_pred_min     C_pred_max     C_true_min     C_true_max        phi_min        phi_max       phi_mean    peak/mean
#         3097      5.1813%     0.0000e+00     5.2443e+05     0.0000e+00     5.0624e+05     0.0000e+00     1.9904e+07     4.1728e+06         4.77
#         3332      3.2901%     0.0000e+00     6.5367e+05     0.0000e+00     6.5218e+05     0.0000e+00     2.7570e+07     5.6328e+06         4.89
#         3303      3.2671%     0.0000e+00     6.0153e+05     0.0000e+00     6.0714e+05     0.0000e+00     2.3874e+07     4.9835e+06         4.79
#         3452      2.9430%     0.0000e+00     7.0413e+05     0.0000e+00     7.0248e+05     0.0000e+00     2.6565e+07     5.3956e+06         4.92
#         3534      2.5446%     0.0000e+00     7.7968e+05     0.0000e+00     7.7770e+05     0.0000e+00     2.4031e+07     7.1289e+06         3.37
#         3839      2.4704%     0.0000e+00     7.9185e+05     0.0000e+00     7.8992e+05     0.0000e+00     3.0240e+07     6.8469e+06         4.42
#         3811      2.3859%     0.0000e+00     5.3905e+05     0.0000e+00     5.3121e+05     0.0000e+00     1.8269e+07     4.7504e+06         3.85
#         3212      2.1437%     0.0000e+00     9.6027e+05     0.0000e+00     9.5707e+05     0.0000e+00     3.7026e+07     1.0316e+07         3.59
#         3740      2.1202%     0.0000e+00     9.8384e+05     0.0000e+00     9.7918e+05     0.0000e+00     2.8323e+07     9.7061e+06         2.92
#         3905      2.0075%    -8.7740e+01     6.0401e+05     0.0000e+00     6.1816e+05     0.0000e+00     2.1525e+07     4.8138e+06         4.47
# Saved figure to teacher_phi2C_outlier_chunk_3_inst_3097.png
#
# ==============================================================================================================
# Chunk 4: top 10 worst Teacher-vs-True cases (global indices [4000, 5000))
# ==============================================================================================================
#   global_idx     rel_l2_%     C_pred_min     C_pred_max     C_true_min     C_true_max        phi_min        phi_max       phi_mean    peak/mean
#         4263      3.5720%     0.0000e+00     5.9216e+05     0.0000e+00     5.9062e+05     0.0000e+00     2.8301e+07     6.2735e+06         4.51
#         4941      3.2407%     0.0000e+00     7.9704e+05     0.0000e+00     7.9421e+05     0.0000e+00     2.7380e+07     7.1391e+06         3.84
#         4092      3.0779%     0.0000e+00     4.9382e+05     0.0000e+00     4.9246e+05     0.0000e+00     1.7442e+07     5.1382e+06         3.39
#         4353      2.9676%     0.0000e+00     4.5856e+05     0.0000e+00     4.5714e+05     0.0000e+00     1.6048e+07     4.5656e+06         3.52
#         4431      2.9378%     0.0000e+00     4.7287e+05     0.0000e+00     4.7132e+05     0.0000e+00     1.5015e+07     4.9861e+06         3.01
#         4897      2.8222%     0.0000e+00     6.8093e+05     0.0000e+00     6.7873e+05     0.0000e+00     2.8186e+07     5.1998e+06         5.42
#         4093      2.8176%     0.0000e+00     5.7910e+05     0.0000e+00     5.7729e+05     0.0000e+00     2.8305e+07     4.5715e+06         6.19
#         4959      2.6286%     0.0000e+00     2.7046e+05     0.0000e+00     2.6971e+05     0.0000e+00     1.4246e+07     2.4786e+06         5.75
#         4552      2.5595%     0.0000e+00     3.6646e+05     0.0000e+00     3.6536e+05     0.0000e+00     1.0698e+07     3.9277e+06         2.72
#         4098      2.5560%     0.0000e+00     4.6219e+05     0.0000e+00     4.6091e+05     0.0000e+00     1.7093e+07     4.7485e+06         3.60
# Saved figure to teacher_phi2C_outlier_chunk_4_inst_4263.png
#
# ==============================================================================================================
# Chunk 5: top 10 worst Teacher-vs-True cases (global indices [5000, 6000))
# ==============================================================================================================
#   global_idx     rel_l2_%     C_pred_min     C_pred_max     C_true_min     C_true_max        phi_min        phi_max       phi_mean    peak/mean
#         5840      2.4777%     0.0000e+00     3.1818e+05     0.0000e+00     3.1707e+05     0.0000e+00     1.1467e+07     2.7279e+06         4.20
#         5441      2.4718%     0.0000e+00     4.1662e+05     0.0000e+00     4.1502e+05     0.0000e+00     1.3820e+07     3.5029e+06         3.95
#         5388      2.4588%     0.0000e+00     6.2176e+05     0.0000e+00     6.2069e+05     0.0000e+00     5.4377e+07     5.7555e+06         9.45
#         5955      2.3830%     0.0000e+00     5.4152e+05     0.0000e+00     5.3939e+05     0.0000e+00     2.6489e+07     4.8116e+06         5.51
#         5390      2.0963%     0.0000e+00     6.6032e+05     0.0000e+00     6.5789e+05     0.0000e+00     2.5258e+07     7.1404e+06         3.54
#         5349      2.0571%     0.0000e+00     7.0417e+05     0.0000e+00     7.0306e+05     0.0000e+00     2.5043e+07     5.9745e+06         4.19
#         5638      2.0469%     0.0000e+00     7.2655e+05     0.0000e+00     7.2377e+05     0.0000e+00     2.5704e+07     7.1114e+06         3.61
#         5845      1.9808%     0.0000e+00     5.2566e+05     0.0000e+00     5.2393e+05     0.0000e+00     2.5119e+07     4.8202e+06         5.21
#         5041      1.9629%     0.0000e+00     6.0940e+05     0.0000e+00     6.0879e+05     0.0000e+00     2.0997e+07     5.1064e+06         4.11
#         5238      1.8776%     0.0000e+00     6.3746e+05     0.0000e+00     6.3513e+05     0.0000e+00     2.1080e+07     5.2680e+06         4.00
# Saved figure to teacher_phi2C_outlier_chunk_5_inst_5840.png
#
# ==============================================================================================================
# Chunk 6: top 10 worst Teacher-vs-True cases (global indices [6000, 7000))
# ==============================================================================================================
#   global_idx     rel_l2_%     C_pred_min     C_pred_max     C_true_min     C_true_max        phi_min        phi_max       phi_mean    peak/mean
#         6765      1.8699%     0.0000e+00     7.6804e+05     0.0000e+00     7.6613e+05     0.0000e+00     4.4187e+07     6.4843e+06         6.81
#         6858      1.6517%     0.0000e+00     6.5900e+05     0.0000e+00     6.5799e+05     0.0000e+00     3.4303e+07     6.0925e+06         5.63
#         6280      1.5618%     0.0000e+00     6.9536e+05     0.0000e+00     6.9378e+05     0.0000e+00     2.7038e+07     7.3511e+06         3.68
#         6284      1.5137%     0.0000e+00     5.5027e+05     0.0000e+00     4.9901e+05     0.0000e+00     2.0242e+07     5.3310e+06         3.80
#         6963      1.4930%     0.0000e+00     9.8231e+05     0.0000e+00     9.7870e+05     0.0000e+00     3.8427e+07     9.0909e+06         4.23
#         6110      1.4913%     0.0000e+00     5.0452e+05     0.0000e+00     5.0311e+05     0.0000e+00     2.4829e+07     4.6187e+06         5.38
#         6353      1.4890%     0.0000e+00     9.7037e+05     0.0000e+00     9.6816e+05     0.0000e+00     4.0042e+07     9.1572e+06         4.37
#         6637      1.4857%     0.0000e+00     6.2456e+05     0.0000e+00     6.2325e+05     0.0000e+00     3.5207e+07     6.0898e+06         5.78
#         6027      1.4854%     0.0000e+00     7.6057e+05     0.0000e+00     7.5813e+05     0.0000e+00     2.9078e+07     7.1922e+06         4.04
#         6682      1.4812%     0.0000e+00     6.7510e+05     0.0000e+00     6.7310e+05     0.0000e+00     3.7404e+07     6.4306e+06         5.82
# Saved figure to teacher_phi2C_outlier_chunk_6_inst_6765.png
#
# ==============================================================================================================
# Chunk 7: top 10 worst Teacher-vs-True cases (global indices [7000, 8000))
# ==============================================================================================================
#   global_idx     rel_l2_%     C_pred_min     C_pred_max     C_true_min     C_true_max        phi_min        phi_max       phi_mean    peak/mean
#         7279      1.7995%     0.0000e+00     7.4218e+05     0.0000e+00     7.2505e+05     0.0000e+00     3.5674e+07     7.3971e+06         4.82
#         7649      1.4322%     0.0000e+00     8.7508e+05     0.0000e+00     8.7208e+05     0.0000e+00     2.8615e+07     8.8135e+06         3.25
#         7922      1.4249%     0.0000e+00     1.0006e+06     0.0000e+00     9.9793e+05     0.0000e+00     4.4473e+07     9.3413e+06         4.76
#         7864      1.3164%     0.0000e+00     6.5236e+05     0.0000e+00     6.4951e+05     0.0000e+00     2.3499e+07     5.5591e+06         4.23
#         7559      1.3062%     0.0000e+00     7.6424e+05     0.0000e+00     7.6178e+05     0.0000e+00     2.7644e+07     7.1549e+06         3.86
#         7177      1.2561%     0.0000e+00     8.3077e+05     0.0000e+00     8.2731e+05     0.0000e+00     4.0259e+07     5.9118e+06         6.81
#         7695      1.2541%     0.0000e+00     6.8364e+05     0.0000e+00     6.8041e+05     0.0000e+00     2.3479e+07     7.4676e+06         3.14
#         7297      1.2401%     0.0000e+00     6.4121e+05     0.0000e+00     6.3982e+05     0.0000e+00     2.6498e+07     5.4158e+06         4.89
#         7698      1.2219%     0.0000e+00     3.9840e+05     0.0000e+00     3.9761e+05     0.0000e+00     1.6771e+07     3.4391e+06         4.88
#         7504      1.1699%     0.0000e+00     7.3638e+05     0.0000e+00     7.3282e+05     0.0000e+00     2.7651e+07     5.5080e+06         5.02
# Saved figure to teacher_phi2C_outlier_chunk_7_inst_7279.png
#
# ==============================================================================================================
# Chunk 8: top 10 worst Teacher-vs-True cases (global indices [8000, 9000))
# ==============================================================================================================
#   global_idx     rel_l2_%     C_pred_min     C_pred_max     C_true_min     C_true_max        phi_min        phi_max       phi_mean    peak/mean
#         8587      2.8786%     0.0000e+00     9.2267e+05     0.0000e+00     8.4679e+05     0.0000e+00     2.7088e+07     8.4241e+06         3.22
#         8392      2.8073%     0.0000e+00     7.3421e+05     0.0000e+00     6.7618e+05     0.0000e+00     3.4776e+07     6.5150e+06         5.34
#         8390      2.6446%     0.0000e+00     9.7905e+05     0.0000e+00     9.3640e+05     0.0000e+00     4.5063e+07     8.0788e+06         5.58
#         8727      2.3236%     0.0000e+00     7.9297e+05     0.0000e+00     7.8379e+05     0.0000e+00     2.8507e+07     6.8173e+06         4.18
#         8771      2.2290%     0.0000e+00     7.0609e+05     0.0000e+00     6.6232e+05     0.0000e+00     2.7974e+07     6.1946e+06         4.52
#         8606      2.0861%     0.0000e+00     7.9433e+05     0.0000e+00     7.4866e+05     0.0000e+00     2.2462e+07     8.0822e+06         2.78
#         8442      1.7411%     0.0000e+00     6.5751e+05     0.0000e+00     6.5551e+05     0.0000e+00     1.9302e+07     6.4969e+06         2.97
#         8432      1.5786%     0.0000e+00     6.4937e+05     0.0000e+00     6.4758e+05     0.0000e+00     2.1563e+07     6.0800e+06         3.55
#         8291      1.5379%     0.0000e+00     7.5960e+05     0.0000e+00     7.5217e+05     0.0000e+00     2.4631e+07     7.5050e+06         3.28
#         8559      1.5245%     0.0000e+00     4.7284e+05     0.0000e+00     4.6574e+05     0.0000e+00     1.5008e+07     4.5086e+06         3.33
# Saved figure to teacher_phi2C_outlier_chunk_8_inst_8587.png
#
# ==============================================================================================================
# Chunk 9: top 10 worst Teacher-vs-True cases (global indices [9000, 10000))
# ==============================================================================================================
#   global_idx     rel_l2_%     C_pred_min     C_pred_max     C_true_min     C_true_max        phi_min        phi_max       phi_mean    peak/mean
#         9891      1.7214%     0.0000e+00     7.4920e+05     0.0000e+00     7.6506e+05     0.0000e+00     2.6796e+07     7.7925e+06         3.44
#         9506      1.4986%     0.0000e+00     4.1207e+05     0.0000e+00     4.1540e+05     0.0000e+00     1.7404e+07     3.4191e+06         5.09
#         9902      1.3584%     0.0000e+00     3.6413e+05     0.0000e+00     3.6749e+05     0.0000e+00     1.0336e+07     2.9377e+06         3.52
#         9400      1.1838%     0.0000e+00     1.0890e+06     0.0000e+00     1.1067e+06     0.0000e+00     4.6655e+07     1.0470e+07         4.46
#         9742      1.1379%     0.0000e+00     7.1341e+05     0.0000e+00     7.1001e+05     0.0000e+00     2.5153e+07     7.6671e+06         3.28
#         9667      1.0911%     0.0000e+00     4.3704e+05     0.0000e+00     4.2989e+05     0.0000e+00     2.2625e+07     4.2850e+06         5.28
#         9087      0.9017%     0.0000e+00     8.6953e+05     0.0000e+00     8.7359e+05     0.0000e+00     3.6749e+07     8.0814e+06         4.55
#         9250      0.8281%     0.0000e+00     6.2285e+05     0.0000e+00     6.2139e+05     0.0000e+00     2.6776e+07     5.2077e+06         5.14
#         9855      0.8171%     0.0000e+00     8.0617e+05     0.0000e+00     8.0407e+05     0.0000e+00     3.6660e+07     6.2384e+06         5.88
#         9397      0.7999%     0.0000e+00     6.9046e+05     0.0000e+00     6.8802e+05     0.0000e+00     2.5973e+07     7.8112e+06         3.33
# Saved figure to teacher_phi2C_outlier_chunk_9_inst_9891.png
#
# Process finished with exit code 0



if __name__ == "__main__":
    main()
