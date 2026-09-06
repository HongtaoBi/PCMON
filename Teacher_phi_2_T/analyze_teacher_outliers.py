"""
Analyze worst-outlier samples from the trained Teacher predictions.

Prints per-chunk top-k worst cases and compares:
  - Teacher prediction vs True Reference error
  - phi statistics of the worst cases

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


def print_chunk_outlier_table(phi_chunk, T_pred_chunk, T_true_chunk,
                               chunk_id, start_idx, top_k=10):
    N = phi_chunk.shape[0]
    rel_l2 = relative_l2(
        T_pred_chunk.reshape(N, -1),
        T_true_chunk.reshape(N, -1),
        axis=1
    )

    worst_idx = np.argsort(rel_l2)[-top_k:][::-1]

    print("\n" + "=" * 110)
    print(f"Chunk {chunk_id}: top {top_k} worst Teacher-vs-True cases "
          f"(global indices [{start_idx}, {start_idx + N}))")
    print("=" * 110)
    print(f"{'global_idx':>12} {'rel_l2_%':>12} {'phi_min':>14} {'phi_max':>14} "
          f"{'phi_mean':>14} {'phi_std':>14} {'peak/mean':>12} {'T_min':>10} {'T_max':>10}")

    for idx in worst_idx:
        global_idx = start_idx + idx
        stats = phi_statistics(phi_chunk[idx])
        T_min = T_true_chunk[idx].min()
        T_max = T_true_chunk[idx].max()
        print(f"{global_idx:12d} {rel_l2[idx]*100:11.4f}% "
              f"{stats['min']:14.4e} {stats['max']:14.4e} {stats['mean']:14.4e} "
              f"{stats['std']:14.4e} {stats['peak_to_mean']:12.2f} {T_min:10.2f} {T_max:10.2f}")

    return worst_idx, rel_l2


def plot_worst(X, Y, phi_i, T_pred, T_true, global_idx, save_path):
    rel_l2 = relative_l2(T_pred, T_true)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    c = ax.contourf(X, Y, phi_i, levels=20, cmap='viridis')
    ax.set_title(f"Instance {global_idx}: phi")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    vmin = min(T_pred.min(), T_true.min())
    vmax = max(T_pred.max(), T_true.max())

    ax = axes[0, 1]
    c = ax.contourf(X, Y, T_pred, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("Teacher Prediction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[0, 2]
    c = ax.contourf(X, Y, T_true, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("True Reference")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 0]
    err = np.abs(T_pred - T_true)
    c = ax.contourf(X, Y, err, levels=20, cmap='coolwarm')
    ax.set_title(f"|Teacher - True| (RelL2={rel_l2*100:.3f}%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 1]
    rel_err = np.abs(T_pred - T_true) / (np.abs(T_true) + 1e-12)
    c = ax.contourf(X, Y, rel_err * 100, levels=20, cmap='coolwarm')
    ax.set_title("Relative Error (%)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.colorbar(c, ax=ax)

    ax = axes[1, 2]
    mid_j = T_true.shape[1] // 2
    ax.plot(X[mid_j, :], T_true[mid_j, :], 'b-', label='True', linewidth=2)
    ax.plot(X[mid_j, :], T_pred[mid_j, :], 'r--', label='Teacher', linewidth=2)
    ax.set_title(f"Midline y={Y[mid_j, 0]:.2f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("T [K]")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Teacher worst outliers.')
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--T_pred_file', type=str, default='T_pred_10000_101_201.npy')
    parser.add_argument('--T_true_file', type=str, default='T_true_10000_101_201.npy')
    parser.add_argument('--chunk_id', type=int, default=None,
                        help='Chunk to analyze (0-9). If None, analyze all chunks.')
    parser.add_argument('--instances_per_chunk', type=int, default=1000)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    phi = np.load(args.phi_file).transpose(0, 2, 1)  # (N, ny, nx)
    T_pred = np.load(args.T_pred_file)                # (N, ny, nx)
    T_true = np.load(args.T_true_file)                # (N, ny, nx)

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
        T_pred_chunk = T_pred[start:end]
        T_true_chunk = T_true[start:end]

        worst_idx, rel_l2 = print_chunk_outlier_table(
            phi_chunk, T_pred_chunk, T_true_chunk,
            chunk_id, start, top_k=args.top_k
        )

        # Plot the single worst case for this chunk
        worst_local = worst_idx[0]
        global_idx = start + worst_local
        save_path = f"teacher_outlier_chunk_{chunk_id}_inst_{global_idx}.png"
        plot_worst(X, Y, phi_chunk[worst_local],
                   T_pred_chunk[worst_local], T_true_chunk[worst_local],
                   global_idx, save_path)

# "C:\python\Program Files\Python310\python.exe" "C:\python\PythonFiles\CNRS Benchmark\2D_PCMON\Teacher_phi_2_T\analyze_teacher_outliers.py"
#
# ==============================================================================================================
# Chunk 0: top 10 worst Teacher-vs-True cases (global indices [0, 1000))
# ==============================================================================================================
#   global_idx     rel_l2_%        phi_min        phi_max       phi_mean        phi_std    peak/mean      T_min      T_max
#          359      0.9964%     0.0000e+00     3.3523e+07     8.3054e+06     7.8080e+06         4.04     600.00    1781.02
#           52      0.6517%     0.0000e+00     2.7353e+07     5.9312e+06     5.4964e+06         4.61     600.00    1423.24
#          495      0.5610%     0.0000e+00     2.8866e+07     6.5931e+06     6.2467e+06         4.38     600.00    1429.34
#          691      0.5052%     0.0000e+00     2.8960e+07     6.6882e+06     7.1527e+06         4.33     600.00    1632.01
#          838      0.4829%     0.0000e+00     3.9429e+07     1.2591e+07     9.8954e+06         3.13     600.00    2183.62
#          231      0.3815%     0.0000e+00     2.6173e+07     6.1817e+06     6.7027e+06         4.23     600.00    1510.59
#          638      0.3569%     0.0000e+00     2.7740e+07     6.3140e+06     7.0161e+06         4.39     600.00    1498.55
#          145      0.3557%     0.0000e+00     2.4096e+07     7.2088e+06     6.0887e+06         3.34     600.00    1531.37
#          779      0.3466%     0.0000e+00     2.3455e+07     6.4598e+06     5.5765e+06         3.63     600.00    1243.55
#          811      0.3302%     0.0000e+00     3.3497e+07     1.0883e+07     8.6950e+06         3.08     600.00    2003.32
# Saved figure to teacher_outlier_chunk_0_inst_359.png
#
# ==============================================================================================================
# Chunk 1: top 10 worst Teacher-vs-True cases (global indices [1000, 2000))
# ==============================================================================================================
#   global_idx     rel_l2_%        phi_min        phi_max       phi_mean        phi_std    peak/mean      T_min      T_max
#         1770      1.5738%     0.0000e+00     2.0561e+07     5.4300e+06     5.0331e+06         3.79     600.00    1365.88
#         1049      1.1555%     0.0000e+00     3.2072e+07     8.0197e+06     7.9544e+06         4.00     600.00    1586.62
#         1161      0.9431%     0.0000e+00     2.3485e+07     5.7030e+06     5.0508e+06         4.12     600.00    1389.45
#         1485      0.6704%     0.0000e+00     3.5640e+07     7.6334e+06     7.3367e+06         4.67     600.00    1408.76
#         1593      0.5697%     0.0000e+00     1.6749e+07     5.7249e+06     4.3021e+06         2.93     600.00    1297.03
#         1617      0.4901%     0.0000e+00     3.1856e+07     7.8942e+06     7.3939e+06         4.04     600.00    1640.72
#         1373      0.4388%     0.0000e+00     1.8266e+07     4.8211e+06     3.8509e+06         3.79     600.00    1269.42
#         1727      0.4220%     0.0000e+00     1.4265e+07     3.0824e+06     2.9818e+06         4.63     600.00    1088.40
#         1471      0.3648%     0.0000e+00     2.8089e+07     1.0029e+07     7.0921e+06         2.80     600.00    1709.29
#         1691      0.3561%     0.0000e+00     4.2004e+07     7.2961e+06     8.0268e+06         5.76     600.00    1797.85
# Saved figure to teacher_outlier_chunk_1_inst_1770.png
#
# ==============================================================================================================
# Chunk 2: top 10 worst Teacher-vs-True cases (global indices [2000, 3000))
# ==============================================================================================================
#   global_idx     rel_l2_%        phi_min        phi_max       phi_mean        phi_std    peak/mean      T_min      T_max
#         2845      4.6917%     0.0000e+00     2.7073e+07     8.4297e+06     6.4581e+06         3.21     600.00    1542.65
#         2799      4.3261%     0.0000e+00     3.3121e+07     7.0970e+06     5.8803e+06         4.67     600.00    1498.55
#         2078      4.3170%     0.0000e+00     2.6136e+07     8.1554e+06     6.3941e+06         3.20     600.00    1519.47
#         2853      3.7184%     0.0000e+00     1.7047e+07     4.7911e+06     3.8790e+06         3.56     600.00    1190.59
#         2720      1.4254%     0.0000e+00     2.4386e+07     6.8403e+06     5.9635e+06         3.57     600.00    1476.10
#         2905      0.9330%     0.0000e+00     1.8379e+07     6.2210e+06     4.4139e+06         2.95     600.00    1323.42
#         2443      0.8820%     0.0000e+00     2.4820e+07     7.1513e+06     4.9419e+06         3.47     600.00    1409.01
#         2240      0.7754%     0.0000e+00     1.8756e+07     5.5419e+06     4.8451e+06         3.38     600.00    1279.56
#         2038      0.6664%     0.0000e+00     5.2525e+07     6.2400e+06     8.4189e+06         8.42     600.00    1265.33
#         2360      0.6555%     0.0000e+00     1.5687e+07     4.0970e+06     3.0360e+06         3.83     600.00    1036.75
# Saved figure to teacher_outlier_chunk_2_inst_2845.png
#
# ==============================================================================================================
# Chunk 3: top 10 worst Teacher-vs-True cases (global indices [3000, 4000))
# ==============================================================================================================
#   global_idx     rel_l2_%        phi_min        phi_max       phi_mean        phi_std    peak/mean      T_min      T_max
#         3497      1.7376%     0.0000e+00     4.0461e+07     7.0993e+06     8.0030e+06         5.70     600.00    1488.36
#         3589      1.5964%     0.0000e+00     3.8327e+07     1.0432e+07     8.4392e+06         3.67     600.00    1907.69
#         3714      1.2143%     0.0000e+00     2.4368e+07     5.6966e+06     5.7816e+06         4.28     600.00    1207.11
#         3107      0.7026%     0.0000e+00     2.4450e+07     6.2334e+06     6.3019e+06         3.92     600.00    1317.66
#         3680      0.6771%     0.0000e+00     2.0993e+07     4.3607e+06     4.3695e+06         4.81     600.00    1081.48
#         3341      0.5732%     0.0000e+00     2.6399e+07     8.0546e+06     6.3872e+06         3.28     600.00    1599.22
#         3726      0.4885%     0.0000e+00     1.9001e+07     4.7915e+06     4.3857e+06         3.97     600.00    1075.16
#         3257      0.4665%     0.0000e+00     2.1280e+07     6.7617e+06     5.1712e+06         3.15     600.00    1354.91
#         3324      0.4625%     0.0000e+00     4.3818e+07     1.1198e+07     9.6852e+06         3.91     600.00    1770.28
#         3390      0.4260%     0.0000e+00     2.4997e+07     5.7302e+06     6.0675e+06         4.36     600.00    1188.56
# Saved figure to teacher_outlier_chunk_3_inst_3497.png
#
# ==============================================================================================================
# Chunk 4: top 10 worst Teacher-vs-True cases (global indices [4000, 5000))
# ==============================================================================================================
#   global_idx     rel_l2_%        phi_min        phi_max       phi_mean        phi_std    peak/mean      T_min      T_max
#         4696      5.1681%     0.0000e+00     2.2870e+07     6.0484e+06     5.1662e+06         3.78     600.00    1450.24
#         4353      2.8011%     0.0000e+00     1.6048e+07     4.5656e+06     4.0101e+06         3.52     600.00    1160.84
#         4819      2.5865%     0.0000e+00     2.5869e+07     6.5936e+06     5.7684e+06         3.92     600.00    1561.58
#         4587      1.9539%     0.0000e+00     3.4353e+07     6.8239e+06     6.2751e+06         5.03     600.00    1314.05
#         4193      1.4667%     0.0000e+00     2.5556e+07     7.6040e+06     6.7944e+06         3.36     600.00    1342.40
#         4808      1.0076%     0.0000e+00     2.9801e+07     8.2806e+06     7.5716e+06         3.60     600.00    1629.15
#         4543      0.8618%     0.0000e+00     2.3488e+07     5.3204e+06     5.1982e+06         4.41     600.00    1475.89
#         4769      0.7052%     0.0000e+00     1.8279e+07     4.9262e+06     4.2664e+06         3.71     600.00    1275.14
#         4620      0.6652%     0.0000e+00     1.8248e+07     4.4381e+06     4.9251e+06         4.11     600.00    1230.38
#         4912      0.6264%     0.0000e+00     3.2072e+07     5.4559e+06     5.8296e+06         5.88     600.00    1399.14
# Saved figure to teacher_outlier_chunk_4_inst_4696.png
#
# ==============================================================================================================
# Chunk 5: top 10 worst Teacher-vs-True cases (global indices [5000, 6000))
# ==============================================================================================================
#   global_idx     rel_l2_%        phi_min        phi_max       phi_mean        phi_std    peak/mean      T_min      T_max
#         5056      3.9349%     0.0000e+00     2.7119e+07     8.9957e+06     5.7317e+06         3.01     600.00    1627.58
#         5854      0.7000%     0.0000e+00     1.4085e+07     3.3966e+06     3.4737e+06         4.15     600.00    1107.56
#         5429      0.5323%     0.0000e+00     3.0517e+07     6.0329e+06     6.8547e+06         5.06     600.00    1530.21
#         5398      0.4659%     0.0000e+00     2.6698e+07     6.0300e+06     6.8909e+06         4.43     600.00    1452.04
#         5589      0.4651%     0.0000e+00     1.2043e+07     4.5953e+06     3.3034e+06         2.62     600.00    1156.97
#         5703      0.4394%     0.0000e+00     2.9210e+07     7.5099e+06     8.0838e+06         3.89     600.00    1495.87
#         5565      0.4247%     0.0000e+00     2.3134e+07     5.8115e+06     4.5020e+06         3.98     600.00    1380.52
#         5652      0.4077%     0.0000e+00     2.4743e+07     5.5399e+06     6.2392e+06         4.47     600.00    1277.22
#         5659      0.3943%     0.0000e+00     2.6423e+07     5.9383e+06     5.5430e+06         4.45     600.00    1215.31
#         5738      0.3921%     0.0000e+00     2.5156e+07     5.1396e+06     5.2027e+06         4.89     600.00    1176.43
# Saved figure to teacher_outlier_chunk_5_inst_5056.png
#
# ==============================================================================================================
# Chunk 6: top 10 worst Teacher-vs-True cases (global indices [6000, 7000))
# ==============================================================================================================
#   global_idx     rel_l2_%        phi_min        phi_max       phi_mean        phi_std    peak/mean      T_min      T_max
#         6502      6.4969%     0.0000e+00     4.3561e+07     9.8307e+06     9.4850e+06         4.43     600.00    2070.37
#         6840      4.0295%     0.0000e+00     2.4889e+07     5.4828e+06     4.4602e+06         4.54     600.00    1328.18
#         6239      3.5659%     0.0000e+00     3.0329e+07     7.2047e+06     6.3004e+06         4.21     600.00    1531.97
#         6283      3.2113%     0.0000e+00     1.6608e+07     5.2251e+06     3.7096e+06         3.18     600.00    1254.31
#         6839      3.1068%     0.0000e+00     2.2545e+07     5.3024e+06     5.0660e+06         4.25     600.00    1391.50
#         6545      3.1054%     0.0000e+00     2.3701e+07     5.9760e+06     4.7521e+06         3.97     600.00    1321.39
#         6948      2.5609%     0.0000e+00     1.5771e+07     3.3170e+06     3.1828e+06         4.75     600.00    1048.77
#         6673      2.4630%     0.0000e+00     2.3775e+07     6.6104e+06     5.8888e+06         3.60     600.00    1258.79
#         6657      2.4446%     0.0000e+00     1.3240e+07     4.3300e+06     3.2421e+06         3.06     600.00    1160.87
#         6759      2.3517%     0.0000e+00     3.2416e+07     8.9911e+06     7.5263e+06         3.61     600.00    1653.53
# Saved figure to teacher_outlier_chunk_6_inst_6502.png
#
# ==============================================================================================================
# Chunk 7: top 10 worst Teacher-vs-True cases (global indices [7000, 8000))
# ==============================================================================================================
#   global_idx     rel_l2_%        phi_min        phi_max       phi_mean        phi_std    peak/mean      T_min      T_max
#         7556      1.1859%     0.0000e+00     2.2419e+07     7.0469e+06     5.3966e+06         3.18     600.00    1490.00
#         7285      1.1693%     0.0000e+00     1.6060e+07     5.0014e+06     4.3283e+06         3.21     600.00    1298.96
#         7424      0.9235%     0.0000e+00     2.1460e+07     7.0491e+06     5.6859e+06         3.04     600.00    1470.21
#         7265      0.7957%     0.0000e+00     1.7547e+07     6.8862e+06     5.3138e+06         2.55     600.00    1423.45
#         7927      0.7506%     0.0000e+00     2.5375e+07     7.9533e+06     6.2456e+06         3.19     600.00    1668.15
#         7089      0.6690%     0.0000e+00     2.6660e+07     8.9960e+06     6.0055e+06         2.96     600.00    1698.58
#         7932      0.6106%     0.0000e+00     1.0378e+07     4.2592e+06     2.6444e+06         2.44     600.00    1063.37
#         7706      0.5595%     0.0000e+00     1.5972e+07     6.2124e+06     4.0455e+06         2.57     600.00    1256.40
#         7325      0.4961%     0.0000e+00     2.2727e+07     5.7457e+06     4.8770e+06         3.96     600.00    1346.55
#         7000      0.4884%     0.0000e+00     2.3185e+07     6.7895e+06     5.2493e+06         3.41     600.00    1509.08
# Saved figure to teacher_outlier_chunk_7_inst_7556.png
#
# ==============================================================================================================
# Chunk 8: top 10 worst Teacher-vs-True cases (global indices [8000, 9000))
# ==============================================================================================================
#   global_idx     rel_l2_%        phi_min        phi_max       phi_mean        phi_std    peak/mean      T_min      T_max
#         8339      1.9844%     0.0000e+00     3.2947e+07     8.0240e+06     7.3395e+06         4.11     600.00    1563.43
#         8552      1.2503%     0.0000e+00     2.1889e+07     5.4004e+06     5.1462e+06         4.05     600.00    1402.66
#         8672      0.7008%     0.0000e+00     2.0781e+07     6.6287e+06     5.2412e+06         3.13     600.00    1546.80
#         8377      0.6694%     0.0000e+00     3.2454e+07     8.3340e+06     7.4162e+06         3.89     600.00    1444.00
#         8500      0.6210%     0.0000e+00     2.6905e+07     8.0125e+06     6.6541e+06         3.36     600.00    1817.77
#         8252      0.5709%     0.0000e+00     2.5835e+07     5.5982e+06     6.3696e+06         4.61     600.00    1209.81
#         8534      0.4562%     0.0000e+00     2.1031e+07     6.1331e+06     5.0204e+06         3.43     600.00    1427.43
#         8333      0.4457%     0.0000e+00     2.1750e+07     5.5186e+06     5.0477e+06         3.94     600.00    1254.00
#         8510      0.4435%     0.0000e+00     2.5384e+07     5.8901e+06     5.5452e+06         4.31     600.00    1487.94
#         8392      0.4185%     0.0000e+00     3.4776e+07     6.5150e+06     6.6137e+06         5.34     600.00    1405.45
# Saved figure to teacher_outlier_chunk_8_inst_8339.png
#
# ==============================================================================================================
# Chunk 9: top 10 worst Teacher-vs-True cases (global indices [9000, 10000))
# ==============================================================================================================
#   global_idx     rel_l2_%        phi_min        phi_max       phi_mean        phi_std    peak/mean      T_min      T_max
#         9049      1.9430%     0.0000e+00     2.8991e+07     8.9225e+06     7.0158e+06         3.25     600.00    1701.75
#         9950      1.7804%     0.0000e+00     2.3757e+07     4.9298e+06     4.9421e+06         4.82     600.00    1332.97
#         9589      1.7734%     0.0000e+00     1.8434e+07     5.6354e+06     4.3944e+06         3.27     600.00    1394.11
#         9315      1.7653%     0.0000e+00     3.2257e+07     5.3948e+06     7.1238e+06         5.98     600.00    1215.30
#         9363      1.7135%     0.0000e+00     2.1314e+07     6.1752e+06     5.4099e+06         3.45     600.00    1359.10
#         9386      1.7092%     0.0000e+00     1.7370e+07     5.5949e+06     4.7056e+06         3.10     600.00    1250.73
#         9521      1.6799%     0.0000e+00     3.6322e+07     8.7680e+06     9.2998e+06         4.14     600.00    1561.51
#         9161      1.5754%     0.0000e+00     3.1470e+07     6.6731e+06     7.6599e+06         4.72     600.00    1317.87
#         9607      1.4494%     0.0000e+00     1.6428e+07     4.2158e+06     3.6122e+06         3.90     600.00    1164.34
#         9539      1.4258%     0.0000e+00     2.9073e+07     6.6369e+06     7.2797e+06         4.38     600.00    1294.18
# Saved figure to teacher_outlier_chunk_9_inst_9049.png
#
# Process finished with exit code 0


if __name__ == "__main__":
    main()
