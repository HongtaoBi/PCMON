"""
Validate all trained chunk checkpoints against the numerical reference solver.

Usage:
    python validate_chunks.py --num_chunks 10 --instances_per_chunk 1000
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import os

from PDE_Solver import physics_params


C_SCALE = 5e5


class ParallelPINOs(nn.Module):
    def __init__(self, num_instances=10000, hidden_dim=64, H=1.0):
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

        C_inlet = 0.0
        sigma = 0.05
        coeff = 1.0 - torch.exp(-x_raw / sigma)
        return coeff * output + C_inlet


def predict_all(model, grid_points, num_instances, batch_size=200):
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
            shift = torch.stack([z_x, z_y], dim=-1)
            X = grid_points.unsqueeze(0) + shift
            pred = (model(X, batch_indices) * C_SCALE).squeeze(-1)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_chunk(phi_chunk, C_ref_chunk, chunk_id, nx=201, ny=101, batch_size=50, checkpoint_dir='.', predict_only=False):
    L = physics_params['L']
    H = physics_params['H']
    N = phi_chunk.shape[0]

    model_path = os.path.join(checkpoint_dir, f'Teacher_operator_phi_2_C_adam_fd_chunk_{chunk_id}.pth')
    if predict_only:
        print(f"\nPredicting chunk {chunk_id}: {N} instances from {model_path}")
    else:
        print(f"\nEvaluating chunk {chunk_id}: {N} instances from {model_path}")

    model = ParallelPINOs(num_instances=N, hidden_dim=64, H=H).double().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y)
    grid_points_single = np.column_stack([X.flatten(), Y.flatten()])
    grid_points = torch.tensor(grid_points_single, dtype=torch.float64, device='cuda')

    t0 = time.time()
    C_preds = predict_all(model, grid_points, N, batch_size=batch_size)
    C_preds = C_preds.reshape(N, ny, nx)
    print(f"Prediction done in {time.time() - t0:.2f}s")

    if predict_only:
        return None, None, None, C_preds

    mse_list = []
    l_inf_list = []
    rel_l2_list = []

    for i in range(N):
        C_pred = C_preds[i]
        C_ref = C_ref_chunk[i]
        mse = np.mean((C_pred - C_ref) ** 2)
        l_inf = np.max(np.abs(C_pred - C_ref))
        rel_l2 = np.linalg.norm(C_pred - C_ref) / (np.linalg.norm(C_ref) + 1e-12)
        mse_list.append(mse)
        l_inf_list.append(l_inf)
        rel_l2_list.append(rel_l2)

    mse_arr = np.array(mse_list)
    l_inf_arr = np.array(l_inf_list)
    rel_l2_arr = np.array(rel_l2_list)

    print(f"Chunk {chunk_id} statistics:")
    print(f"  MSE:         mean={mse_arr.mean():.6e}, min={mse_arr.min():.6e}, max={mse_arr.max():.6e}")
    print(f"  L_inf:       mean={l_inf_arr.mean():.6e}, min={l_inf_arr.min():.6e}, max={l_inf_arr.max():.6e}")
    print(f"  Relative L2: mean={rel_l2_arr.mean():.6e} ({rel_l2_arr.mean()*100:.4f}%), "
          f"min={rel_l2_arr.min():.6e} ({rel_l2_arr.min()*100:.4f}%), "
          f"max={rel_l2_arr.max():.6e} ({rel_l2_arr.max()*100:.4f}%)")

    return mse_arr, l_inf_arr, rel_l2_arr, C_preds


def main():
    parser = argparse.ArgumentParser(description='Validate all chunk checkpoints.')
    parser.add_argument('--num_chunks', type=int, default=10)
    parser.add_argument('--instances_per_chunk', type=int, default=1000)
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--c_true_file', type=str, default='C_true_10000_101_201.npy',
                        help='Precomputed reference C fields (N, ny, nx).')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default='Teacher_operator_phi_2_C_adam_fd_chunks',
                        help='Directory containing chunk .pth checkpoints.')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Save all predicted C fields as a single .npy file.')
    parser.add_argument('--predict_only', action='store_true',
                        help='Only run forward predictions and save them; skip reference solver and error metrics.')
    args = parser.parse_args()

    phi_dataset = np.load(args.phi_file)
    phi_dataset = phi_dataset.transpose(0, 2, 1)
    print(f"Loaded full phi dataset: {phi_dataset.shape}")

    C_true = np.load(args.c_true_file)
    print(f"Loaded reference C dataset: {C_true.shape}")
    assert C_true.shape[0] == phi_dataset.shape[0], \
        f"phi has {phi_dataset.shape[0]} instances but C_true has {C_true.shape[0]}"

    all_mse = []
    all_l_inf = []
    all_rel_l2 = []
    all_C_preds = []

    for k in range(args.num_chunks):
        start = k * args.instances_per_chunk
        end = min(start + args.instances_per_chunk, phi_dataset.shape[0])
        phi_chunk = phi_dataset[start:end]
        C_ref_chunk = C_true[start:end]

        mse_arr, l_inf_arr, rel_l2_arr, C_preds = evaluate_chunk(
            phi_chunk, C_ref_chunk, k, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir,
            predict_only=args.predict_only
        )
        all_mse.append(mse_arr)
        all_l_inf.append(l_inf_arr)
        all_rel_l2.append(rel_l2_arr)
        if args.save_predictions or args.predict_only:
            all_C_preds.append(C_preds)

    if not args.predict_only:
        all_mse = np.concatenate(all_mse)
        all_l_inf = np.concatenate(all_l_inf)
        all_rel_l2 = np.concatenate(all_rel_l2)

        print("\n========================================")
        print("OVERALL STATISTICS FOR ALL 10000 INSTANCES")
        print("========================================")
        print(f"  MSE:         mean={all_mse.mean():.6e}, min={all_mse.min():.6e}, max={all_mse.max():.6e}")
        print(f"  L_inf:       mean={all_l_inf.mean():.6e}, min={all_l_inf.min():.6e}, max={all_l_inf.max():.6e}")
        print(f"  Relative L2: mean={all_rel_l2.mean():.6e} ({all_rel_l2.mean()*100:.4f}%), "
              f"min={all_rel_l2.min():.6e} ({all_rel_l2.min()*100:.4f}%), "
              f"max={all_rel_l2.max():.6e} ({all_rel_l2.max()*100:.4f}%)")

    if args.save_predictions or args.predict_only:
        C_all = np.concatenate(all_C_preds, axis=0)
        output_file = 'C_pred_10000_101_201.npy'
        np.save(output_file, C_all)
        print(f"\nSaved all predictions to {output_file}, shape {C_all.shape}")

#
# "C:\python\Program Files\Python310\python.exe" "C:\python\PythonFiles\CNRS Benchmark\2D_PCMON\Teacher_phi_2_C\validate_chunks.py"
# Loaded full phi dataset: (10000, 101, 201)
# Loaded reference C dataset: (10000, 101, 201)
#
# Evaluating chunk 0: 1000 instances from Teacher_operator_phi_2_C_adam_fd_chunks\Teacher_operator_phi_2_C_adam_fd_chunk_0.pth
# Prediction done in 7.64s
# Chunk 0 statistics:
#   MSE:         mean=9.491722e+06, min=1.749252e+05, max=1.948301e+08
#   L_inf:       mean=2.293980e+04, min=7.971020e+02, max=1.714363e+05
#   Relative L2: mean=7.300202e-03 (0.7300%), min=2.243040e-03 (0.2243%), max=3.502436e-02 (3.5024%)
#
# Evaluating chunk 1: 1000 instances from Teacher_operator_phi_2_C_adam_fd_chunks\Teacher_operator_phi_2_C_adam_fd_chunk_1.pth
# Prediction done in 6.79s
# Chunk 1 statistics:
#   MSE:         mean=1.449457e+07, min=2.587797e+05, max=1.655417e+08
#   L_inf:       mean=3.280739e+04, min=2.054580e+03, max=1.504132e+05
#   Relative L2: mean=1.051702e-02 (1.0517%), min=2.538690e-03 (0.2539%), max=3.734250e-02 (3.7342%)
#
# Evaluating chunk 2: 1000 instances from Teacher_operator_phi_2_C_adam_fd_chunks\Teacher_operator_phi_2_C_adam_fd_chunk_2.pth
# Prediction done in 6.81s
# Chunk 2 statistics:
#   MSE:         mean=2.198872e+06, min=1.150466e+05, max=4.888916e+07
#   L_inf:       mean=7.317535e+03, min=8.976228e+02, max=8.899181e+04
#   Relative L2: mean=4.459460e-03 (0.4459%), min=1.894175e-03 (0.1894%), max=1.785634e-02 (1.7856%)
#
# Evaluating chunk 3: 1000 instances from Teacher_operator_phi_2_C_adam_fd_chunks\Teacher_operator_phi_2_C_adam_fd_chunk_3.pth
# Prediction done in 6.76s
# Chunk 3 statistics:
#   MSE:         mean=3.612700e+06, min=1.162602e+05, max=1.804784e+08
#   L_inf:       mean=1.015513e+04, min=8.817298e+02, max=1.192385e+05
#   Relative L2: mean=4.522168e-03 (0.4522%), min=1.960542e-03 (0.1961%), max=5.181294e-02 (5.1813%)
#
# Evaluating chunk 4: 1000 instances from Teacher_operator_phi_2_C_adam_fd_chunks\Teacher_operator_phi_2_C_adam_fd_chunk_4.pth
# Prediction done in 6.80s
# Chunk 4 statistics:
#   MSE:         mean=1.172696e+07, min=1.125830e+05, max=1.855053e+08
#   L_inf:       mean=2.746288e+04, min=8.425084e+02, max=1.442474e+05
#   Relative L2: mean=9.360555e-03 (0.9361%), min=2.390258e-03 (0.2390%), max=3.571983e-02 (3.5720%)
#
# Evaluating chunk 5: 1000 instances from Teacher_operator_phi_2_C_adam_fd_chunks\Teacher_operator_phi_2_C_adam_fd_chunk_5.pth
# Prediction done in 6.79s
# Chunk 5 statistics:
#   MSE:         mean=5.471870e+06, min=2.282412e+05, max=7.459443e+07
#   L_inf:       mean=1.833840e+04, min=9.633532e+02, max=8.101422e+04
#   Relative L2: mean=6.640010e-03 (0.6640%), min=3.050312e-03 (0.3050%), max=2.477750e-02 (2.4777%)
#
# Evaluating chunk 6: 1000 instances from Teacher_operator_phi_2_C_adam_fd_chunks\Teacher_operator_phi_2_C_adam_fd_chunk_6.pth
# Prediction done in 6.81s
# Chunk 6 statistics:
#   MSE:         mean=4.085934e+06, min=1.226119e+05, max=6.361252e+07
#   L_inf:       mean=1.615760e+04, min=8.097774e+02, max=1.049448e+05
#   Relative L2: mean=5.322201e-03 (0.5322%), min=2.118141e-03 (0.2118%), max=1.869906e-02 (1.8699%)
#
# Evaluating chunk 7: 1000 instances from Teacher_operator_phi_2_C_adam_fd_chunks\Teacher_operator_phi_2_C_adam_fd_chunk_7.pth
# Prediction done in 6.79s
# Chunk 7 statistics:
#   MSE:         mean=3.177206e+06, min=1.304754e+05, max=5.965665e+07
#   L_inf:       mean=1.251826e+04, min=8.717053e+02, max=9.110705e+04
#   Relative L2: mean=5.105731e-03 (0.5106%), min=2.412656e-03 (0.2413%), max=1.799474e-02 (1.7995%)
#
# Evaluating chunk 8: 1000 instances from Teacher_operator_phi_2_C_adam_fd_chunks\Teacher_operator_phi_2_C_adam_fd_chunk_8.pth
# Prediction done in 6.80s
# Chunk 8 statistics:
#   MSE:         mean=4.286523e+06, min=1.187553e+05, max=1.829844e+08
#   L_inf:       mean=1.772125e+04, min=1.100267e+03, max=1.779590e+05
#   Relative L2: mean=5.647486e-03 (0.5647%), min=2.368415e-03 (0.2368%), max=2.878614e-02 (2.8786%)
#
# Evaluating chunk 9: 1000 instances from Teacher_operator_phi_2_C_adam_fd_chunks\Teacher_operator_phi_2_C_adam_fd_chunk_9.pth
# Prediction done in 6.79s
# Chunk 9 statistics:
#   MSE:         mean=2.112989e+06, min=1.443121e+05, max=5.215167e+07
#   L_inf:       mean=7.265377e+03, min=1.058696e+03, max=6.132562e+04
#   Relative L2: mean=4.361176e-03 (0.4361%), min=2.216023e-03 (0.2216%), max=1.721390e-02 (1.7214%)
#
# ========================================
# OVERALL STATISTICS FOR ALL 10000 INSTANCES
# ========================================
#   MSE:         mean=6.065934e+06, min=1.125830e+05, max=1.948301e+08
#   L_inf:       mean=1.726836e+04, min=7.971020e+02, max=1.779590e+05
#   Relative L2: mean=6.323601e-03 (0.6324%), min=1.894175e-03 (0.1894%), max=5.181294e-02 (5.1813%)
#
# Saved all predictions to C_pred_10000_101_201.npy, shape (10000, 101, 201)
#
# Process finished with exit code 0



if __name__ == '__main__':
    main()
