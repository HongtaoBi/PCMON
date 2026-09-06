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

from PDE_Solver import physics_params, solve_2d_phi_2_T


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

        T_inlet = 1.0
        sigma = 0.05
        coeff = 1.0 - torch.exp(-x_raw / sigma)
        return coeff * output + T_inlet


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
            pred = (model(X, batch_indices) * 600.0).squeeze(-1)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_chunk(phi_chunk, chunk_id, nx=201, ny=101, batch_size=50,
                   checkpoint_dir='.', predict_only=False,
                   T_true_path='T_true_10000_101_201.npy'):
    L = physics_params['L']
    H = physics_params['H']
    N = phi_chunk.shape[0]

    model_path = os.path.join(checkpoint_dir, f'Teacher_operator_phi_2_T_adam_fd_chunk_{chunk_id}.pth')
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
    T_preds = predict_all(model, grid_points, N, batch_size=batch_size)
    T_preds = T_preds.reshape(N, ny, nx)
    print(f"Prediction done in {time.time() - t0:.2f}s")

    if predict_only:
        return None, None, None, T_preds

    print(f"Loading reference T from {T_true_path}...")
    T_true = np.load(T_true_path)
    start = chunk_id * N
    end = start + N
    T_ref_chunk = T_true[start:end]
    assert T_ref_chunk.shape == (N, ny, nx), f"T_true chunk shape {T_ref_chunk.shape} != ({N}, {ny}, {nx})"

    mse_list = []
    l_inf_list = []
    rel_l2_list = []

    for i in range(N):
        T_pred = T_preds[i]
        T_ref = T_ref_chunk[i]
        mse = np.mean((T_pred - T_ref) ** 2)
        l_inf = np.max(np.abs(T_pred - T_ref))
        rel_l2 = np.linalg.norm(T_pred - T_ref) / np.linalg.norm(T_ref)
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

    return mse_arr, l_inf_arr, rel_l2_arr, T_preds


def main():
    parser = argparse.ArgumentParser(description='Validate all chunk checkpoints.')
    parser.add_argument('--num_chunks', type=int, default=10)
    parser.add_argument('--instances_per_chunk', type=int, default=1000)
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--T_true_file', type=str, default='T_true_10000_101_201.npy')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default='Teacher_operator_phi_2_T_adam_fd_chunks',
                        help='Directory containing chunk .pth checkpoints.')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Save all predicted T fields as a single .npy file.')
    parser.add_argument('--no_save_predictions', action='store_true',
                        help='Do not save predicted T fields.')
    parser.add_argument('--predict_only', action='store_true',
                        help='Only run forward predictions and save them; skip reference solver and error metrics.')
    args = parser.parse_args()

    phi_dataset = np.load(args.phi_file)
    phi_dataset = phi_dataset.transpose(0, 2, 1)
    print(f"Loaded full dataset: {phi_dataset.shape}")

    all_mse = []
    all_l_inf = []
    all_rel_l2 = []
    all_T_preds = []

    for k in range(args.num_chunks):
        start = k * args.instances_per_chunk
        end = min(start + args.instances_per_chunk, phi_dataset.shape[0])
        phi_chunk = phi_dataset[start:end]

        mse_arr, l_inf_arr, rel_l2_arr, T_preds = evaluate_chunk(
            phi_chunk, k, batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            predict_only=args.predict_only,
            T_true_path=args.T_true_file
        )
        all_mse.append(mse_arr)
        all_l_inf.append(l_inf_arr)
        all_rel_l2.append(rel_l2_arr)
        if (args.save_predictions and not args.no_save_predictions) or args.predict_only:
            all_T_preds.append(T_preds)

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

    if (args.save_predictions and not args.no_save_predictions) or args.predict_only:
        T_all = np.concatenate(all_T_preds, axis=0)
        output_file = 'T_pred_10000_101_201.npy'
        np.save(output_file, T_all)
        print(f"\nSaved all predictions to {output_file}, shape {T_all.shape}")


# "C:\python\Program Files\Python310\python.exe" "C:\python\PythonFiles\CNRS Benchmark\2D_PCMON\Teacher_phi_2_T\validate_chunks.py"
# Loaded full dataset: (10000, 101, 201)
#
# Evaluating chunk 0: 1000 instances from Teacher_operator_phi_2_T_adam_fd_chunks\Teacher_operator_phi_2_T_adam_fd_chunk_0.pth
# Prediction done in 7.61s
# Loading reference T from T_true_10000_101_201.npy...
# Chunk 0 statistics:
#   MSE:         mean=2.202947e+00, min=2.170833e-01, max=1.147616e+02
#   L_inf:       mean=3.884437e+00, min=9.888004e-01, max=8.256137e+01
#   Relative L2: mean=1.507238e-03 (0.1507%), min=6.410293e-04 (0.0641%), max=9.963913e-03 (0.9964%)
#
# Evaluating chunk 1: 1000 instances from Teacher_operator_phi_2_T_adam_fd_chunks\Teacher_operator_phi_2_T_adam_fd_chunk_1.pth
# Prediction done in 6.52s
# Loading reference T from T_true_10000_101_201.npy...
# Chunk 1 statistics:
#   MSE:         mean=2.592523e+00, min=2.702581e-01, max=2.023807e+02
#   L_inf:       mean=3.837911e+00, min=9.483398e-01, max=7.265621e+01
#   Relative L2: mean=1.576668e-03 (0.1577%), min=7.371767e-04 (0.0737%), max=1.573831e-02 (1.5738%)
#
# Evaluating chunk 2: 1000 instances from Teacher_operator_phi_2_T_adam_fd_chunks\Teacher_operator_phi_2_T_adam_fd_chunk_2.pth
# Prediction done in 6.50s
# Loading reference T from T_true_10000_101_201.npy...
# Chunk 2 statistics:
#   MSE:         mean=1.039428e+01, min=1.179625e-01, max=2.694811e+03
#   L_inf:       mean=5.376910e+00, min=8.943792e-01, max=2.668775e+02
#   Relative L2: mean=1.761955e-03 (0.1762%), min=4.949851e-04 (0.0495%), max=4.691748e-02 (4.6917%)
#
# Evaluating chunk 3: 1000 instances from Teacher_operator_phi_2_T_adam_fd_chunks\Teacher_operator_phi_2_T_adam_fd_chunk_3.pth
# Prediction done in 6.50s
# Loading reference T from T_true_10000_101_201.npy...
# Chunk 3 statistics:
#   MSE:         mean=2.869427e+00, min=2.604774e-01, max=3.654538e+02
#   L_inf:       mean=3.797036e+00, min=9.131649e-01, max=1.143119e+02
#   Relative L2: mean=1.556900e-03 (0.1557%), min=7.093060e-04 (0.0709%), max=1.737598e-02 (1.7376%)
#
# Evaluating chunk 4: 1000 instances from Teacher_operator_phi_2_T_adam_fd_chunks\Teacher_operator_phi_2_T_adam_fd_chunk_4.pth
# Prediction done in 6.50s
# Loading reference T from T_true_10000_101_201.npy...
# Chunk 4 statistics:
#   MSE:         mean=6.554559e+00, min=1.650503e-01, max=2.415735e+03
#   L_inf:       mean=4.942481e+00, min=7.871387e-01, max=2.997928e+02
#   Relative L2: mean=1.708309e-03 (0.1708%), min=5.832447e-04 (0.0583%), max=5.168144e-02 (5.1681%)
#
# Evaluating chunk 5: 1000 instances from Teacher_operator_phi_2_T_adam_fd_chunks\Teacher_operator_phi_2_T_adam_fd_chunk_5.pth
# Prediction done in 6.50s
# Loading reference T from T_true_10000_101_201.npy...
# Chunk 5 statistics:
#   MSE:         mean=4.024105e+00, min=1.926494e-01, max=1.941332e+03
#   L_inf:       mean=4.319804e+00, min=8.355089e-01, max=2.749392e+02
#   Relative L2: mean=1.565730e-03 (0.1566%), min=6.270202e-04 (0.0627%), max=3.934857e-02 (3.9349%)
#
# Evaluating chunk 6: 1000 instances from Teacher_operator_phi_2_T_adam_fd_chunks\Teacher_operator_phi_2_T_adam_fd_chunk_6.pth
# Prediction done in 6.50s
# Loading reference T from T_true_10000_101_201.npy...
# Chunk 6 statistics:
#   MSE:         mean=2.291754e+01, min=2.046560e-01, max=6.134044e+03
#   L_inf:       mean=8.358781e+00, min=9.865176e-01, max=3.458282e+02
#   Relative L2: mean=2.414933e-03 (0.2415%), min=6.189269e-04 (0.0619%), max=6.496940e-02 (6.4969%)
#
# Evaluating chunk 7: 1000 instances from Teacher_operator_phi_2_T_adam_fd_chunks\Teacher_operator_phi_2_T_adam_fd_chunk_7.pth
# Prediction done in 6.49s
# Loading reference T from T_true_10000_101_201.npy...
# Chunk 7 statistics:
#   MSE:         mean=2.616398e+00, min=1.823536e-01, max=1.449222e+02
#   L_inf:       mean=4.373650e+00, min=8.888998e-01, max=7.203480e+01
#   Relative L2: mean=1.555543e-03 (0.1556%), min=6.024996e-04 (0.0602%), max=1.185944e-02 (1.1859%)
#
# Evaluating chunk 8: 1000 instances from Teacher_operator_phi_2_T_adam_fd_chunks\Teacher_operator_phi_2_T_adam_fd_chunk_8.pth
# Prediction done in 6.51s
# Loading reference T from T_true_10000_101_201.npy...
# Chunk 8 statistics:
#   MSE:         mean=3.068972e+00, min=2.021128e-01, max=4.811966e+02
#   L_inf:       mean=5.130155e+00, min=1.151741e+00, max=1.026464e+02
#   Relative L2: mean=1.653337e-03 (0.1653%), min=6.397457e-04 (0.0640%), max=1.984374e-02 (1.9844%)
#
# Evaluating chunk 9: 1000 instances from Teacher_operator_phi_2_T_adam_fd_chunks\Teacher_operator_phi_2_T_adam_fd_chunk_9.pth
# Prediction done in 6.50s
# Loading reference T from T_true_10000_101_201.npy...
# Chunk 9 statistics:
#   MSE:         mean=1.017165e+01, min=1.590268e-01, max=4.959875e+02
#   L_inf:       mean=1.103243e+01, min=9.169562e-01, max=1.806950e+02
#   Relative L2: mean=2.347534e-03 (0.2348%), min=5.476176e-04 (0.0548%), max=1.943043e-02 (1.9430%)
#
# ========================================
# OVERALL STATISTICS FOR ALL 10000 INSTANCES
# ========================================
#   MSE:         mean=6.741241e+00, min=1.179625e-01, max=6.134044e+03
#   L_inf:       mean=5.505359e+00, min=7.871387e-01, max=3.458282e+02
#   Relative L2: mean=1.764815e-03 (0.1765%), min=4.949851e-04 (0.0495%), max=6.496940e-02 (6.4969%)
#
# Saved all predictions to T_pred_10000_101_201.npy, shape (10000, 101, 201)
#
# Process finished with exit code 0

if __name__ == '__main__':
    main()
