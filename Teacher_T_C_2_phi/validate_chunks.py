"""
Validate all trained T,C -> phi chunk checkpoints against the numerical reference phi_true.

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


PHI_SCALE = 2.5e7


class ParallelPINOs(nn.Module):
    def __init__(self, num_instances=10000, hidden_dim=64, L=2.0, H=1.0):
        super(ParallelPINOs, self).__init__()
        self.num_instances = num_instances
        self.L = L
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

        sigma_x = 0.05
        sigma_y = 0.05
        coeff_x = 1.0 - torch.exp(-x_raw / sigma_x) - torch.exp(-(self.L - x_raw) / sigma_x)
        coeff_y = 1.0 - torch.exp(-y_raw / sigma_y) - torch.exp(-(self.H - y_raw) / sigma_y)
        coeff = coeff_x * coeff_y
        return coeff * output


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
            pred = (model(X, batch_indices) * PHI_SCALE).squeeze(-1)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_chunk(T_chunk, C_chunk, phi_ref_chunk, chunk_id, nx=201, ny=101,
                   batch_size=50, checkpoint_dir='.', predict_only=False):
    L = physics_params['L']
    H = physics_params['H']
    N = T_chunk.shape[0]

    model_path = os.path.join(checkpoint_dir, f'Teacher_operator_T_C_2_phi_adam_fd_chunk_{chunk_id}.pth')
    if predict_only:
        print(f"\nPredicting chunk {chunk_id}: {N} instances from {model_path}")
    else:
        print(f"\nEvaluating chunk {chunk_id}: {N} instances from {model_path}")

    model = ParallelPINOs(num_instances=N, hidden_dim=64, L=L, H=H).double().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y)
    grid_points_single = np.column_stack([X.flatten(), Y.flatten()])
    grid_points = torch.tensor(grid_points_single, dtype=torch.float64, device='cuda')

    t0 = time.time()
    phi_preds = predict_all(model, grid_points, N, batch_size=batch_size)
    phi_preds = phi_preds.reshape(N, ny, nx)
    print(f"Prediction done in {time.time() - t0:.2f}s")

    if predict_only:
        return None, None, None, phi_preds

    mse_list = []
    l_inf_list = []
    rel_l2_list = []

    for i in range(N):
        phi_pred = phi_preds[i]
        phi_ref = phi_ref_chunk[i]
        mse = np.mean((phi_pred - phi_ref) ** 2)
        l_inf = np.max(np.abs(phi_pred - phi_ref))
        rel_l2 = np.linalg.norm(phi_pred - phi_ref) / (np.linalg.norm(phi_ref) + 1e-12)
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

    return mse_arr, l_inf_arr, rel_l2_arr, phi_preds


def main():
    parser = argparse.ArgumentParser(description='Validate all T,C -> phi chunk checkpoints.')
    parser.add_argument('--num_chunks', type=int, default=10)
    parser.add_argument('--instances_per_chunk', type=int, default=1000)
    parser.add_argument('--T_file', type=str, default='1_FFT_T_10000_101_201.npy')
    parser.add_argument('--C_file', type=str, default='1_FFT_C_10000_101_201.npy')
    parser.add_argument('--phi_true_file', type=str, default='phi_true_10000_201_101.npy',
                        help='Precomputed reference phi fields (N, nx, ny).')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default='Teacher_operator_T_C_2_phi_adam_fd_chunks',
                        help='Directory containing chunk .pth checkpoints.')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Save all predicted phi fields as a single .npy file.')
    parser.add_argument('--predict_only', action='store_true',
                        help='Only run forward predictions and save them; skip reference solver and error metrics.')
    args = parser.parse_args()

    T_dataset = np.load(args.T_file)
    C_dataset = np.load(args.C_file)
    print(f"Loaded T: {T_dataset.shape}, C: {C_dataset.shape}")

    phi_true = np.load(args.phi_true_file)
    phi_true = phi_true.transpose(0, 2, 1)
    print(f"Loaded reference phi: {phi_true.shape}")

    N_full = T_dataset.shape[0]
    assert C_dataset.shape[0] == N_full
    assert phi_true.shape[0] == N_full

    all_mse = []
    all_l_inf = []
    all_rel_l2 = []
    all_phi_preds = []

    for k in range(args.num_chunks):
        start = k * args.instances_per_chunk
        end = min(start + args.instances_per_chunk, N_full)
        T_chunk = T_dataset[start:end]
        C_chunk = C_dataset[start:end]
        phi_ref_chunk = phi_true[start:end]

        mse_arr, l_inf_arr, rel_l2_arr, phi_preds = evaluate_chunk(
            T_chunk, C_chunk, phi_ref_chunk, k,
            batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir,
            predict_only=args.predict_only
        )
        all_mse.append(mse_arr)
        all_l_inf.append(l_inf_arr)
        all_rel_l2.append(rel_l2_arr)
        if args.save_predictions or args.predict_only:
            all_phi_preds.append(phi_preds)

    if not args.predict_only:
        all_mse = np.concatenate(all_mse)
        all_l_inf = np.concatenate(all_l_inf)
        all_rel_l2 = np.concatenate(all_rel_l2)

        print("\n========================================")
        print("OVERALL STATISTICS FOR ALL INSTANCES")
        print("========================================")
        print(f"  MSE:         mean={all_mse.mean():.6e}, min={all_mse.min():.6e}, max={all_mse.max():.6e}")
        print(f"  L_inf:       mean={all_l_inf.mean():.6e}, min={all_l_inf.min():.6e}, max={all_l_inf.max():.6e}")
        print(f"  Relative L2: mean={all_rel_l2.mean():.6e} ({all_rel_l2.mean()*100:.4f}%), "
              f"min={all_rel_l2.min():.6e} ({all_rel_l2.min()*100:.4f}%), "
              f"max={all_rel_l2.max():.6e} ({all_rel_l2.max()*100:.4f}%)")

    if args.save_predictions or args.predict_only:
        phi_all = np.concatenate(all_phi_preds, axis=0)
        output_file = 'phi_pred_10000_201_101.npy'
        np.save(output_file, phi_all)
        print(f"\nSaved all predictions to {output_file}, shape {phi_all.shape}")


# "C:\python\Program Files\Python310\python.exe" "C:\python\PythonFiles\CNRS Benchmark\2D_PCMON\Teacher_T_C_2_phi\validate_chunks.py"
# Loaded T: (10000, 101, 201), C: (10000, 101, 201)
# Loaded reference phi: (10000, 101, 201)
#
# Evaluating chunk 0: 1000 instances from Teacher_operator_T_C_2_phi_adam_fd_chunks\Teacher_operator_T_C_2_phi_adam_fd_chunk_0.pth
# Prediction done in 7.53s
# Chunk 0 statistics:
#   MSE:         mean=4.429945e+07, min=1.887562e+06, max=8.760559e+09
#   L_inf:       mean=1.933923e+04, min=5.005080e+03, max=6.832684e+05
#   Relative L2: mean=7.564636e-04 (0.0756%), min=2.137909e-04 (0.0214%), max=1.204110e-02 (1.2041%)
#
# Evaluating chunk 1: 1000 instances from Teacher_operator_T_C_2_phi_adam_fd_chunks\Teacher_operator_T_C_2_phi_adam_fd_chunk_1.pth
# Prediction done in 6.40s
# Chunk 1 statistics:
#   MSE:         mean=3.078412e+07, min=7.137790e+05, max=9.445773e+08
#   L_inf:       mean=1.564172e+04, min=4.062038e+03, max=8.431893e+04
#   Relative L2: mean=6.982838e-04 (0.0698%), min=1.157756e-04 (0.0116%), max=4.640218e-03 (0.4640%)
#
# Evaluating chunk 2: 1000 instances from Teacher_operator_T_C_2_phi_adam_fd_chunks\Teacher_operator_T_C_2_phi_adam_fd_chunk_2.pth
# Prediction done in 6.61s
# Chunk 2 statistics:
#   MSE:         mean=4.313313e+07, min=2.272398e+06, max=9.668055e+08
#   L_inf:       mean=2.085303e+04, min=5.327983e+03, max=1.432827e+05
#   Relative L2: mean=8.390734e-04 (0.0839%), min=2.096807e-04 (0.0210%), max=4.107057e-03 (0.4107%)
#
# Evaluating chunk 3: 1000 instances from Teacher_operator_T_C_2_phi_adam_fd_chunks\Teacher_operator_T_C_2_phi_adam_fd_chunk_3.pth
# Prediction done in 6.79s
# Chunk 3 statistics:
#   MSE:         mean=1.413050e+07, min=1.252684e+06, max=1.011844e+09
#   L_inf:       mean=1.115696e+04, min=4.116323e+03, max=8.651107e+04
#   Relative L2: mean=4.558573e-04 (0.0456%), min=1.721212e-04 (0.0172%), max=4.860637e-03 (0.4861%)
#
# Evaluating chunk 4: 1000 instances from Teacher_operator_T_C_2_phi_adam_fd_chunks\Teacher_operator_T_C_2_phi_adam_fd_chunk_4.pth
# Prediction done in 6.59s
# Chunk 4 statistics:
#   MSE:         mean=3.346638e+07, min=1.333400e+06, max=7.595350e+08
#   L_inf:       mean=1.603353e+04, min=5.036153e+03, max=7.415407e+04
#   Relative L2: mean=7.412506e-04 (0.0741%), min=1.608142e-04 (0.0161%), max=3.568136e-03 (0.3568%)
#
# Evaluating chunk 5: 1000 instances from Teacher_operator_T_C_2_phi_adam_fd_chunks\Teacher_operator_T_C_2_phi_adam_fd_chunk_5.pth
# Prediction done in 6.56s
# Chunk 5 statistics:
#   MSE:         mean=2.411396e+07, min=1.520533e+06, max=5.480724e+08
#   L_inf:       mean=1.517651e+04, min=4.715613e+03, max=7.799070e+04
#   Relative L2: mean=6.157451e-04 (0.0616%), min=1.852152e-04 (0.0185%), max=3.627855e-03 (0.3628%)
#
# Evaluating chunk 6: 1000 instances from Teacher_operator_T_C_2_phi_adam_fd_chunks\Teacher_operator_T_C_2_phi_adam_fd_chunk_6.pth
# Prediction done in 6.52s
# Chunk 6 statistics:
#   MSE:         mean=1.062912e+08, min=1.670810e+06, max=7.233926e+09
#   L_inf:       mean=3.515319e+04, min=5.928932e+03, max=6.814752e+05
#   Relative L2: mean=9.674237e-04 (0.0967%), min=1.968077e-04 (0.0197%), max=1.165523e-02 (1.1655%)
#
# Evaluating chunk 7: 1000 instances from Teacher_operator_T_C_2_phi_adam_fd_chunks\Teacher_operator_T_C_2_phi_adam_fd_chunk_7.pth
# Prediction done in 6.49s
# Chunk 7 statistics:
#   MSE:         mean=3.105189e+07, min=1.337773e+06, max=2.276522e+09
#   L_inf:       mean=1.686510e+04, min=5.122457e+03, max=2.133610e+05
#   Relative L2: mean=6.378133e-04 (0.0638%), min=1.771077e-04 (0.0177%), max=6.236390e-03 (0.6236%)
#
# Evaluating chunk 8: 1000 instances from Teacher_operator_T_C_2_phi_adam_fd_chunks\Teacher_operator_T_C_2_phi_adam_fd_chunk_8.pth
# Prediction done in 7.06s
# Chunk 8 statistics:
#   MSE:         mean=6.400317e+07, min=2.470399e+06, max=2.019937e+10
#   L_inf:       mean=2.179806e+04, min=6.805371e+03, max=7.943299e+05
#   Relative L2: mean=8.323863e-04 (0.0832%), min=2.463107e-04 (0.0246%), max=1.950133e-02 (1.9501%)
#
# Evaluating chunk 9: 1000 instances from Teacher_operator_T_C_2_phi_adam_fd_chunks\Teacher_operator_T_C_2_phi_adam_fd_chunk_9.pth
# Prediction done in 6.45s
# Chunk 9 statistics:
#   MSE:         mean=9.879483e+07, min=2.279966e+06, max=3.845768e+09
#   L_inf:       mean=3.579961e+04, min=5.328713e+03, max=2.633152e+05
#   Relative L2: mean=1.195812e-03 (0.1196%), min=2.159424e-04 (0.0216%), max=8.331870e-03 (0.8332%)
#
# ========================================
# OVERALL STATISTICS FOR ALL INSTANCES
# ========================================
#   MSE:         mean=4.900686e+07, min=7.137790e+05, max=2.019937e+10
#   L_inf:       mean=2.078169e+04, min=4.062038e+03, max=7.943299e+05
#   Relative L2: mean=7.740109e-04 (0.0774%), min=1.157756e-04 (0.0116%), max=1.950133e-02 (1.9501%)
#
# Saved all predictions to phi_pred_10000_201_101.npy, shape (10000, 101, 201)
#
# Process finished with exit code 0



if __name__ == '__main__':
    main()
