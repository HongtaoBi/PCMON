"""Export held-out Student FNO predictions for the (T, C) -> phi mapping.

The held-out split reproduces Student_FNO_T_C_2_phi.py exactly: the 10,000
samples are shuffled by NumPy ``default_rng(seed=42)`` and the final 1,000
form the validation set.  Stored predictions are converted from normalized
network output back to the original neutron-flux scale.

Usage:
    python save_Student_FNO_T_C_2_phi_val.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from Student_FNO_T_C_2_phi import FNO2d, PHI_SCALE


def validation_indices(num_samples, train_ratio=0.9, seed=42):
    """Return original full-dataset indices of the deterministic validation split."""
    permutation = np.random.default_rng(seed).permutation(num_samples)
    return permutation[int(num_samples * train_ratio):]


def predict_in_batches(model, T_val, C_val, norm, device, batch_size):
    """Normalize two input fields, infer phi, and restore its original scale."""
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(T_val), batch_size):
            stop = start + batch_size
            T_batch = (T_val[start:stop] - norm['T_mean']) / (norm['T_std'] + 1e-12)
            C_batch = (C_val[start:stop] - norm['C_mean']) / (norm['C_std'] + 1e-12)
            inputs = np.stack([T_batch, C_batch], axis=1)
            inputs_tensor = torch.as_tensor(inputs, dtype=torch.float32, device=device)
            phi_normalized = model(inputs_tensor).squeeze(1)
            predictions.append((phi_normalized * PHI_SCALE).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description='Save Student FNO predictions on the fixed 1,000-sample validation set.')
    parser.add_argument('--T-file', default='1_FFT_T_10000_101_201.npy')
    parser.add_argument('--C-file', default='1_FFT_C_10000_101_201.npy')
    parser.add_argument('--model', default='Student_FNO_T_C_2_phi_1.pth')
    parser.add_argument('--norm-file', default='T_C_2_phi_normalization.npz')
    parser.add_argument('--output', default='phi_Student_val_1000_101_201.npy')
    parser.add_argument('--indices-output', default='Student_FNO_T_C_2_phi_val_indices_seed42.npz')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--modes1', type=int, default=16)
    parser.add_argument('--modes2', type=int, default=16)
    parser.add_argument('--n-layers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # T and C are already stored in the Student FNO layout (N, ny, nx).
    T = np.load(args.T_file, mmap_mode='r')
    C = np.load(args.C_file, mmap_mode='r')
    if T.shape != C.shape:
        raise ValueError(f'T/C shape mismatch: {T.shape} versus {C.shape}')
    val_idx = validation_indices(T.shape[0], seed=args.seed)
    T_val = np.asarray(T[val_idx], dtype=np.float32)
    C_val = np.asarray(C[val_idx], dtype=np.float32)
    print(f'Validation split: {len(val_idx)} / {T.shape[0]} samples (seed={args.seed})')

    norm = np.load(args.norm_file)
    model = FNO2d(width=args.width, modes1=args.modes1, modes2=args.modes2,
                  n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f'Loaded model: {args.model}; phi output scale={PHI_SCALE:.6e}')

    predictions = predict_in_batches(model, T_val, C_val, norm, device, args.batch_size)
    output_path = Path(args.output)
    np.save(output_path, predictions.astype(np.float32, copy=False))
    np.savez(Path(args.indices_output),
             val_indices=val_idx,
             seed=args.seed,
             train_ratio=0.9,
             source_order='original_10000_sample_indices')

    print(f'Saved Student validation predictions: {output_path.resolve()} {predictions.shape}')
    print(f'Saved validation indices: {Path(args.indices_output).resolve()}')


if __name__ == '__main__':
    main()
