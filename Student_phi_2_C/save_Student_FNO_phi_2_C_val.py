"""Export held-out Student FNO predictions for the phi -> C mapping.

The split matches Student_FNO_phi_2_C.py exactly: NumPy ``default_rng(seed=42)``
shuffles 10,000 samples, of which the final 1,000 form the validation set.
Predictions are restored to the original C scale before writing to disk.

Usage:
    python save_Student_FNO_phi_2_C_val.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from Student_FNO_phi_2_C import C_SCALE, FNO2d


def validation_indices(num_samples, train_ratio=0.9, seed=42):
    """Return original 10,000-sample indices for the deterministic val split."""
    permutation = np.random.default_rng(seed).permutation(num_samples)
    return permutation[int(num_samples * train_ratio):]


def predict_in_batches(model, phi_val, phi_mean, phi_std, device, batch_size):
    """Infer normalized C and restore it to the original physical scale."""
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(phi_val), batch_size):
            phi_batch = phi_val[start:start + batch_size]
            phi_batch = (phi_batch - phi_mean) / (phi_std + 1e-12)
            phi_tensor = torch.as_tensor(phi_batch, dtype=torch.float32,
                                         device=device).unsqueeze(1)
            c_normalized = model(phi_tensor).squeeze(1)
            predictions.append((c_normalized * C_SCALE).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description='Save Student FNO predictions on the fixed 1,000-sample validation set.')
    parser.add_argument('--phi-file', default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--model', default='Student_FNO_phi_2_C_1.pth')
    parser.add_argument('--norm-file', default='phi_normalization.npz')
    parser.add_argument('--output', default='C_Student_val_1000_101_201.npy')
    parser.add_argument('--indices-output', default='Student_FNO_phi_2_C_val_indices_seed42.npz')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--modes1', type=int, default=16)
    parser.add_argument('--modes2', type=int, default=16)
    parser.add_argument('--n-layers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # FFT phi is stored as (N, nx, ny); the Student FNO expects (N, ny, nx).
    phi = np.load(args.phi_file, mmap_mode='r').transpose(0, 2, 1)
    val_idx = validation_indices(phi.shape[0], seed=args.seed)
    phi_val = np.asarray(phi[val_idx], dtype=np.float32)
    print(f'Validation split: {len(val_idx)} / {phi.shape[0]} samples (seed={args.seed})')

    norm = np.load(args.norm_file)
    model = FNO2d(width=args.width, modes1=args.modes1, modes2=args.modes2,
                  n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f'Loaded model: {args.model}; C output scale={C_SCALE:.6e}')

    predictions = predict_in_batches(model, phi_val, norm['mean'], norm['std'],
                                     device, args.batch_size)
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
