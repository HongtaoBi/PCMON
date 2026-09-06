"""Export the 1,000 held-out predictions of the trained DDP PI-DeepONet.

This script is inference-only.  It loads the trusted checkpoint created by
``PI-DeepONet_phi_2_T_DDP.py`` and reproduces the exact validation indices
stored during DDP training.  It does not initialise DDP or perform any
optimizer step.

Usage:
    python save_PI-DeepONet_phi_2_T_val.py
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / 'PI-DeepONet_phi_2_T.py'
spec = importlib.util.spec_from_file_location('pi_deeponet_core', CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def load_validation_indices(index_file, num_samples, train_ratio, split_seed):
    """Load saved DDP validation indices, or reproduce the deterministic split."""
    index_path = Path(index_file)
    if index_path.is_file():
        with np.load(index_path) as split:
            if 'val_indices' not in split:
                raise KeyError(f'{index_path} does not contain val_indices')
            indices = split['val_indices'].astype(int)
        print(f'Loaded {len(indices)} validation indices from: {index_path}')
        return indices

    permutation = np.random.default_rng(split_seed).permutation(num_samples)
    indices = permutation[int(num_samples * train_ratio):]
    print(f'Warning: {index_path} not found; reproduced split with seed={split_seed}')
    return indices


def main():
    parser = argparse.ArgumentParser(description='Export DDP PI-DeepONet phi->T validation predictions.')
    parser.add_argument('--phi-file', default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--true-T-file', default='T_true_10000_101_201.npy')
    parser.add_argument('--checkpoint', default='PI_DeepONet_phi_2_T_DDP_best.pth')
    parser.add_argument('--split-indices-file', default='PI_DeepONet_phi_2_T_DDP_split_indices.npz')
    parser.add_argument('--output', default='PI-DeepONet_phi_2_T_val_1000_101_201.npy')
    parser.add_argument('--indices-output', default='PI-DeepONet_phi_2_T_val_indices_seed42.npz')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--phi-scale', type=float, default=30550000.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    core.DEVICE = device
    print(f'Using device: {device}; dtype={core.DTYPE}')

    # The checkpoint was generated locally and deliberately carries trusted
    # NumPy metadata (the split indices and argument dictionary).
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get('args', {})
    latent_dim = int(checkpoint_args.get('latent_dim', 128))
    branch_width = int(checkpoint_args.get('branch_width', 256))
    trunk_width = int(checkpoint_args.get('trunk_width', 128))
    print(f'Checkpoint architecture: latent={latent_dim}, branch={branch_width}, trunk={trunk_width}')

    model = core.PIDeepONetPhiToT(
        nx=201, ny=101,
        latent_dim=latent_dim,
        branch_width=branch_width,
        trunk_width=trunk_width,
        T_inlet=core.physics_params['T0'],
        T_scale=600.0,
    ).to(device=device, dtype=core.DTYPE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    phi = np.load(args.phi_file, mmap_mode='r').transpose(0, 2, 1)
    true_temperature = np.load(args.true_T_file, mmap_mode='r')
    if phi.shape != true_temperature.shape or phi.shape[1:] != (101, 201):
        raise ValueError(f'Expected matching (N, 101, 201) phi/T arrays, got {phi.shape} and {true_temperature.shape}')
    val_indices = load_validation_indices(
        args.split_indices_file, len(phi), args.train_ratio, args.split_seed,
    )
    if len(val_indices) != 1000:
        raise ValueError(f'Expected 1,000 validation samples, found {len(val_indices)}')

    grid_x = np.linspace(0.0, core.physics_params['L'], 201)
    grid_y = np.linspace(0.0, core.physics_params['H'], 101)
    X, Y = np.meshgrid(grid_x, grid_y, indexing='xy')
    coordinates = torch.as_tensor(
        np.stack((X.ravel(), Y.ravel()), axis=-1), dtype=core.DTYPE, device=device,
    ).unsqueeze(0)

    predictions, errors = [], []
    with torch.no_grad():
        for start in range(0, len(val_indices), args.batch_size):
            batch_indices = val_indices[start:start + args.batch_size]
            phi_batch = np.asarray(phi[batch_indices], dtype=np.float64)
            phi_tensor = torch.as_tensor(phi_batch, dtype=core.DTYPE, device=device)
            branch_features = model.encode_branch(phi_tensor / args.phi_scale)
            prediction = model.forward_from_branch(
                branch_features, coordinates.expand(len(batch_indices), -1, -1),
            ).cpu().numpy().reshape(len(batch_indices), 101, 201)
            reference = np.asarray(true_temperature[batch_indices], dtype=np.float64)
            errors.append(
                np.linalg.norm((prediction - reference).reshape(len(batch_indices), -1), axis=1)
                / (np.linalg.norm(reference.reshape(len(batch_indices), -1), axis=1) + 1e-12)
            )
            predictions.append(prediction)

    predictions = np.concatenate(predictions, axis=0)
    errors = np.concatenate(errors, axis=0)
    np.save(args.output, predictions)
    np.savez(args.indices_output, val_indices=val_indices, train_ratio=args.train_ratio,
             split_seed=args.split_seed, source='PI_DeepONet_phi_2_T_DDP_split_indices.npz')
    print(f'Saved validation predictions: {Path(args.output).resolve()} {predictions.shape}')
    print(f'Saved validation indices: {Path(args.indices_output).resolve()}')
    print(f'Validation relative L2: mean={errors.mean() * 100:.4f}%, '
          f'median={np.median(errors) * 100:.4f}%, p95={np.percentile(errors, 95) * 100:.4f}%, '
          f'max={errors.max() * 100:.4f}%')


if __name__ == '__main__':
    main()
