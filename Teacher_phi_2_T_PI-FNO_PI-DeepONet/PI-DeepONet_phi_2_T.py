"""Shared-parameter physics-informed DeepONet baseline for phi -> T.

This script is deliberately a *global/shared-parameter* baseline for the
P-PINOs Teacher in ``Teacher_operator_phi_2_T.py``.  It uses exactly the same
steady energy equation, hard inlet condition, zero-coordinate-shift (ZCS)
differentiation, and boundary residuals.  The difference is architectural:
all 10,000 input fields share one Branch--Trunk DeepONet.

The model is trained without paired temperature labels.  Numerical PDE
solutions are loaded only by the optional evaluation routine.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PDE_Solver import physics_params


DTYPE = torch.float64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sync_cuda():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


class MLP(nn.Module):
    """Tanh MLP used by both DeepONet branches."""

    def __init__(self, widths):
        super().__init__()
        layers = []
        for in_features, out_features in zip(widths[:-1], widths[1:]):
            layer = nn.Linear(in_features, out_features)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


class PIDeepONetPhiToT(nn.Module):
    """Shared Branch--Trunk PI-DeepONet with a hard inlet temperature BC.

    The trunk uses physical coordinates and low-order transverse cosine
    features.  They are coordinate-only features, so local values of ``phi``
    remain available exclusively through the Branch; this is still a standard
    DeepONet.  The cosine terms mirror the high-frequency y representation in
    the P-PINOs teacher and alleviate raw-coordinate Tanh spectral bias.
    """

    def __init__(self, nx=201, ny=101, latent_dim=192, branch_width=384,
                 trunk_width=192, T_inlet=600.0, T_scale=600.0, sigma=0.05):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.T_inlet = T_inlet
        self.T_scale = T_scale
        self.sigma = sigma
        self.branch = MLP([nx * ny, branch_width, branch_width, latent_dim])
        self.trunk = MLP([5, trunk_width, trunk_width, latent_dim])
        self.bias = nn.Parameter(torch.zeros(1, dtype=DTYPE))

    def encode_branch(self, phi_normalized):
        """Encode a batch of normalized phi fields, shape (B, ny, nx)."""
        return self.branch(phi_normalized.reshape(phi_normalized.shape[0], -1))

    def forward_from_branch(self, branch_features, coordinates):
        """Predict physical temperature at coordinates of shape (B, N, 2)."""
        x = coordinates[..., 0:1]
        y = coordinates[..., 1:2]
        trunk_input = torch.cat((
            x, y,
            torch.cos(np.pi * y),
            torch.cos(2.0 * np.pi * y),
            torch.cos(4.0 * np.pi * y),
        ), dim=-1)
        trunk_features = self.trunk(trunk_input)
        raw_temperature = torch.einsum('bp,bnp->bn', branch_features, trunk_features)
        raw_temperature = raw_temperature / np.sqrt(branch_features.shape[-1]) + self.bias
        inlet_envelope = 1.0 - torch.exp(-x.squeeze(-1) / self.sigma)
        return self.T_inlet + self.T_scale * inlet_envelope * raw_temperature


def split_train_val(phi, T, train_ratio=0.9, seed=42):
    """Shuffle 10,000 instances into reproducible train/validation indices.

    This is index-based rather than returning ``phi[indices]`` and
    ``T[indices]`` arrays, because each full float64 field collection is about
    1.6 GB.  It implements the same permutation and 90/10 split as the Student
    script without materializing duplicate arrays in host memory.
    """
    if phi.shape[0] != T.shape[0]:
        raise ValueError(f'phi has {phi.shape[0]} instances but T has {T.shape[0]}')
    if not 0.0 < train_ratio < 1.0:
        raise ValueError('train_ratio must be between 0 and 1')
    rng = np.random.default_rng(seed)
    perm = rng.permutation(phi.shape[0])
    n_train = int(phi.shape[0] * train_ratio)
    return perm[:n_train], perm[n_train:]


def apply_zcs(base_points, z_x, z_y):
    """Apply per-instance ZCS shifts to fixed collocation coordinates."""
    return base_points + torch.stack((z_x, z_y), dim=-1)


def sample_collocation_points(L, H, num_points, seed):
    """Sample one fixed interior set and the four boundary point sets."""
    rng = np.random.default_rng(seed)
    interior = np.column_stack((
        rng.uniform(1e-8, L - 1e-8, num_points),
        rng.uniform(1e-8, H - 1e-8, num_points),
    ))
    left_y = np.linspace(0.0, H, 25)
    right_y = np.linspace(0.0, H, 25)
    bottom_x = np.linspace(0.01, L - 0.01, 50)
    top_x = np.linspace(0.01, L - 0.01, 50)
    return {
        'pde': interior,
        'left': np.column_stack((np.zeros_like(left_y), left_y)),
        'right': np.column_stack((np.full_like(right_y, L), right_y)),
        'bottom': np.column_stack((bottom_x, np.zeros_like(bottom_x))),
        'top': np.column_stack((top_x, np.full_like(top_x, H))),
    }


def make_base_tensors(point_sets):
    """Move fixed point sets to DEVICE once, retaining a leading batch axis."""
    return {
        name: torch.as_tensor(points, dtype=DTYPE, device=DEVICE).unsqueeze(0)
        for name, points in point_sets.items()
    }


def precompute_pde_forcing(phi_dataset, base_pde, L, H, batch_size=64):
    """Interpolate and cache q''' for all fields at the fixed PDE points once."""
    num_instances, num_points = phi_dataset.shape[0], base_pde.shape[1]
    forcing = torch.empty((num_instances, num_points), dtype=DTYPE, device=DEVICE)
    source_scale = physics_params['epsilon_f'] * physics_params['Sigma_f']
    for start in range(0, num_instances, batch_size):
        stop = min(start + batch_size, num_instances)
        if torch.is_tensor(phi_dataset):
            phi = phi_dataset[start:stop]
        else:
            phi = torch.as_tensor(np.asarray(phi_dataset[start:stop], dtype=np.float64),
                                  dtype=DTYPE, device=DEVICE)
        forcing[start:stop] = source_scale * interpolate_phi_at_points(phi, base_pde[0], L, H)
    return forcing


def interpolate_phi_at_points(phi_batch, points, L, H):
    """Bilinearly interpolate phi (B, ny, nx) at fixed physical points."""
    batch_size = phi_batch.shape[0]
    normalized_points = points.clone()
    normalized_points[:, 0] = 2.0 * normalized_points[:, 0] / L - 1.0
    normalized_points[:, 1] = 2.0 * normalized_points[:, 1] / H - 1.0
    grid = normalized_points.view(1, -1, 1, 2).expand(batch_size, -1, -1, -1)
    sampled = F.grid_sample(
        phi_batch.unsqueeze(1), grid, mode='bilinear', padding_mode='border', align_corners=True,
    )
    return sampled[:, 0, :, 0]


def pde_loss_zcs_fd(model, branch_features, phi_batch, base_pde, z_x, z_y,
                    L, H, eps_x=1e-5, eps_y=1e-5):
    """Energy-equation loss: ZCS first derivatives + FD second derivatives.

    This is the same differentiation strategy used in the existing P-PINOs
    Teacher, which avoids retaining a costly second-order autograd graph.
    """
    # The formal run supplies cached q''' of shape (B, N); the demo supplies
    # full phi fields and retains on-the-fly interpolation for simplicity.
    q = (phi_batch if phi_batch.ndim == 2 else
         physics_params['epsilon_f'] * physics_params['Sigma_f'] *
         interpolate_phi_at_points(phi_batch, base_pde[0], L, H))
    y = base_pde[0, :, 1].unsqueeze(0)
    u = physics_params['U_max'] * 4.0 * y * (H - y) / H ** 2

    coordinates = apply_zcs(base_pde, z_x, z_y)
    all_coordinates = torch.cat((
        coordinates,
        apply_zcs(base_pde, z_x + eps_x, z_y),
        apply_zcs(base_pde, z_x - eps_x, z_y),
        apply_zcs(base_pde, z_x, z_y + eps_y),
        apply_zcs(base_pde, z_x, z_y - eps_y),
    ), dim=1)
    all_temperatures = model.forward_from_branch(branch_features, all_coordinates)
    T, T_x_plus, T_x_minus, T_y_plus, T_y_minus = torch.chunk(all_temperatures, 5, dim=1)
    T_x, _ = torch.autograd.grad(T.sum(), (z_x, z_y), create_graph=True, retain_graph=True)
    T_xx = (T_x_plus - 2.0 * T + T_x_minus) / eps_x ** 2
    T_yy = (T_y_plus - 2.0 * T + T_y_minus) / eps_y ** 2

    residual = (
        physics_params['rho_Cp'] * u * T_x
        - physics_params['k_th'] * (T_xx + T_yy)
        - q
    ) / 1e3
    return torch.mean(residual.square())


def boundary_losses(model, branch_features, bases):
    """Evaluate the same inlet Dirichlet and three Neumann losses as P-PINOs."""
    batch_size = branch_features.shape[0]
    zeros_25 = torch.zeros((batch_size, 25), dtype=DTYPE, device=DEVICE)
    zeros_50 = torch.zeros((batch_size, 50), dtype=DTYPE, device=DEVICE)
    z_x_right = torch.zeros((batch_size, 25), dtype=DTYPE, device=DEVICE, requires_grad=True)
    z_y_bottom = torch.zeros((batch_size, 50), dtype=DTYPE, device=DEVICE, requires_grad=True)
    z_y_top = torch.zeros((batch_size, 50), dtype=DTYPE, device=DEVICE, requires_grad=True)

    left = apply_zcs(bases['left'], zeros_25, zeros_25)
    right = apply_zcs(bases['right'], z_x_right, zeros_25)
    bottom = apply_zcs(bases['bottom'], zeros_50, z_y_bottom)
    top = apply_zcs(bases['top'], zeros_50, z_y_top)
    all_bc = torch.cat((left, right, bottom, top), dim=1)
    temperature = model.forward_from_branch(branch_features, all_bc)

    left_T, right_T = temperature[:, :25], temperature[:, 25:50]
    bottom_T, top_T = temperature[:, 50:100], temperature[:, 100:150]
    left_loss = torch.mean((left_T - physics_params['T0']).square())
    right_loss = torch.mean(torch.autograd.grad(right_T.sum(), z_x_right, create_graph=True)[0].square())
    bottom_loss = torch.mean(torch.autograd.grad(bottom_T.sum(), z_y_bottom, create_graph=True)[0].square())
    top_loss = torch.mean(torch.autograd.grad(top_T.sum(), z_y_top, create_graph=True)[0].square())
    return left_loss, right_loss, bottom_loss, top_loss


def load_phi_batch(phi_dataset, indices, phi_scale):
    """Load one batch from a memory-mapped (N, ny, nx) phi dataset."""
    if torch.is_tensor(phi_dataset):
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=DEVICE)
        phi = phi_dataset.index_select(0, index_tensor)
    else:
        phi = np.asarray(phi_dataset[indices], dtype=np.float64)
        phi = torch.as_tensor(phi, dtype=DTYPE, device=DEVICE)
    return phi, phi / phi_scale


def compute_batch_loss(model, phi_dataset, indices, bases, L, H, phi_scale, loss_weights,
                       pde_forcing=None):
    """Return total and component losses for a randomly sampled instance batch."""
    phi, phi_normalized = load_phi_batch(phi_dataset, indices, phi_scale)
    batch_size = phi.shape[0]
    n_pde = bases['pde'].shape[1]
    branch_features = model.encode_branch(phi_normalized)

    z_x_pde = torch.zeros((batch_size, n_pde), dtype=DTYPE, device=DEVICE, requires_grad=True)
    z_y_pde = torch.zeros((batch_size, n_pde), dtype=DTYPE, device=DEVICE, requires_grad=True)
    forcing_batch = phi if pde_forcing is None else pde_forcing.index_select(
        0, torch.as_tensor(indices, dtype=torch.long, device=DEVICE),
    )
    pde_loss = pde_loss_zcs_fd(
        model, branch_features, forcing_batch, bases['pde'], z_x_pde, z_y_pde, L, H,
    )
    left_loss, right_loss, bottom_loss, top_loss = boundary_losses(model, branch_features, bases)
    components = (pde_loss, left_loss, right_loss, bottom_loss, top_loss)
    total = sum(weight * component for weight, component in zip(loss_weights, components))
    return total, components


def evaluate_relative_l2(model, phi_dataset, true_temperature, grid_points,
                         phi_scale, indices, batch_size=8, prediction_path=None):
    """Evaluate a specified split and optionally save its predicted fields."""
    model.eval()
    errors = []
    predictions = [] if prediction_path is not None else None
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            batch_indices = indices[start:stop]
            _, phi_normalized = load_phi_batch(phi_dataset, batch_indices, phi_scale)
            branch_features = model.encode_branch(phi_normalized)
            coordinates = grid_points.expand(stop - start, -1, -1)
            prediction = model.forward_from_branch(branch_features, coordinates).cpu().numpy()
            prediction = prediction.reshape(stop - start, *true_temperature.shape[1:])
            reference = np.asarray(true_temperature[batch_indices], dtype=np.float64)
            errors.append(
                np.linalg.norm((prediction - reference).reshape(stop - start, -1), axis=1)
                / np.maximum(np.linalg.norm(reference.reshape(stop - start, -1), axis=1), 1e-12)
            )
            if predictions is not None:
                predictions.append(prediction)
    errors = np.concatenate(errors)
    metrics = {
        'mean': float(errors.mean()),
        'median': float(np.median(errors)),
        'p95': float(np.percentile(errors, 95)),
        'max': float(errors.max()),
    }
    print('Validation relative L2: '
        f'mean={errors.mean() * 100:.4f}%, median={np.median(errors) * 100:.4f}%, '
        f'p95={np.percentile(errors, 95) * 100:.4f}%, max={errors.max() * 100:.4f}%')
    if predictions is not None:
        prediction_path = Path(prediction_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(prediction_path, np.concatenate(predictions, axis=0))
        print(f'Saved {len(indices)} validation predictions to: {prediction_path}')
    return errors, metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Shared PI-DeepONet baseline for phi -> T.')
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--true_T_file', type=str, default=None,
                        help='Optional numerical T reference used only after training.')
    parser.add_argument('--checkpoint', type=str, default='PI_DeepONet_phi_2_T_best.pth')
    parser.add_argument('--log_csv', type=str, default='PI_DeepONet_phi_2_T_training_log.csv')
    parser.add_argument('--val_predictions_file', type=str,
                        default='PI_DeepONet_phi_2_T_val_pred_1000_101_201.npy')
    parser.add_argument('--split_indices_file', type=str,
                        default='PI_DeepONet_phi_2_T_split_indices.npz')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=40000,
                        help='Full passes over all training instances.')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Instances per shared-parameter mini-batch.')
    parser.add_argument('--num_points', type=int, default=1000)
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Shared Branch--Trunk latent dimension.')
    parser.add_argument('--branch_width', type=int, default=512,
                        help='Hidden width of the phi-field Branch MLP.')
    parser.add_argument('--trunk_width', type=int, default=256,
                        help='Hidden width of the coordinate/Fourier Trunk MLP.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_interval_epochs', type=int, default=1000)
    parser.add_argument('--val_interval_epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--phi_scale', type=float, default=30550000.0)
    parser.add_argument('--evaluate_only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if DEVICE.type != 'cuda':
        print('Warning: CUDA is unavailable; float64 ZCS training on CPU will be very slow.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    L, H = physics_params['L'], physics_params['H']
    nx, ny = 201, 101
    phi_dataset = np.load(args.phi_file, mmap_mode='r').transpose(0, 2, 1)
    if phi_dataset.shape[1:] != (ny, nx):
        raise ValueError(f'Expected phi data shaped (N, {nx}, {ny}) before transpose; got {phi_dataset.shape}')
    print(f'Device={DEVICE}; loaded {phi_dataset.shape[0]} phi fields as {phi_dataset.shape}')
    if args.true_T_file is None:
        raise ValueError('--true_T_file is required to construct and evaluate the 90/10 split')
    true_temperature = np.load(args.true_T_file, mmap_mode='r')
    if true_temperature.shape != phi_dataset.shape:
        raise ValueError(f'True T shape {true_temperature.shape} != phi shape {phi_dataset.shape}')
    train_indices, val_indices = split_train_val(
        phi_dataset, true_temperature, train_ratio=args.train_ratio, seed=args.split_seed,
    )
    print(f'Train/validation split: {len(train_indices)} / {len(val_indices)} '
          f'(seed={args.split_seed})')
    split_indices_path = Path(args.split_indices_file)
    split_indices_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(split_indices_path, train_indices=train_indices, val_indices=val_indices)
    print(f'Saved split indices to: {split_indices_path}')

    # The complete float64 phi collection is only about 1.62 GB.  Keep it on
    # the A100 so every epoch performs GPU-only batch gathering instead of
    # repeatedly reading/transferring complete fields from the host.
    print('Uploading all phi fields to GPU...')
    phi_device = torch.as_tensor(
        np.ascontiguousarray(phi_dataset), dtype=DTYPE, device=DEVICE,
    )
    print(f'GPU-resident phi: {tuple(phi_device.shape)} '
          f'({phi_device.numel() * phi_device.element_size() / 2**30:.2f} GiB)')

    model = PIDeepONetPhiToT(
        nx=nx, ny=ny, latent_dim=args.latent_dim,
        branch_width=args.branch_width, trunk_width=args.trunk_width,
        T_inlet=physics_params['T0'], T_scale=600.0,
    ).to(device=DEVICE, dtype=DTYPE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f'Shared PI-DeepONet parameters: {parameter_count:,}')

    grid_x, grid_y = np.linspace(0.0, L, nx), np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(grid_x, grid_y, indexing='xy')
    grid_points = torch.as_tensor(np.stack((X.ravel(), Y.ravel()), axis=-1), dtype=DTYPE, device=DEVICE)
    grid_points = grid_points.unsqueeze(0)

    if args.evaluate_only:
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE)['model_state_dict'])
        evaluate_relative_l2(
            model, phi_device, true_temperature, grid_points, args.phi_scale,
            val_indices, args.batch_size, args.val_predictions_file,
        )
        return

    bases = make_base_tensors(sample_collocation_points(L, H, args.num_points, args.seed))
    print("Precomputing q''' forcing at fixed PDE collocation points...")
    pde_forcing = precompute_pde_forcing(phi_device, bases['pde'], L, H)
    print(f'Cached PDE forcing: {tuple(pde_forcing.shape)} on {DEVICE}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(args.num_epochs * ratio)) for ratio in (0.50, 0.9)],
        gamma=0.1,
    )
    loss_weights = (1.0, 1.0, 1.0, 1.0, 1.0)
    best_val_rel_l2 = float('inf')
    rng = np.random.default_rng(args.seed)
    start_time = time.time()
    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    with open(args.log_csv, 'w', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            'epoch', 'lr', 'total', 'pde', 'left', 'right', 'bottom', 'top',
            'val_rel_l2_mean', 'val_rel_l2_median', 'val_rel_l2_p95', 'val_rel_l2_max', 'elapsed_s',
        ])
        model.train()
        for epoch in range(1, args.num_epochs + 1):
            # Match the Teacher protocol: every training instance is visited
            # exactly once per epoch, but each mini-batch updates the shared
            # parameters immediately.
            epoch_indices = rng.permutation(train_indices)
            weighted_sums = np.zeros(6, dtype=np.float64)  # total + 5 components
            samples_seen = 0
            for start in range(0, len(epoch_indices), args.batch_size):
                indices = epoch_indices[start:start + args.batch_size]
                batch_count = len(indices)
                optimizer.zero_grad(set_to_none=True)
                total_loss, components = compute_batch_loss(
                    model, phi_device, indices, bases, L, H, args.phi_scale, loss_weights,
                    pde_forcing=pde_forcing,
                )
                total_loss.backward()
                optimizer.step()

                batch_values = [float(total_loss.detach().cpu())]
                batch_values.extend(float(item.detach().cpu()) for item in components)
                weighted_sums += batch_count * np.asarray(batch_values)
                samples_seen += batch_count

            # These are exact sample-weighted averages over all 9,000 training
            # fields for this epoch, not the final mini-batch loss.
            total_value, *component_values = weighted_sums / samples_seen
            scheduler.step()
            should_validate = (epoch == 1 or epoch % args.val_interval_epochs == 0
                               or epoch == args.num_epochs)
            if should_validate:
                _, val_metrics = evaluate_relative_l2(
                    model, phi_device, true_temperature, grid_points, args.phi_scale,
                    val_indices, args.batch_size,
                )
                model.train()
            else:
                val_metrics = None

            if val_metrics is not None and val_metrics['mean'] < best_val_rel_l2:
                best_val_rel_l2 = val_metrics['mean']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_rel_l2': best_val_rel_l2,
                    'train_indices': train_indices,
                    'val_indices': val_indices,
                    'args': vars(args),
                }, args.checkpoint)

            if (epoch == 1 or epoch % args.log_interval_epochs == 0
                    or epoch == args.num_epochs):
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]['lr']
                metric_values = ([val_metrics[key] for key in ('mean', 'median', 'p95', 'max')]
                                 if val_metrics is not None else [np.nan] * 4)
                writer.writerow([epoch, lr, total_value, *component_values, *metric_values, elapsed])
                log_file.flush()
                val_text = (' | Val RelL2: '
                            f"{val_metrics['mean'] * 100:.4f}%"
                            if val_metrics is not None else '')
                print(
                    f'[Adam] Epoch {epoch:4d}/{args.num_epochs}, LR: {lr:.2e}, Total: {total_value:.4e}, '
                    f'PDE: {component_values[0]:.4e}, Left: {component_values[1]:.4e}, '
                    f'Right: {component_values[2]:.4e}, Bottom: {component_values[3]:.4e}, '
                    f'Top: {component_values[4]:.4e}{val_text}, Time: {elapsed:.1f}s'
                )

    print(f'Training complete. Best validation relative L2: {best_val_rel_l2 * 100:.4f}%')
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE)['model_state_dict'])
    evaluate_relative_l2(
        model, phi_device, true_temperature, grid_points, args.phi_scale,
        val_indices, args.batch_size, args.val_predictions_file,
    )


if __name__ == '__main__':
    main()
