"""Shared physics-informed FNO baseline for the 2D phi -> T mapping.

This is a label-free PI-FNO baseline for comparison with the P-PINOs
Teacher--Student workflow.  It uses the same energy equation, inlet condition,
9000/1000 split, Adam schedule, and validation metric as PI-DeepONet.  The
FNO predicts the complete structured temperature grid, so its PDE derivatives
are evaluated by second-order finite differences on that grid (rather than
pointwise ZCS queries used by the DeepONet baseline).
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PDE_Solver import physics_params


DTYPE = torch.float32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpectralConv2d(nn.Module):
    """Real 2D FFT convolution retaining low Fourier modes."""

    def __init__(self, in_channels, out_channels, modes_y, modes_x):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_y = modes_y
        self.modes_x = modes_x
        scale = 1.0 / (in_channels * out_channels)
        self.weights_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, 2, dtype=DTYPE)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, 2, dtype=DTYPE)
        )

    @staticmethod
    def complex_multiply(input_ft, weights):
        return torch.einsum('bixy,ioxy->boxy', input_ft, weights)

    def forward(self, x):
        batch_size, _, ny, nx = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')
        modes_y = min(self.modes_y, ny)
        modes_x = min(self.modes_x, nx // 2 + 1)
        out_ft = torch.zeros(
            batch_size, self.out_channels, ny, nx // 2 + 1,
            dtype=x_ft.dtype, device=x.device,
        )
        positive_weight = torch.view_as_complex(self.weights_pos[:, :, :modes_y, :modes_x].contiguous())
        negative_weight = torch.view_as_complex(self.weights_neg[:, :, :modes_y, :modes_x].contiguous())
        out_ft[:, :, :modes_y, :modes_x] = self.complex_multiply(
            x_ft[:, :, :modes_y, :modes_x], positive_weight,
        )
        out_ft[:, :, -modes_y:, :modes_x] = self.complex_multiply(
            x_ft[:, :, -modes_y:, :modes_x], negative_weight,
        )
        return torch.fft.irfft2(out_ft, s=(ny, nx), norm='ortho')


class FNOBlock(nn.Module):
    def __init__(self, width, modes_y, modes_x):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_y, modes_x)
        self.local = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.local(x))


class PIFNOPhiToT(nn.Module):
    """Grid-to-grid FNO with hard inlet temperature enforcement."""

    def __init__(self, nx=201, ny=101, width=64, modes_y=24, modes_x=24, n_layers=4):
        super().__init__()
        x = torch.linspace(0.0, physics_params['L'], nx, dtype=DTYPE)
        y = torch.linspace(0.0, physics_params['H'], ny, dtype=DTYPE)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        self.register_buffer('coordinates', torch.stack((X, Y), dim=0).unsqueeze(0))
        self.lift = nn.Conv2d(3, width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock(width, modes_y, modes_x) for _ in range(n_layers))
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1), nn.GELU(), nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, phi_normalized):
        batch_size = phi_normalized.shape[0]
        coords = self.coordinates.expand(batch_size, -1, -1, -1)
        features = torch.cat((phi_normalized, coords), dim=1)
        hidden = self.lift(features)
        for block in self.blocks:
            hidden = block(hidden)
        raw_temperature = self.project(hidden)
        x = coords[:, 0:1]
        envelope = 1.0 - torch.exp(-x / 0.05)
        return physics_params['T0'] + 600.0 * envelope * raw_temperature


def split_train_val(phi, temperature, train_ratio=0.9, seed=42):
    if phi.shape[0] != temperature.shape[0]:
        raise ValueError('phi and temperature must contain the same number of fields')
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(phi.shape[0])
    n_train = int(train_ratio * phi.shape[0])
    return permutation[:n_train], permutation[n_train:]


def load_phi_batch(phi_dataset, indices, phi_scale):
    phi = torch.as_tensor(np.asarray(phi_dataset[indices], dtype=np.float32), dtype=DTYPE, device=DEVICE)
    return phi.unsqueeze(1), (phi / phi_scale).unsqueeze(1)


def physics_loss(temperature, phi, dx, dy):
    """Structured-grid energy PDE residual and the three Neumann losses."""
    T = temperature[:, 0]
    source = physics_params['epsilon_f'] * physics_params['Sigma_f'] * phi[:, 0]
    y = torch.linspace(0.0, physics_params['H'], T.shape[1], dtype=DTYPE, device=DEVICE)
    velocity = physics_params['U_max'] * 4.0 * y * (physics_params['H'] - y) / physics_params['H'] ** 2

    # Interior centered differences: shape (B, ny-2, nx-2).
    T_center = T[:, 1:-1, 1:-1]
    T_x = (T[:, 1:-1, 2:] - T[:, 1:-1, :-2]) / (2.0 * dx)
    T_xx = (T[:, 1:-1, 2:] - 2.0 * T_center + T[:, 1:-1, :-2]) / (dx ** 2)
    T_yy = (T[:, 2:, 1:-1] - 2.0 * T_center + T[:, :-2, 1:-1]) / (dy ** 2)
    residual = (
        physics_params['rho_Cp'] * velocity[1:-1].view(1, -1, 1) * T_x
        - physics_params['k_th'] * (T_xx + T_yy)
        - source[:, 1:-1, 1:-1]
    ) / 1e3
    pde = torch.mean(residual.square())

    # The inlet is hard-enforced; retain it in diagnostics for transparency.
    left = torch.mean((T[:, :, 0] - physics_params['T0']).square())
    right = torch.mean(((T[:, :, -1] - T[:, :, -2]) / dx).square())
    bottom = torch.mean(((T[:, 1, :] - T[:, 0, :]) / dy).square())
    top = torch.mean(((T[:, -1, :] - T[:, -2, :]) / dy).square())
    total = pde + left + right + bottom + top
    return total, (pde, left, right, bottom, top)


def evaluate_relative_l2(model, phi_dataset, true_temperature, indices, batch_size, phi_scale,
                         prediction_path=None):
    model.eval()
    predictions, errors = ([] if prediction_path else None), []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            _, phi_normalized = load_phi_batch(phi_dataset, batch_indices, phi_scale)
            prediction = model(phi_normalized).squeeze(1).cpu().numpy()
            reference = np.asarray(true_temperature[batch_indices], dtype=np.float64)
            errors.append(np.linalg.norm((prediction - reference).reshape(len(batch_indices), -1), axis=1) /
                          np.maximum(np.linalg.norm(reference.reshape(len(batch_indices), -1), axis=1), 1e-12))
            if predictions is not None:
                predictions.append(prediction)
    errors = np.concatenate(errors)
    metrics = {key: float(value) for key, value in {
        'mean': errors.mean(), 'median': np.median(errors), 'p95': np.percentile(errors, 95), 'max': errors.max(),
    }.items()}
    print('Validation relative L2: ' + ', '.join(f'{key}={value * 100:.4f}%' for key, value in metrics.items()))
    if predictions is not None:
        np.save(prediction_path, np.concatenate(predictions, axis=0))
        print(f'Saved {len(indices)} validation predictions to: {prediction_path}')
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Shared PI-FNO baseline for phi -> T.')
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--true_T_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='PI_FNO_phi_2_T_best.pth')
    parser.add_argument('--log_csv', type=str, default='PI_FNO_phi_2_T_training_log.csv')
    parser.add_argument('--val_predictions_file', type=str, default='PI_FNO_phi_2_T_val_pred_1000_101_201.npy')
    parser.add_argument('--split_indices_file', type=str, default='PI_FNO_phi_2_T_split_indices.npz')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--modes_y', type=int, default=24)
    parser.add_argument('--modes_x', type=int, default=24)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_interval_epochs', type=int, default=1000)
    parser.add_argument('--val_interval_epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--phi_scale', type=float, default=30550000.0)
    parser.add_argument('--evaluate_only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    phi_dataset = np.load(args.phi_file, mmap_mode='r').transpose(0, 2, 1)
    true_temperature = np.load(args.true_T_file, mmap_mode='r')
    if phi_dataset.shape != true_temperature.shape:
        raise ValueError(f'phi shape {phi_dataset.shape} != T shape {true_temperature.shape}')
    train_indices, val_indices = split_train_val(phi_dataset, true_temperature, args.train_ratio, args.split_seed)
    np.savez(args.split_indices_file, train_indices=train_indices, val_indices=val_indices)
    print(f'Device={DEVICE}; train/validation = {len(train_indices)} / {len(val_indices)}')

    model = PIFNOPhiToT(width=args.width, modes_y=args.modes_y, modes_x=args.modes_x,
                         n_layers=args.n_layers).to(device=DEVICE, dtype=DTYPE)
    print(f'Shared PI-FNO parameters: {sum(parameter.numel() for parameter in model.parameters()):,}')
    if args.evaluate_only:
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE)['model_state_dict'])
        evaluate_relative_l2(model, phi_dataset, true_temperature, val_indices, args.batch_size,
                             args.phi_scale, args.val_predictions_file)
        return

    dx = physics_params['L'] / (phi_dataset.shape[2] - 1)
    dy = physics_params['H'] / (phi_dataset.shape[1] - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestones = sorted({max(1, int(args.num_epochs * ratio)) for ratio in (0.15, 0.50, 0.85)})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1,
    )
    best_validation = float('inf')
    rng, start_time = np.random.default_rng(args.seed), time.time()
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_csv, 'w', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(['epoch', 'lr', 'total', 'pde', 'left', 'right', 'bottom', 'top',
                         'val_rel_l2_mean', 'val_rel_l2_median', 'val_rel_l2_p95', 'val_rel_l2_max', 'elapsed_s'])
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            weighted_sums, seen = np.zeros(6), 0
            epoch_indices = rng.permutation(train_indices)
            for start in range(0, len(train_indices), args.batch_size):
                indices = epoch_indices[start:start + args.batch_size]
                phi, phi_normalized = load_phi_batch(phi_dataset, indices, args.phi_scale)
                optimizer.zero_grad(set_to_none=True)
                total, components = physics_loss(model(phi_normalized), phi, dx, dy)
                total.backward()
                optimizer.step()
                values = [float(total.detach().cpu())] + [float(item.detach().cpu()) for item in components]
                weighted_sums += len(indices) * np.asarray(values)
                seen += len(indices)
            averages = weighted_sums / seen
            scheduler.step()

            should_validate = epoch == 1 or epoch % args.val_interval_epochs == 0 or epoch == args.num_epochs
            metrics = evaluate_relative_l2(model, phi_dataset, true_temperature, val_indices, args.batch_size,
                                           args.phi_scale) if should_validate else None
            if metrics is not None and metrics['mean'] < best_validation:
                best_validation = metrics['mean']
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch, 'best_val_rel_l2': best_validation, 'train_indices': train_indices,
                            'val_indices': val_indices, 'args': vars(args)}, args.checkpoint)

            if epoch == 1 or epoch % args.log_interval_epochs == 0 or epoch == args.num_epochs:
                metric_values = [metrics[key] for key in ('mean', 'median', 'p95', 'max')] if metrics else [np.nan] * 4
                elapsed, lr = time.time() - start_time, optimizer.param_groups[0]['lr']
                writer.writerow([epoch, lr, *averages, *metric_values, elapsed])
                log_file.flush()
                print(f'[Adam] Epoch {epoch:5d}/{args.num_epochs}, LR={lr:.2e}, Total={averages[0]:.4e}, '
                      f'PDE={averages[1]:.4e}, BC=[{averages[2]:.2e}, {averages[3]:.2e}, '
                      f'{averages[4]:.2e}, {averages[5]:.2e}], Time={elapsed:.1f}s')

    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE)['model_state_dict'])
    evaluate_relative_l2(model, phi_dataset, true_temperature, val_indices, args.batch_size,
                         args.phi_scale, args.val_predictions_file)


if __name__ == '__main__':
    main()
