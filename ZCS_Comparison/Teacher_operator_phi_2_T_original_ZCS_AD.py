"""10-step resource benchmark: original aggregate ZCS + pure second-order AD.

This file deliberately reuses ``ParallelPINOs`` and the physics/data utilities
from ``Teacher_operator_phi_2_T.py``.  The only changed component is the
coordinate-derivative strategy:

* original ZCS: one scalar shift per coordinate direction and point weights a;
* first derivatives: AD(root, z) followed by AD(., a);
* second derivatives: repeat AD(root, z), then recover pointwise values by a.

The default configuration runs ten updates on 500 instances, matching a
ten-mini-batch one-epoch run of the existing teacher with batch size 50, and
ten Adam updates, with no checkpoints or prediction files written.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import Teacher_operator_phi_2_T as shift_zcs
from PDE_Solver import physics_params


METHOD = "original_aggregate_zcs_second_order_ad"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def scalar_shift_points(base, batch_size, z_x=None, z_y=None):
    """Apply the original ZCS scalar shifts to every point in every instance."""
    x = base.expand(batch_size, -1, -1)
    if z_x is None:
        z_x = torch.zeros(1, dtype=x.dtype, device=x.device)
    if z_y is None:
        z_y = torch.zeros(1, dtype=x.dtype, device=x.device)
    return x + torch.stack([
        z_x.expand(batch_size, x.shape[1]),
        z_y.expand(batch_size, x.shape[1]),
    ], dim=-1)


def original_first_derivatives(field, z_x, z_y):
    """Recover pointwise first derivatives using original ZCS weights a_ij."""
    a = torch.ones_like(field, requires_grad=True)
    root = torch.sum(a * field)
    root_x, root_y = torch.autograd.grad(
        root, (z_x, z_y), create_graph=True, retain_graph=True
    )
    field_x = torch.autograd.grad(root_x, a, create_graph=True, retain_graph=True)[0]
    field_y = torch.autograd.grad(root_y, a, create_graph=True, retain_graph=True)[0]
    return field_x, field_y, a, root_x, root_y


def pde_loss_original_ad(model, base, phi_input, batch_indices, profile=False):
    """Thermal PDE loss using original aggregate ZCS for first and second AD."""
    device = base.device
    z_x = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    z_y = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    batch_size = len(batch_indices)

    t0 = time.perf_counter()
    coords = scalar_shift_points(base, batch_size, z_x, z_y)
    temperature = (model(coords, batch_indices) * 600.0).squeeze(-1)
    sync(); forward_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    T_x, T_y, a, root_x, root_y = original_first_derivatives(temperature, z_x, z_y)
    sync(); first_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    root_xx = torch.autograd.grad(root_x, z_x, create_graph=True, retain_graph=True)[0]
    root_yy = torch.autograd.grad(root_y, z_y, create_graph=True, retain_graph=True)[0]
    T_xx = torch.autograd.grad(root_xx, a, create_graph=True, retain_graph=True)[0]
    T_yy = torch.autograd.grad(root_yy, a, create_graph=True, retain_graph=True)[0]
    sync(); second_ms = (time.perf_counter() - t0) * 1e3

    q = phi_input[:, :, 0]
    velocity = phi_input[:, :, 1]
    residual = (
        physics_params['rho_Cp'] * velocity * T_x
        - physics_params['k_th'] * (T_xx + T_yy)
        - q
    ) / 1e3
    loss = torch.mean(residual.square())
    if profile:
        return loss, {
            'pde_forward_ms': forward_ms,
            'pde_first_ad_ms': first_ms,
            'pde_second_ad_ms': second_ms,
        }
    return loss


def boundary_losses_original_zcs(model, precomputed, batch_indices):
    """Use aggregate ZCS to recover Neumann derivatives at boundary points."""
    batch_size = len(batch_indices)
    device = batch_indices.device
    dtype = torch.float64

    z_right_x = torch.zeros(1, dtype=dtype, device=device, requires_grad=True)
    z_bottom_y = torch.zeros(1, dtype=dtype, device=device, requires_grad=True)
    z_top_y = torch.zeros(1, dtype=dtype, device=device, requires_grad=True)

    left = scalar_shift_points(precomputed['left_base'], batch_size)
    right = scalar_shift_points(precomputed['right_base'], batch_size, z_right_x, None)
    bottom = scalar_shift_points(precomputed['bottom_base'], batch_size, None, z_bottom_y)
    top = scalar_shift_points(precomputed['top_base'], batch_size, None, z_top_y)
    all_boundary = torch.cat([left, right, bottom, top], dim=1)
    values = (model(all_boundary, batch_indices) * 600.0).squeeze(-1)

    left_T, right_T = values[:, :25], values[:, 25:50]
    bottom_T, top_T = values[:, 50:100], values[:, 100:150]
    left_loss = torch.mean((left_T - 600.0).square())

    def recover_normal(field, shift):
        weights = torch.ones_like(field, requires_grad=True)
        root = torch.sum(weights * field)
        root_d = torch.autograd.grad(root, shift, create_graph=True, retain_graph=True)[0]
        return torch.autograd.grad(root_d, weights, create_graph=True, retain_graph=True)[0]

    right_x = recover_normal(right_T, z_right_x)
    bottom_y = recover_normal(bottom_T, z_bottom_y)
    top_y = recover_normal(top_T, z_top_y)
    return (
        left_loss,
        torch.mean(right_x.square()),
        torch.mean(bottom_y.square()),
        torch.mean(top_y.square()),
    )


def one_step(model, optimizer, precomputed, batch_idx, use_fd_second=False):
    """One Adam update.  FD mode is used by the companion script."""
    batch_indices = torch.as_tensor(batch_idx, dtype=torch.long, device='cuda')
    phi_input = precomputed['phi_input_pde'][batch_idx]
    optimizer.zero_grad(set_to_none=True)

    if use_fd_second:
        pde_loss, pde_times = pde_loss_original_fd(
            model, precomputed['pde_base'], phi_input, batch_indices, profile=True
        )
    else:
        pde_loss, pde_times = pde_loss_original_ad(
            model, precomputed['pde_base'], phi_input, batch_indices, profile=True
        )

    t0 = time.perf_counter()
    bc_losses = boundary_losses_original_zcs(model, precomputed, batch_indices)
    sync(); bc_ms = (time.perf_counter() - t0) * 1e3
    total = pde_loss + sum(bc_losses)

    t0 = time.perf_counter()
    total.backward()
    sync(); backward_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    optimizer.step()
    sync(); step_ms = (time.perf_counter() - t0) * 1e3
    return total.detach(), pde_loss.detach(), {
        **pde_times,
        'bc_ms': bc_ms,
        'backward_ms': backward_ms,
        'optimizer_ms': step_ms,
    }


def pde_loss_original_fd(model, base, phi_input, batch_indices, profile=False,
                         eps_x=1e-5, eps_y=1e-5):
    """Original aggregate ZCS for first AD, but central FD for second terms."""
    device = base.device
    z_x = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    z_y = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    batch_size = len(batch_indices)

    t0 = time.perf_counter()
    T0 = (model(scalar_shift_points(base, batch_size, z_x, z_y), batch_indices) * 600.0).squeeze(-1)
    sync(); forward_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    T_x, T_y, _, _, _ = original_first_derivatives(T0, z_x, z_y)
    sync(); first_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    xp = (model(scalar_shift_points(base, batch_size, z_x + eps_x, z_y), batch_indices) * 600.0).squeeze(-1)
    xm = (model(scalar_shift_points(base, batch_size, z_x - eps_x, z_y), batch_indices) * 600.0).squeeze(-1)
    yp = (model(scalar_shift_points(base, batch_size, z_x, z_y + eps_y), batch_indices) * 600.0).squeeze(-1)
    ym = (model(scalar_shift_points(base, batch_size, z_x, z_y - eps_y), batch_indices) * 600.0).squeeze(-1)
    T_xx = (xp - 2.0 * T0 + xm) / eps_x**2
    T_yy = (yp - 2.0 * T0 + ym) / eps_y**2
    sync(); second_ms = (time.perf_counter() - t0) * 1e3

    q, velocity = phi_input[:, :, 0], phi_input[:, :, 1]
    residual = (physics_params['rho_Cp'] * velocity * T_x - physics_params['k_th'] * (T_xx + T_yy) - q) / 1e3
    loss = torch.mean(residual.square())
    if profile:
        return loss, {
            'pde_forward_ms': forward_ms,
            'pde_first_ad_ms': first_ms,
            'pde_second_fd_ms': second_ms,
        }
    return loss


def parse_args(default_second_derivative='ad'):
    parser = argparse.ArgumentParser(description='Original-ZCS 10-step resource benchmark for phi -> T.')
    parser.add_argument('--num_instances', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_points', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phi_file', default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--second_derivative', choices=['ad', 'fd'], default=default_second_derivative)
    return parser.parse_args()


def main(default_second_derivative='ad'):
    args = parse_args(default_second_derivative)
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA so that GPU memory can be compared.')
    if args.num_instances < args.batch_size:
        raise ValueError('--num_instances must be at least --batch_size.')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_default_dtype(torch.float64)
    torch.cuda.empty_cache()

    L, H = physics_params['L'], physics_params['H']
    phi = np.load(args.phi_file)[:args.num_instances].transpose(0, 2, 1)
    x_all = np.linspace(0.0, L, 201)
    y_all = np.linspace(0.0, H, 101)
    print(f'[{METHOD if args.second_derivative == "ad" else "original_aggregate_zcs_second_order_fd"}]')
    print(f'Instances={args.num_instances}; batch={args.batch_size}; points={args.num_points}; measured steps={args.steps}')

    model = shift_zcs.ParallelPINOs(num_instances=args.num_instances, hidden_dim=64, H=H).double().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    precomputed = shift_zcs.precompute_all_data(phi, x_all, y_all, args.num_points, 0.0, L, 0.0, H)
    parameter_count = sum(p.numel() for p in model.parameters())
    static_memory_mib = torch.cuda.memory_allocated() / 2**20
    rng = np.random.default_rng(args.seed)

    # Warm-up is deliberately excluded from the reported timing and memory peak.
    for _ in range(args.warmup_steps):
        batch = rng.choice(args.num_instances, size=args.batch_size, replace=False)
        one_step(model, optimizer, precomputed, batch, use_fd_second=args.second_derivative == 'fd')
    sync(); torch.cuda.reset_peak_memory_stats()

    step_records = []
    wall_start = time.perf_counter()
    try:
        for step in range(1, args.steps + 1):
            batch = rng.choice(args.num_instances, size=args.batch_size, replace=False)
            sync(); t0 = time.perf_counter()
            total, pde, timings = one_step(model, optimizer, precomputed, batch, use_fd_second=args.second_derivative == 'fd')
            sync(); elapsed_ms = (time.perf_counter() - t0) * 1e3
            record = {'step': step, 'step_ms': elapsed_ms, 'total_loss': float(total), 'pde_loss': float(pde), **timings}
            step_records.append(record)
            print(f"Step {step:2d}/{args.steps}: total={record['total_loss']:.4e}, pde={record['pde_loss']:.4e}, step={elapsed_ms:.1f} ms")
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError('CUDA OOM: reduce --batch_size or --num_points for this baseline.') from exc

    sync()
    metric_keys = [key for key in step_records[0] if key.endswith('_ms')]
    summary = {
        'method': METHOD if args.second_derivative == 'ad' else 'original_aggregate_zcs_second_order_fd',
        'settings': vars(args),
        'parameter_count': parameter_count,
        'static_memory_mib_before_measurement': static_memory_mib,
        'peak_memory_mib': torch.cuda.max_memory_allocated() / 2**20,
        'total_measured_wall_s': time.perf_counter() - wall_start,
        'mean_ms': {key: float(np.mean([row[key] for row in step_records])) for key in metric_keys},
        'step_records': step_records,
    }
    out = Path(f"zcs_benchmark_{'original_ad' if args.second_derivative == 'ad' else 'original_fd'}_metrics.json")
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n=== Benchmark summary ===')
    print(f"Parameters: {parameter_count:,}; peak allocated: {summary['peak_memory_mib']:.1f} MiB")
    for key, value in summary['mean_ms'].items():
        print(f'{key}: {value:.2f} ms')
    print(f"Saved: {out.resolve()}")


if __name__ == '__main__':
    main()
