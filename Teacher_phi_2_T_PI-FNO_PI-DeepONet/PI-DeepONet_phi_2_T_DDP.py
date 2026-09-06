"""Four-A100 DDP version of the shared physics-informed DeepONet phi -> T baseline.

Launch with ``torchrun --standalone --nproc_per_node=4``.  Every process owns
one replica of the model and a shard of each global batch.  DDP all-reduces
parameter gradients during ``backward()``, so all replicas remain exactly the
same shared-parameter PI-DeepONet after every optimizer step.
"""

import argparse
import csv
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / 'PI-DeepONet_phi_2_T.py'
spec = importlib.util.spec_from_file_location('pi_deeponet_core', CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def parse_args():
    parser = argparse.ArgumentParser(description='Four-GPU DDP PI-DeepONet baseline for phi -> T.')
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--true_T_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='PI_DeepONet_phi_2_T_DDP_best.pth')
    parser.add_argument('--log_csv', type=str, default='PI_DeepONet_phi_2_T_DDP_training_log.csv')
    parser.add_argument('--val_predictions_file', type=str,
                        default='PI_DeepONet_phi_2_T_DDP_val_pred_1000_101_201.npy')
    parser.add_argument('--split_indices_file', type=str,
                        default='PI_DeepONet_phi_2_T_DDP_split_indices.npz')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=40000)
    parser.add_argument('--global_batch_size', type=int, default=50)
    parser.add_argument('--num_points', type=int, default=500)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--branch_width', type=int, default=256)
    parser.add_argument('--trunk_width', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_interval_epochs', type=int, default=100)
    parser.add_argument('--val_interval_epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--phi_scale', type=float, default=30550000.0)
    parser.add_argument('--evaluate_only', action='store_true')
    return parser.parse_args()


def split_train_val_indices(num_instances, train_ratio, seed):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_instances)
    n_train = int(train_ratio * num_instances)
    return permutation[:n_train], permutation[n_train:]


def is_main(rank):
    return rank == 0


def synchronize_epoch_sums(values, device):
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


class PhysicsLossModule(nn.Module):
    """Expose the complete physics loss through DDP's ``forward`` method.

    Calling the loss through this wrapper is essential: DDP can then register
    and all-reduce every shared DeepONet parameter gradient during backward.
    """

    def __init__(self, operator, L, H, phi_scale, loss_weights):
        super().__init__()
        self.operator = operator
        self.L = L
        self.H = H
        self.phi_scale = phi_scale
        self.loss_weights = loss_weights

    def forward(self, phi, forcing, bases):
        batch_size = phi.shape[0]
        n_pde = bases['pde'].shape[1]
        branch_features = self.operator.encode_branch(phi / self.phi_scale)
        z_x_pde = torch.zeros((batch_size, n_pde), dtype=core.DTYPE, device=phi.device, requires_grad=True)
        z_y_pde = torch.zeros((batch_size, n_pde), dtype=core.DTYPE, device=phi.device, requires_grad=True)
        pde = core.pde_loss_zcs_fd(
            self.operator, branch_features, forcing, bases['pde'], z_x_pde, z_y_pde, self.L, self.H,
        )
        left, right, bottom, top = core.boundary_losses(self.operator, branch_features, bases)
        components = (pde, left, right, bottom, top)
        total = sum(weight * component for weight, component in zip(self.loss_weights, components))
        return total, components


def main():
    args = parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    # The imported single-GPU utilities use this module-level device value.
    core.DEVICE = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.global_batch_size < world_size:
        raise ValueError('global_batch_size must be at least the number of DDP processes')
    phi_mmap = np.load(args.phi_file, mmap_mode='r').transpose(0, 2, 1)
    if phi_mmap.shape[1:] != (101, 201):
        raise ValueError(f'Expected phi shaped (N, 101, 201), got {phi_mmap.shape}')
    train_indices, val_indices = split_train_val_indices(phi_mmap.shape[0], args.train_ratio, args.split_seed)
    if len(train_indices) % args.global_batch_size != 0:
        raise ValueError('For synchronized DDP steps, train size must divide global_batch_size exactly')
    if is_main(rank):
        true_temperature = np.load(args.true_T_file, mmap_mode='r')
        if true_temperature.shape != phi_mmap.shape:
            raise ValueError(f'True T shape {true_temperature.shape} != phi shape {phi_mmap.shape}')
        Path(args.split_indices_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.split_indices_file, train_indices=train_indices, val_indices=val_indices)
        print(f'DDP world size={world_size}; train/validation={len(train_indices)} / {len(val_indices)}')
        print('Uploading all phi fields to each GPU...')
    else:
        true_temperature = None
    # Each A100 keeps its own 1.51-GiB immutable copy; batches are gathered
    # locally on-device and never transferred from the host during training.
    phi_device = torch.as_tensor(np.ascontiguousarray(phi_mmap), dtype=core.DTYPE, device=device)
    if is_main(rank):
        print(f'GPU-resident phi per rank: {phi_device.numel() * phi_device.element_size() / 2**30:.2f} GiB')

    L, H = core.physics_params['L'], core.physics_params['H']
    operator = core.PIDeepONetPhiToT(
        nx=201, ny=101, latent_dim=args.latent_dim, branch_width=args.branch_width,
        trunk_width=args.trunk_width, T_inlet=core.physics_params['T0'], T_scale=600.0,
    ).to(device=device, dtype=core.DTYPE)
    loss_weights = (1.0, 1.0, 1.0, 1.0, 1.0)
    model = DDP(PhysicsLossModule(operator, L, H, args.phi_scale, loss_weights),
                device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    if is_main(rank):
        print(f'Shared PI-DeepONet parameters: {sum(p.numel() for p in model.module.operator.parameters()):,}')

    grid_x = np.linspace(0.0, L, 201)
    grid_y = np.linspace(0.0, H, 101)
    X, Y = np.meshgrid(grid_x, grid_y, indexing='xy')
    grid_points = torch.as_tensor(np.stack((X.ravel(), Y.ravel()), axis=-1), dtype=core.DTYPE, device=device).unsqueeze(0)
    if args.evaluate_only:
        # This checkpoint is created locally by this script and includes NumPy
        # split-index arrays; PyTorch >= 2.6 otherwise defaults to
        # weights_only=True and rejects those trusted metadata objects.
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.module.operator.load_state_dict(checkpoint['model_state_dict'])
        if is_main(rank):
            core.evaluate_relative_l2(model.module.operator, phi_device, true_temperature, grid_points, args.phi_scale,
                                      val_indices, args.global_batch_size, args.val_predictions_file)
        dist.barrier()
        dist.destroy_process_group()
        return

    bases = core.make_base_tensors(core.sample_collocation_points(L, H, args.num_points, args.seed))
    if is_main(rank):
        print("Precomputing q''' forcing on each GPU...")
    pde_forcing = core.precompute_pde_forcing(phi_device, bases['pde'], L, H)
    if is_main(rank):
        print(f'Cached PDE forcing per rank: {tuple(pde_forcing.shape)}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestones = [max(1, int(args.num_epochs * ratio)) for ratio in (0.15, 0.50, 0.85)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    epoch_rng = np.random.default_rng(args.seed)
    best_validation = float('inf')
    start_time = time.time()
    if is_main(rank):
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log_csv, 'w', newline='', encoding='utf-8')
        writer = csv.writer(log_file)
        writer.writerow(['epoch', 'lr', 'total', 'pde', 'left', 'right', 'bottom', 'top',
                         'val_rel_l2_mean', 'val_rel_l2_median', 'val_rel_l2_p95', 'val_rel_l2_max', 'elapsed_s'])
    else:
        log_file = writer = None

    try:
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            epoch_indices = epoch_rng.permutation(train_indices)
            local_sums = np.zeros(6, dtype=np.float64)
            local_seen = 0
            # Every rank executes every optimizer step.  A global group of 50
            # is split into 13/13/12/12 local shards; loss scaling makes the
            # DDP-averaged gradient equal to the global sample-weighted mean.
            for start in range(0, len(epoch_indices), args.global_batch_size):
                global_indices = epoch_indices[start:start + args.global_batch_size]
                local_indices = np.array_split(global_indices, world_size)[rank]
                local_count = len(local_indices)
                optimizer.zero_grad(set_to_none=True)
                phi_batch, _ = core.load_phi_batch(phi_device, local_indices, args.phi_scale)
                forcing_batch = pde_forcing.index_select(
                    0, torch.as_tensor(local_indices, dtype=torch.long, device=device),
                )
                local_loss, components = model(phi_batch, forcing_batch, bases)
                # DDP averages rank gradients.  This factor restores the
                # sample-weighted global batch average for uneven 13/12 shards.
                backward_loss = local_loss * (world_size * local_count / len(global_indices))
                backward_loss.backward()
                optimizer.step()
                values = [float(local_loss.detach().cpu())]
                values.extend(float(value.detach().cpu()) for value in components)
                local_sums += local_count * np.asarray(values)
                local_seen += local_count

            global_sums = synchronize_epoch_sums(np.r_[local_sums, local_seen], device)
            averages = global_sums[:6] / global_sums[6]
            scheduler.step()
            should_validate = (epoch == 1 or epoch % args.val_interval_epochs == 0 or epoch == args.num_epochs)
            if should_validate:
                dist.barrier()
                if is_main(rank):
                    _, metrics = core.evaluate_relative_l2(
                        model.module.operator, phi_device, true_temperature, grid_points, args.phi_scale,
                        val_indices, args.global_batch_size,
                    )
                    if metrics['mean'] < best_validation:
                        best_validation = metrics['mean']
                        torch.save({'model_state_dict': model.module.operator.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch,
                                    'best_val_rel_l2': best_validation, 'train_indices': train_indices,
                                    'val_indices': val_indices, 'args': vars(args)}, args.checkpoint)
                else:
                    metrics = None
                dist.barrier()
            else:
                metrics = None

            if is_main(rank) and (epoch == 1 or epoch % args.log_interval_epochs == 0 or epoch == args.num_epochs):
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]['lr']
                metric_values = ([metrics[key] for key in ('mean', 'median', 'p95', 'max')]
                                 if metrics is not None else [np.nan] * 4)
                writer.writerow([epoch, lr, *averages, *metric_values, elapsed])
                log_file.flush()
                print(f'[DDP Adam] Epoch {epoch:5d}/{args.num_epochs}, LR={lr:.2e}, Total={averages[0]:.4e}, '
                      f'PDE={averages[1]:.4e}, BC=[{averages[2]:.2e}, {averages[3]:.2e}, '
                      f'{averages[4]:.2e}, {averages[5]:.2e}], Time={elapsed:.1f}s')

        dist.barrier()
        if is_main(rank):
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.module.operator.load_state_dict(checkpoint['model_state_dict'])
            print(f'Training complete. Best validation relative L2: {best_validation * 100:.4f}%')
            core.evaluate_relative_l2(model.module.operator, phi_device, true_temperature, grid_points, args.phi_scale,
                                      val_indices, args.global_batch_size, args.val_predictions_file)
        dist.barrier()
    finally:
        if log_file is not None:
            log_file.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
