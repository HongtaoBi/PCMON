"""Four-A100 DDP shared physics-informed FNO baseline for phi -> T.

The PI-FNO architecture, structured-grid finite-difference PDE loss, and
hard inlet condition are imported unchanged from ``PI-FNO_phi_2_T.py``.
DDP partitions every global batch across GPUs and all-reduces gradients, so
the result remains one shared-parameter PI-FNO rather than four independent
models.
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
from torch.nn.parallel import DistributedDataParallel as DDP


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / 'PI-FNO_phi_2_T.py'
spec = importlib.util.spec_from_file_location('pi_fno_core', CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def parse_args():
    parser = argparse.ArgumentParser(description='Four-GPU DDP PI-FNO baseline for phi -> T.')
    parser.add_argument('--phi_file', type=str, default='1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--true_T_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='PI_FNO_phi_2_T_DDP_best.pth')
    parser.add_argument('--log_csv', type=str, default='PI_FNO_phi_2_T_DDP_training_log.csv')
    parser.add_argument('--val_predictions_file', type=str,
                        default='PI_FNO_phi_2_T_DDP_val_pred_1000_101_201.npy')
    parser.add_argument('--split_indices_file', type=str,
                        default='PI_FNO_phi_2_T_DDP_split_indices.npz')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_seed', type=int, default=42)
    # A larger global batch reduces the number of DDP gradient synchronizations
    # per epoch (9000 / 200 = 45 optimizer steps) while keeping one shared FNO.
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--global_batch_size', type=int, default=200)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--modes_y', type=int, default=16)
    parser.add_argument('--modes_x', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_interval_epochs', type=int, default=100)
    parser.add_argument('--val_interval_epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--phi_scale', type=float, default=30550000.0)
    parser.add_argument('--evaluate_only', action='store_true')
    return parser.parse_args()


def split_train_val_indices(num_instances, train_ratio, seed):
    permutation = np.random.default_rng(seed).permutation(num_instances)
    n_train = int(train_ratio * num_instances)
    return permutation[:n_train], permutation[n_train:]


def is_main(rank):
    return rank == 0


def all_reduce_weighted_sums(local_components, local_count, device):
    """Return sample-weighted global means of total/PDE/BC loss components."""
    local_sums = torch.as_tensor(local_components, dtype=torch.float64, device=device) * local_count
    count = torch.tensor(float(local_count), dtype=torch.float64, device=device)
    dist.all_reduce(local_sums, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return (local_sums / count).cpu().numpy()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('PI-FNO DDP requires CUDA GPUs.')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{local_rank}'))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    core.DEVICE = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log_file = None
    try:
        if args.global_batch_size < world_size:
            raise ValueError('global_batch_size must be at least the DDP world size')

        phi_mmap = np.load(args.phi_file, mmap_mode='r').transpose(0, 2, 1)
        if phi_mmap.shape[1:] != (101, 201):
            raise ValueError(f'Expected phi shaped (N, 101, 201), got {phi_mmap.shape}')
        train_indices, val_indices = split_train_val_indices(
            phi_mmap.shape[0], args.train_ratio, args.split_seed,
        )
        if len(train_indices) % args.global_batch_size != 0:
            raise ValueError('The training-set size must divide global_batch_size exactly')

        if is_main(rank):
            true_temperature = np.load(args.true_T_file, mmap_mode='r')
            if true_temperature.shape != phi_mmap.shape:
                raise ValueError(f'True T shape {true_temperature.shape} != phi shape {phi_mmap.shape}')
            Path(args.split_indices_file).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.split_indices_file, train_indices=train_indices, val_indices=val_indices)
            print(f'DDP world size={world_size}; train/validation={len(train_indices)} / {len(val_indices)}')
            print('Uploading immutable phi fields to each GPU...')
        else:
            true_temperature = None

        phi_device = torch.as_tensor(np.ascontiguousarray(phi_mmap), dtype=core.DTYPE, device=device)
        if is_main(rank):
            print(f'GPU-resident phi per rank: {phi_device.numel() * phi_device.element_size() / 2**30:.2f} GiB')

        operator = core.PIFNOPhiToT(
            nx=201, ny=101, width=args.width, modes_y=args.modes_y,
            modes_x=args.modes_x, n_layers=args.n_layers,
        ).to(device=device, dtype=core.DTYPE)
        model = DDP(operator, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, static_graph=True)
        if is_main(rank):
            print(f'Shared PI-FNO parameters: {sum(p.numel() for p in model.module.parameters()):,}')

        dx = core.physics_params['L'] / (phi_mmap.shape[2] - 1)
        dy = core.physics_params['H'] / (phi_mmap.shape[1] - 1)

        if args.evaluate_only:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            if is_main(rank):
                core.evaluate_relative_l2(
                    model.module, phi_mmap, true_temperature, val_indices,
                    args.global_batch_size, args.phi_scale, args.val_predictions_file,
                )
            dist.barrier()
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        milestones = sorted({max(1, int(args.num_epochs * ratio)) for ratio in (0.15, 0.50, 0.85)})
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        epoch_rng = np.random.default_rng(args.seed)
        best_validation, start_time = float('inf'), time.time()

        if is_main(rank):
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
            log_file = open(args.log_csv, 'w', newline='', encoding='utf-8')
            writer = csv.writer(log_file)
            writer.writerow(['epoch', 'lr', 'total', 'pde', 'left', 'right', 'bottom', 'top',
                             'val_rel_l2_mean', 'val_rel_l2_median', 'val_rel_l2_p95',
                             'val_rel_l2_max', 'elapsed_s'])
        else:
            writer = None

        for epoch in range(1, args.num_epochs + 1):
            model.train()
            epoch_indices = epoch_rng.permutation(train_indices)
            epoch_sums = np.zeros(6, dtype=np.float64)
            for start in range(0, len(train_indices), args.global_batch_size):
                global_indices = epoch_indices[start:start + args.global_batch_size]
                local_indices = np.array_split(global_indices, world_size)[rank]
                local_count = len(local_indices)
                index_tensor = torch.as_tensor(local_indices, dtype=torch.long, device=device)
                phi = phi_device.index_select(0, index_tensor).unsqueeze(1)
                phi_normalized = phi / args.phi_scale

                optimizer.zero_grad(set_to_none=True)
                total, components = core.physics_loss(model(phi_normalized), phi, dx, dy)
                # DDP averages rank gradients.  This factor recovers the exact
                # sample-weighted gradient of the full, unevenly split batch.
                (total * (world_size * local_count / len(global_indices))).backward()
                optimizer.step()

                values = [float(total.detach().cpu())] + [float(item.detach().cpu()) for item in components]
                epoch_sums += all_reduce_weighted_sums(values, local_count, device)

            averages = epoch_sums / (len(train_indices) // args.global_batch_size)
            scheduler.step()

            should_validate = (epoch == 1 or epoch % args.val_interval_epochs == 0 or epoch == args.num_epochs)
            if should_validate:
                dist.barrier()
                if is_main(rank):
                    metrics = core.evaluate_relative_l2(
                        model.module, phi_mmap, true_temperature, val_indices,
                        args.global_batch_size, args.phi_scale,
                    )
                    if metrics['mean'] < best_validation:
                        best_validation = metrics['mean']
                        torch.save({
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'best_val_rel_l2': best_validation,
                            'train_indices': train_indices,
                            'val_indices': val_indices,
                            'args': vars(args),
                        }, args.checkpoint)
                else:
                    metrics = None
                dist.barrier()
            else:
                metrics = None

            if is_main(rank) and (epoch == 1 or epoch % args.log_interval_epochs == 0 or epoch == args.num_epochs):
                elapsed, lr = time.time() - start_time, optimizer.param_groups[0]['lr']
                metric_values = ([metrics[key] for key in ('mean', 'median', 'p95', 'max')]
                                 if metrics is not None else [np.nan] * 4)
                writer.writerow([epoch, lr, *averages, *metric_values, elapsed])
                log_file.flush()
                print(f'[DDP Adam] Epoch {epoch:5d}/{args.num_epochs}, LR={lr:.2e}, '
                      f'Total={averages[0]:.4e}, PDE={averages[1]:.4e}, '
                      f'BC=[{averages[2]:.2e}, {averages[3]:.2e}, {averages[4]:.2e}, '
                      f'{averages[5]:.2e}], Time={elapsed:.1f}s')

        dist.barrier()
        if is_main(rank):
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            print(f'Training complete. Best validation relative L2: {best_validation * 100:.4f}%')
            core.evaluate_relative_l2(
                model.module, phi_mmap, true_temperature, val_indices,
                args.global_batch_size, args.phi_scale, args.val_predictions_file,
            )
        dist.barrier()
    finally:
        if log_file is not None:
            log_file.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
