"""Two-instance, batch-one physics-only PI-FNO sanity check for phi -> T."""

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / 'PI-FNO_phi_2_T.py'
spec = importlib.util.spec_from_file_location('pi_fno_core', CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def parse_args():
    parser = argparse.ArgumentParser(description='Two-instance PI-FNO phi -> T demo.')
    parser.add_argument('--phi_file', type=Path, default=HERE / '1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--true_T_file', type=Path, default=HERE / 'T_true_10000_101_201.npy')
    parser.add_argument('--num_steps', type=int, default=40000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--phi_scale', type=float, default=30550000.0)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--modes_y', type=int, default=24)
    parser.add_argument('--modes_x', type=int, default=24)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--checkpoint', type=Path, default=HERE / 'PI_FNO_phi_2_T_demo_best.pth')
    parser.add_argument('--prediction_file', type=Path,
                        default=HERE / 'PI_FNO_phi_2_T_demo_pred_2_101_201.npy')
    parser.add_argument('--figure_file', type=Path,
                        default=HERE / 'PI_FNO_phi_2_T_demo_comparison.svg')
    parser.add_argument('--no_show', action='store_true')
    return parser.parse_args()


def compute_loss(model, phi_demo, indices, dx, dy, phi_scale):
    phi, phi_normalized = core.load_phi_batch(phi_demo, indices, phi_scale)
    return core.physics_loss(model(phi_normalized), phi, dx, dy)


def predict_two(model, phi_demo, phi_scale):
    model.eval()
    with torch.no_grad():
        _, phi_normalized = core.load_phi_batch(phi_demo, np.arange(2), phi_scale)
        return model(phi_normalized).squeeze(1).cpu().numpy()


def save_comparison_figure(reference, prediction, figure_file, show):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), squeeze=False)
    for row in range(2):
        error = np.abs(prediction[row] - reference[row])
        temperature_max = max(float(reference[row].max()), float(prediction[row].max()))
        fields = (
            (reference[row], 'Reference T', 'hot', 600.0, temperature_max),
            (prediction[row], 'PI-FNO T', 'hot', 600.0, temperature_max),
            (error, 'Absolute error', 'cividis', 0.0, max(float(error.max()), 1e-12)),
        )
        for col, (field, title, cmap, vmin, vmax) in enumerate(fields):
            axis = axes[row, col]
            image = axis.imshow(field, extent=(0.0, 2.0, 0.0, 1.0), origin='lower', cmap=cmap,
                                vmin=vmin, vmax=vmax, aspect='auto')
            axis.set_box_aspect(1.0)
            axis.set_title(f'Instance {row}: {title}', fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.2)
            fig.colorbar(image, ax=axis, orientation='horizontal', fraction=0.045, pad=0.06)
    fig.tight_layout()
    fig.savefig(figure_file, format='svg', bbox_inches='tight')
    print(f'Saved demo comparison figure to: {figure_file}')
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    phi_demo = np.load(args.phi_file, mmap_mode='r')[:2].transpose(0, 2, 1)
    true_temperature = np.asarray(np.load(args.true_T_file, mmap_mode='r')[:2], dtype=np.float64)
    if phi_demo.shape != true_temperature.shape:
        raise ValueError(f'phi shape {phi_demo.shape} != T_true shape {true_temperature.shape}')
    print(f'Device={core.DEVICE}; demo phi/T shape={phi_demo.shape}; instance=2; batch_size=1')

    _, ny, nx = phi_demo.shape
    dx = core.physics_params['L'] / (nx - 1)
    dy = core.physics_params['H'] / (ny - 1)
    model = core.PIFNOPhiToT(nx=nx, ny=ny, width=args.width, modes_y=args.modes_y,
                             modes_x=args.modes_x, n_layers=args.n_layers).to(core.DEVICE)
    print(f'Demo shared parameter count: {sum(parameter.numel() for parameter in model.parameters()):,}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestones = sorted({max(1, int(args.num_steps * ratio)) for ratio in (0.5, 0.85)})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1,
    )
    rng, best_mean_loss = np.random.default_rng(args.seed), float('inf')
    for step in range(1, args.num_steps + 1):
        index = np.array([rng.integers(0, 2)], dtype=np.int64)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, components = compute_loss(model, phi_demo, index, dx, dy, args.phi_scale)
        total.backward()
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_interval == 0 or step == args.num_steps:
            model.eval()
            with torch.no_grad():
                per_instance = [float(compute_loss(model, phi_demo, np.array([instance]), dx, dy,
                                                   args.phi_scale)[0].cpu())
                                for instance in range(2)]
            mean_loss = float(np.mean(per_instance))
            if mean_loss < best_mean_loss:
                best_mean_loss = mean_loss
                torch.save({'model_state_dict': model.state_dict(), 'step': step,
                            'mean_physics_loss': mean_loss}, args.checkpoint)
            values = [float(value.detach().cpu()) for value in components]
            print(f'[Demo] Step {step:5d}, LR={optimizer.param_groups[0]["lr"]:.2e}, '
                  f'Total={float(total.detach().cpu()):.4e}, PDE={values[0]:.4e}, '
                  f'BC=[{values[1]:.2e}, {values[2]:.2e}, {values[3]:.2e}, {values[4]:.2e}] '
                  f'| Both=[{per_instance[0]:.4e}, {per_instance[1]:.4e}], Mean={mean_loss:.4e}')

    model.load_state_dict(torch.load(args.checkpoint, map_location=core.DEVICE)['model_state_dict'])
    prediction = predict_two(model, phi_demo, args.phi_scale)
    relative_l2 = np.linalg.norm((prediction - true_temperature).reshape(2, -1), axis=1) / np.maximum(
        np.linalg.norm(true_temperature.reshape(2, -1), axis=1), 1e-12,
    )
    np.save(args.prediction_file, prediction)
    print(f'Saved two-instance predictions to: {args.prediction_file}')
    for index, error in enumerate(relative_l2):
        print(f'Instance {index}: relative L2 = {error * 100:.4f}%')
    save_comparison_figure(true_temperature, prediction, args.figure_file, show=not args.no_show)


if __name__ == '__main__':
    main()
