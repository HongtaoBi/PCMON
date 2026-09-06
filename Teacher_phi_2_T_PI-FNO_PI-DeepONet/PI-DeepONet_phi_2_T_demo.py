"""Two-instance shared-parameter PI-DeepONet phi -> T demo.

The demo trains only from PDE and boundary-condition residuals.  It uses the
first two FFT phi fields (instance = 2; batch size = 1), and the same
ZCS/finite-difference physics loss as ``PI-DeepONet_phi_2_T.py``.  Each
optimization step receives one randomly selected field.  ``T_true`` is used
only after training to compare the two physics-trained predictions with their
numerical PDE solutions.
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / 'PI-DeepONet_phi_2_T.py'
spec = importlib.util.spec_from_file_location('pi_deeponet_core', CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def parse_args():
    parser = argparse.ArgumentParser(description='Two-instance PI-DeepONet phi -> T demo.')
    parser.add_argument('--phi_file', type=Path, default=HERE / '1_FFT_phi_10000_201_101.npy')
    parser.add_argument('--true_T_file', type=Path, default=HERE / 'T_true_10000_101_201.npy')
    parser.add_argument('--num_steps', type=int, default=40000)
    parser.add_argument('--num_points', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--phi_scale', type=float, default=30550000.0)
    parser.add_argument('--checkpoint', type=Path, default=HERE / 'PI_DeepONet_phi_2_T_demo_best.pth')
    parser.add_argument('--prediction_file', type=Path,
                        default=HERE / 'PI_DeepONet_phi_2_T_demo_pred_2_101_201.npy')
    parser.add_argument('--figure_file', type=Path,
                        default=HERE / 'PI_DeepONet_phi_2_T_demo_comparison.svg')
    parser.add_argument('--no_show', action='store_true')
    return parser.parse_args()


def make_grid_points(nx, ny, L, H):
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    return torch.as_tensor(
        np.stack((X.ravel(), Y.ravel()), axis=-1), dtype=core.DTYPE, device=core.DEVICE,
    ).unsqueeze(0)


def predict_two(model, phi_demo, grid_points, phi_scale):
    model.eval()
    with torch.no_grad():
        _, phi_normalized = core.load_phi_batch(phi_demo, np.arange(2), phi_scale)
        branch_features = model.encode_branch(phi_normalized)
        prediction = model.forward_from_branch(
            branch_features, grid_points.expand(2, -1, -1),
        ).cpu().numpy()
    return prediction.reshape(2, phi_demo.shape[1], phi_demo.shape[2])


def save_comparison_figure(true_temperature, prediction, figure_file, show):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), squeeze=False)
    for row in range(2):
        reference = true_temperature[row]
        predicted = prediction[row]
        error = np.abs(predicted - reference)
        vmax = max(float(reference.max()), float(predicted.max()))
        fields = [
            (reference, 'Reference T', 'hot', 600.0, vmax),
            (predicted, 'PI-DeepONet T', 'hot', 600.0, vmax),
            (error, 'Absolute error', 'cividis', 0.0, max(float(error.max()), 1e-12)),
        ]
        for col, (data, title, cmap, vmin, field_vmax) in enumerate(fields):
            ax = axes[row, col]
            image = ax.imshow(data, extent=[0.0, 2.0, 0.0, 1.0], origin='lower',
                              cmap=cmap, vmin=vmin, vmax=field_vmax, aspect='auto')
            ax.set_box_aspect(1.0)
            ax.set_title(f'Instance {row}: {title}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.2)
            fig.colorbar(image, ax=ax, orientation='horizontal', fraction=0.045, pad=0.06)
    fig.tight_layout()
    figure_file.parent.mkdir(parents=True, exist_ok=True)
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
    if core.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    phi_demo = np.load(args.phi_file, mmap_mode='r')[:2].transpose(0, 2, 1)
    true_temperature = np.asarray(np.load(args.true_T_file, mmap_mode='r')[:2], dtype=np.float64)
    if phi_demo.shape != true_temperature.shape:
        raise ValueError(f'phi shape {phi_demo.shape} != T_true shape {true_temperature.shape}')
    print(f'Device={core.DEVICE}; demo phi/T shape={phi_demo.shape}; instance=2; batch_size=1')

    L, H = core.physics_params['L'], core.physics_params['H']
    _, ny, nx = phi_demo.shape
    model = core.PIDeepONetPhiToT(
        nx=nx, ny=ny, latent_dim=64, branch_width=128, trunk_width=64,
        T_inlet=core.physics_params['T0'], T_scale=600.0,
    ).to(device=core.DEVICE, dtype=core.DTYPE)
    print(f'Demo shared parameter count: {sum(parameter.numel() for parameter in model.parameters()):,}')

    bases = core.make_base_tensors(core.sample_collocation_points(L, H, args.num_points, args.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(args.num_steps * ratio)) for ratio in (0.5, 0.9)],
        gamma=0.1,
    )
    rng = np.random.default_rng(args.seed)
    best_mean_loss = float('inf')
    for step in range(1, args.num_steps + 1):
        # Batch is deliberately one, but the active field changes every step.
        index = np.array([rng.integers(0, 2)], dtype=np.int64)
        optimizer.zero_grad(set_to_none=True)
        total_loss, components = core.compute_batch_loss(
            model, phi_demo, index, bases, L, H, args.phi_scale, (1.0, 1.0, 1.0, 1.0, 1.0),
        )
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = float(total_loss.detach().cpu())
        if step == 1 or step % args.log_interval == 0 or step == args.num_steps:
            values = [float(value.detach().cpu()) for value in components]
            # A batch loss belongs to only one field.  Evaluate both fields for
            # an honest shared-model convergence signal and checkpoint choice.
            per_instance_loss = []
            for instance_id in range(2):
                evaluation_loss, _ = core.compute_batch_loss(
                    model, phi_demo, np.array([instance_id], dtype=np.int64), bases,
                    L, H, args.phi_scale, (1.0, 1.0, 1.0, 1.0, 1.0),
                )
                per_instance_loss.append(float(evaluation_loss.detach().cpu()))
            mean_loss = float(np.mean(per_instance_loss))
            if mean_loss < best_mean_loss:
                best_mean_loss = mean_loss
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'model_state_dict': model.state_dict(), 'step': step,
                            'mean_loss': best_mean_loss}, args.checkpoint)
            print(f'[Demo] Step {step:5d}, LR={optimizer.param_groups[0]["lr"]:.2e}, '
                  f'Total={loss_value:.4e}, PDE={values[0]:.4e}, '
                  f'BC=[{values[1]:.3e}, {values[2]:.3e}, {values[3]:.3e}, {values[4]:.3e}] '
                  f'| Both=[{per_instance_loss[0]:.4e}, {per_instance_loss[1]:.4e}], '
                  f'Mean={mean_loss:.4e}')

    model.load_state_dict(torch.load(args.checkpoint, map_location=core.DEVICE)['model_state_dict'])
    prediction = predict_two(model, phi_demo, make_grid_points(nx, ny, L, H), args.phi_scale)
    rel_l2 = np.linalg.norm((prediction - true_temperature).reshape(2, -1), axis=1) / \
        np.maximum(np.linalg.norm(true_temperature.reshape(2, -1), axis=1), 1e-12)
    np.save(args.prediction_file, prediction)
    print(f'Saved two-instance predictions to: {args.prediction_file}')
    for index, error in enumerate(rel_l2):
        print(f'Instance {index}: relative L2 = {error * 100:.4f}%')
    save_comparison_figure(true_temperature, prediction, args.figure_file, show=not args.no_show)


if __name__ == '__main__':
    main()
