"""Save the independent-parameter DeepM&Mnet field evolution in one NPZ file.

This script deliberately reuses the solver and training settings from
``DeepM&Mnet_phi_T_C.py``.  It does not save a checkpoint, loss history,
numerical reference, final prediction file, or figures.  Its only artifact is
``DeepM_Mnet_phi_T_C_field_evolution.npz``, which is updated whenever a field
snapshot is requested.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / 'DeepM&Mnet_phi_T_C.py'
OUTPUT_FILE = BASE_DIR / 'DeepM_Mnet_phi_T_C_field_evolution.npz'

# Epoch 0 is the initialized coupled field.  The remaining entries are field
# states after completing the corresponding number of T -> C -> phi updates.
SNAPSHOT_EPOCHS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
                   1000, 2000, 5000, 10000, 15000, 20000)
NUM_EPOCHS = 20000
EARLY_STOP_LOSS_T = 1e-4
EARLY_STOP_LOSS_C = 1e-4
EARLY_STOP_LOSS_PHI = 1e-4
PATIENCE = 1000


def load_independent_solver_module():
    """Load the original script as a module without executing its main()."""
    module_spec = importlib.util.spec_from_file_location('deepmm_independent_source', SOURCE_FILE)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f'Cannot load source solver from: {SOURCE_FILE}')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def write_snapshots(epochs, temperature, concentration, flux):
    """Atomically replace the sole evolution artifact with all snapshots so far."""
    temporary_file = OUTPUT_FILE.with_suffix('.tmp.npz')
    np.savez_compressed(
        temporary_file,
        epochs=np.asarray(epochs, dtype=np.int32),
        T=np.stack(temperature, axis=0).astype(np.float32, copy=False),
        C=np.stack(concentration, axis=0).astype(np.float32, copy=False),
        phi=np.stack(flux, axis=0).astype(np.float32, copy=False),
    )
    temporary_file.replace(OUTPUT_FILE)


def main():
    source = load_independent_solver_module()
    device = source.device
    solver = source.Parallel_Parameters_DeepMMnetSolver_T_C_phi(
        student_paths=[
            str(BASE_DIR / 'Student_FNO_phi_2_T_1.pth'),
            str(BASE_DIR / 'Student_FNO_phi_2_C_1.pth'),
            str(BASE_DIR / 'Student_FNO_T_C_2_phi_1.pth'),
        ],
        nx=source.NX,
        ny=source.NY,
    ).to(device)

    # Keep the independent T -> C -> phi alternating updates unchanged.
    opt_T = torch.optim.Adam([
        {'params': solver.branch_T.parameters()}, {'params': [solver.b_T]}
    ], lr=1e-3)
    opt_C = torch.optim.Adam([
        {'params': solver.branch_C.parameters()}, {'params': [solver.b_C]}
    ], lr=1e-3)
    opt_phi = torch.optim.Adam([
        {'params': solver.branch_phi.parameters()}, {'params': [solver.b_phi]}
    ], lr=1e-3)

    snapshots, T_history, C_history, phi_history = [], [], [], []
    requested_epochs = set(SNAPSHOT_EPOCHS)
    best_loss = float('inf')
    epochs_no_improve = 0

    def capture(epoch):
        with torch.no_grad():
            T_field, C_field, phi_field = solver()
        snapshots.append(epoch)
        T_history.append(T_field.detach().cpu().numpy().squeeze())
        C_history.append(C_field.detach().cpu().numpy().squeeze())
        phi_history.append(phi_field.detach().cpu().numpy().squeeze())
        write_snapshots(snapshots, T_history, C_history, phi_history)
        print(f'Saved field snapshot at epoch {epoch} to: {OUTPUT_FILE.name}')

    def step_field(optimizer, field_index):
        optimizer.zero_grad()
        T_field, C_field, phi_field = solver()
        loss_T, loss_C, loss_phi, *_ = solver.consistency_loss(
            T_field, C_field, phi_field, w_T=1.0, w_C=1.0, w_phi=1.0
        )
        (loss_T, loss_C, loss_phi)[field_index].backward()
        optimizer.step()

    capture(0)
    for epoch in range(1, NUM_EPOCHS + 1):
        step_field(opt_T, 0)
        step_field(opt_C, 1)
        step_field(opt_phi, 2)

        # The original independent-parameter script evaluates the full
        # consistency loss after every alternating update cycle and applies
        # early stopping to those same weighted field losses.
        with torch.no_grad():
            T_field, C_field, phi_field = solver()
            loss_T, loss_C, loss_phi, *_ = solver.consistency_loss(
                T_field, C_field, phi_field, w_T=1.0, w_C=1.0, w_phi=1.0
            )
            total_loss = loss_T + loss_C + loss_phi

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch in requested_epochs:
            capture(epoch)

        if (loss_T.item() < EARLY_STOP_LOSS_T and
                loss_C.item() < EARLY_STOP_LOSS_C and
                loss_phi.item() < EARLY_STOP_LOSS_PHI and
                epochs_no_improve >= PATIENCE):
            if snapshots[-1] != epoch:
                capture(epoch)
            print(f'Early stopping at epoch {epoch}; final field state saved.')
            return

    # Keep the last state even if NUM_EPOCHS is later changed to a value that
    # is not listed in SNAPSHOT_EPOCHS.
    if snapshots[-1] != NUM_EPOCHS:
        capture(NUM_EPOCHS)


if __name__ == '__main__':
    main()
