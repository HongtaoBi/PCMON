import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Windows + PyTorch/SciPy OpenMP runtime conflict workaround.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from PDE_Solver import (
    physics_params,
    solve_2d_coupled_phi_T_C,
)

print("CUDA available:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# -------------------- Physics / normalization constants --------------------
L = physics_params['L']                     # 2.0 m
H = physics_params['H']                     # 1.0 m
T_inlet = 600.0                             # K
C_inlet = 0.0
SIGMA = 0.05                                # boundary smoothing length
C_SCALE = 5e5                               # matches Student_phi_2_C training
PHI_SCALE = 2.5e7                           # matches Student_T_C_2_phi training

# Normalization constants computed on the 10k-instance training set.
# These MUST match the values used when the Student FNOs were trained.
PHI_MEAN = 5227623.902737082
PHI_STD = 5561946.727471371
T_MEAN = 789.214798266294
T_STD = 162.38291685777142
C_MEAN = 198724.75833025252
C_STD = 145147.57470465082

# Grid resolution (must match the Student FNO training resolution)
NX = 201
NY = 101

# -------------------- FNO building blocks (mirrors Student scripts) --------------------
class SpectralConv2d(nn.Module):
    """2D Fourier layer using real FFT (rfft2)."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )

    @staticmethod
    def compl_mul2d(input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            torch.view_as_complex(self.weights2)
        )
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super(FNOBlock, self).__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise_conv = nn.Conv2d(width, width, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.pointwise_conv(x)
        return self.activation(x1 + x2)


class Student_FNO_phi_2_T(nn.Module):
    """phi -> T.  Input (B, 1, ny, nx), output physical T (B, 1, ny, nx)."""
    def __init__(self, width=64, modes1=16, modes2=16, n_layers=4):
        super(Student_FNO_phi_2_T, self).__init__()
        self.width = width
        self.lift = nn.Conv2d(3, width, kernel_size=1)
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2) for _ in range(n_layers)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def _build_coords(self, B, ny, nx, device):
        x = torch.linspace(0.0, L, nx, device=device).view(1, 1, 1, nx)
        y = torch.linspace(0.0, H, ny, device=device).view(1, 1, ny, 1)
        return x.expand(B, 1, ny, nx), y.expand(B, 1, ny, nx)

    def forward(self, x):
        B, _, ny, nx = x.shape
        device = x.device
        x_coord, y_coord = self._build_coords(B, ny, nx, device)
        x = torch.cat([x, x_coord, y_coord], dim=1)
        x = self.lift(x)
        for block in self.fno_blocks:
            x = block(x)
        x = self.project(x)
        coeff = 1.0 - torch.exp(-x_coord / SIGMA)
        return coeff * x + T_inlet


class Student_FNO_phi_2_C(nn.Module):
    """phi -> C.  Input (B, 1, ny, nx), output normalized C (B, 1, ny, nx).
    Multiply by C_SCALE to recover physical C."""
    def __init__(self, width=64, modes1=16, modes2=16, n_layers=4):
        super(Student_FNO_phi_2_C, self).__init__()
        self.width = width
        self.lift = nn.Conv2d(3, width, kernel_size=1)
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2) for _ in range(n_layers)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def _build_coords(self, B, ny, nx, device):
        x = torch.linspace(0.0, L, nx, device=device).view(1, 1, 1, nx)
        y = torch.linspace(0.0, H, ny, device=device).view(1, 1, ny, 1)
        return x.expand(B, 1, ny, nx), y.expand(B, 1, ny, nx)

    def forward(self, x):
        B, _, ny, nx = x.shape
        device = x.device
        x_coord, y_coord = self._build_coords(B, ny, nx, device)
        x = torch.cat([x, x_coord, y_coord], dim=1)
        x = self.lift(x)
        for block in self.fno_blocks:
            x = block(x)
        x = self.project(x)
        coeff = 1.0 - torch.exp(-x_coord / SIGMA)
        return coeff * x + C_inlet / C_SCALE


class Student_FNO_T_C_2_phi(nn.Module):
    """(T, C) -> phi.  Input (B, 2, ny, nx), output normalized phi (B, 1, ny, nx).
    Multiply by PHI_SCALE to recover physical phi."""
    def __init__(self, width=64, modes1=16, modes2=16, n_layers=4):
        super(Student_FNO_T_C_2_phi, self).__init__()
        self.width = width
        self.lift = nn.Conv2d(4, width, kernel_size=1)
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2) for _ in range(n_layers)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def _build_coords(self, B, ny, nx, device):
        x = torch.linspace(0.0, L, nx, device=device).view(1, 1, 1, nx)
        y = torch.linspace(0.0, H, ny, device=device).view(1, 1, ny, 1)
        return x.expand(B, 1, ny, nx), y.expand(B, 1, ny, nx)

    def forward(self, x):
        B, _, ny, nx = x.shape
        device = x.device
        x_coord, y_coord = self._build_coords(B, ny, nx, device)
        x = torch.cat([x, x_coord, y_coord], dim=1)
        x = self.lift(x)
        for block in self.fno_blocks:
            x = block(x)
        x = self.project(x)
        coeff_x = 1.0 - torch.exp(-x_coord / SIGMA) - torch.exp(-(L - x_coord) / SIGMA)
        coeff_y = 1.0 - torch.exp(-y_coord / SIGMA) - torch.exp(-(H - y_coord) / SIGMA)
        coeff = coeff_x * coeff_y
        return coeff * x


# -------------------- DeepM&Mnet T-C-phi coupling solver --------------------
class Parallel_Parameters_DeepMMnetSolver_T_C_phi(nn.Module):
    """
    2D T-C-phi coupling with independent parameters.
    Three MLPs (one per field) map (x, y) to raw field values.
    Hard boundary conditions are added explicitly.
    Frozen Student FNOs enforce the three physical mappings.
    """
    def __init__(self, student_paths, nx=NX, ny=NY):
        super(Parallel_Parameters_DeepMMnetSolver_T_C_phi, self).__init__()
        self.device = device
        self.nx = nx
        self.ny = ny

        # Independent MLP branches: (x, y) -> scalar
        self.branch_T = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, 1)
        ).float().to(self.device)

        self.branch_C = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, 1)
        ).float().to(self.device)

        self.branch_phi = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, 1)
        ).float().to(self.device)

        # Zero-initialize last layer so initial guess = pure boundary profile.
        for branch in [self.branch_T, self.branch_C, self.branch_phi]:
            with torch.no_grad():
                branch[-1].weight.fill_(0.0)
                branch[-1].bias.fill_(0.0)

        # Per-field output scale + bias.  The scales are fixed to the typical
        # physical magnitude of each field; the MLPs only learn O(1) spatial
        # corrections.  This avoids the scale parameters themselves becoming
        # unstable during alternating optimization.
        self.register_buffer('w_T', torch.ones(1, device=self.device))
        self.b_T = nn.Parameter(torch.zeros(1, device=self.device))
        self.register_buffer('w_C', torch.ones(1, device=self.device) * 1e5)
        self.b_C = nn.Parameter(torch.zeros(1, device=self.device))
        self.register_buffer('w_phi', torch.ones(1, device=self.device) * 1e7)
        self.b_phi = nn.Parameter(torch.zeros(1, device=self.device))

        # Initialize biases to dataset means for a reasonable starting guess.
        # With the forward rule  field_raw = w_field * (branch + b_field),
        # setting b_field = mean_field / w_field gives an initial interior value
        # close to the training distribution.  The boundary mask still enforces
        # hard Dirichlet / vacuum BCs.
        with torch.no_grad():
            self.b_T.fill_((T_MEAN - T_inlet) / 1.0)
            self.b_C.fill_(C_MEAN / 1e5)
            self.b_phi.fill_(PHI_MEAN / 1e7)

        # Coordinate grid
        x_coord = torch.linspace(0.0, L, nx, device=self.device)
        y_coord = torch.linspace(0.0, H, ny, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_coord, y_coord, indexing='xy')
        self.register_buffer('x_coord_1d', x_coord)
        self.register_buffer('y_coord_1d', y_coord)
        self.register_buffer('xy_inputs', torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))
        self.register_buffer('x_grid', x_coord.view(1, 1, 1, nx))
        self.register_buffer('y_grid', y_coord.view(1, 1, ny, 1))

        # Frozen student operators
        self.student_phi_2_T = self._load_student(Student_FNO_phi_2_T, student_paths[0])
        self.student_phi_2_C = self._load_student(Student_FNO_phi_2_C, student_paths[1])
        self.student_T_C_2_phi = self._load_student(Student_FNO_T_C_2_phi, student_paths[2])

    def _load_student(self, model_cls, path):
        model = model_cls(width=64, modes1=16, modes2=16, n_layers=4).float().to(self.device)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Student model not found: {path}")
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def forward(self):
        """Returns physical T, C, phi as (1, 1, ny, nx) tensors."""
        # Per-point MLP evaluation.  The MLP learns an O(1) correction; the
        # fixed output scale w_field maps it to the correct physical magnitude.
        T_raw = self.w_T * (self.branch_T(self.xy_inputs).reshape(1, 1, self.ny, self.nx) + self.b_T)
        C_raw = self.w_C * (self.branch_C(self.xy_inputs).reshape(1, 1, self.ny, self.nx) + self.b_C)
        phi_raw = self.w_phi * (self.branch_phi(self.xy_inputs).reshape(1, 1, self.ny, self.nx) + self.b_phi)

        # Hard boundary conditions (same as the Student FNO forwards)
        coeff_T = 1.0 - torch.exp(-self.x_grid / SIGMA)
        T = coeff_T * T_raw + T_inlet

        coeff_C = 1.0 - torch.exp(-self.x_grid / SIGMA)
        C = coeff_C * C_raw + C_inlet

        coeff_x = 1.0 - torch.exp(-self.x_grid / SIGMA) - torch.exp(-(L - self.x_grid) / SIGMA)
        coeff_y = 1.0 - torch.exp(-self.y_grid / SIGMA) - torch.exp(-(H - self.y_grid) / SIGMA)
        coeff_phi = coeff_x * coeff_y
        phi = coeff_phi * phi_raw

        return T, C, phi

    def consistency_loss(self, T, C, phi, w_T=1.0, w_C=1.0, w_phi=1.0):
        """
        Independent-parameter self-consistency losses.
        Each field is compared against the frozen Student prediction from the other fields,
        with .detach() on the source fields to keep gradients disjoint.

        The residuals are normalized by the characteristic physical magnitude of each
        field so that the three losses are O(1) and can be balanced with the weights
        w_T/w_C/w_phi.
        """
        # Normalize source fields for the Student operators
        phi_norm = (phi - PHI_MEAN) / (PHI_STD + 1e-12)
        T_norm = (T - T_MEAN) / (T_STD + 1e-12)
        C_norm = (C - C_MEAN) / (C_STD + 1e-12)

        # Student predictions
        T_prime = self.student_phi_2_T(phi_norm.detach())                 # physical T
        C_prime_norm = self.student_phi_2_C(phi_norm.detach())            # normalized C
        phi_prime_norm = self.student_T_C_2_phi(
            torch.cat([T_norm.detach(), C_norm.detach()], dim=1)
        )                                                                 # normalized phi

        # Convert to physical units
        C_prime = C_prime_norm * C_SCALE
        phi_prime = phi_prime_norm * PHI_SCALE

        # Normalized self-consistency residuals
        loss_T_raw = torch.mean(((T - T_prime) / T_STD) ** 2)
        loss_C_raw = torch.mean(((C - C_prime) / C_STD) ** 2)
        loss_phi_raw = torch.mean(((phi - phi_prime) / PHI_SCALE) ** 2)

        # Weighted losses (used for optimization and early stopping)
        loss_T = w_T * loss_T_raw
        loss_C = w_C * loss_C_raw
        loss_phi = w_phi * loss_phi_raw

        return loss_T, loss_C, loss_phi, loss_T_raw, loss_C_raw, loss_phi_raw


# -------------------- Numerical reference & plotting utilities --------------------
def compute_numerical_references(nx=NX, ny=NY):
    """
    Compute the fully coupled numerical reference for phi-T-C
    using solve_2d_coupled_phi_T_C from PDE_Solver.py.

    Returns:
        T_ref, C_ref, phi_ref as 2D numpy arrays (ny, nx),
        plus S_ext and convergence history for diagnostics.
    """
    X, Y, phi_ref, T_ref, C_ref, S_ext, T_errors = solve_2d_coupled_phi_T_C(
        nx=nx, ny=ny, max_iter=100, tol=1e-6, relax=0.5, verbose=False
    )
    return T_ref, C_ref, phi_ref, S_ext, T_errors


def relative_l2(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)


def plot_fields(T, C, phi, T_ref=None, C_ref=None, phi_ref=None,
                save_prefix='DeepM_Mnet_phi_T_C'):
    """Visualize the coupled fields and optional numerical references."""
    x = np.linspace(0.0, L, NX)
    y = np.linspace(0.0, H, NY)
    X, Y = np.meshgrid(x, y)

    def rel_err_str(pred, ref):
        if ref is None:
            return ''
        return f' (rel L2 {100 * relative_l2(pred, ref):.4f}%)'

    fields = [
        (T, T_ref, 'T [K]', 'hot'),
        (C, C_ref, 'C', 'viridis'),
        (phi, phi_ref, r'$\phi$ [m$^{-2}$ s$^{-1}$]', 'plasma'),
    ]

    n_cols = 3 if any(r is not None for _, r, _, _ in fields) else 2
    fig, axes = plt.subplots(len(fields), n_cols, figsize=(6 * n_cols, 4 * len(fields)))
    if len(fields) == 1:
        axes = axes.reshape(1, -1)

    for row, (pred, ref, title, cmap) in enumerate(fields):
        im = axes[row, 0].imshow(pred, extent=[0, L, 0, H], origin='lower',
                                  cmap=cmap, aspect='auto', interpolation='bicubic')
        axes[row, 0].set_title(f'PCMON {title}{rel_err_str(pred, ref)}')
        axes[row, 0].set_xlabel('x [m]')
        axes[row, 0].set_ylabel('y [m]')
        fig.colorbar(im, ax=axes[row, 0])

        axes[row, 1].plot(pred[NY // 2, :], 'r--', label='PCMON centerline')
        if ref is not None:
            axes[row, 1].plot(ref[NY // 2, :], 'b-', label='Numerical centerline')
        axes[row, 1].set_title(f'{title} centerline (y = H/2)')
        axes[row, 1].set_xlabel('x index')
        axes[row, 1].legend()
        axes[row, 1].grid(True, linestyle='--', alpha=0.5)

        if n_cols == 3 and ref is not None:
            im2 = axes[row, 2].imshow(ref, extent=[0, L, 0, H], origin='lower',
                                       cmap=cmap, aspect='auto', interpolation='bicubic')
            axes[row, 2].set_title(f'Numerical {title}')
            axes[row, 2].set_xlabel('x [m]')
            axes[row, 2].set_ylabel('y [m]')
            fig.colorbar(im2, ax=axes[row, 2])

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_fields.png', dpi=200, bbox_inches='tight')
    plt.show()


def plot_residuals(T, C, phi, T_prime, C_prime, phi_prime,
                   save_prefix='DeepM_Mnet_phi_T_C'):
    """Plot the self-consistency residuals."""
    residuals = [
        (T - T_prime, 'T residual [K]', 'RdBu_r'),
        (C - C_prime, 'C residual', 'RdBu_r'),
        (phi - phi_prime, r'$\phi$ residual', 'RdBu_r'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (res, title, cmap) in zip(axes, residuals):
        im = ax.imshow(res, extent=[0, L, 0, H], origin='lower',
                       cmap=cmap, aspect='auto', interpolation='bicubic')
        ax.set_title(f'{title}\nmax |res| = {np.max(np.abs(res)):.4e}')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_residuals.png', dpi=200, bbox_inches='tight')
    plt.show()


# -------------------- Main training loop --------------------
def main():
    student_paths = [
        'Student_FNO_phi_2_T_1.pth',
        'Student_FNO_phi_2_C_1.pth',
        'Student_FNO_T_C_2_phi_1.pth',
    ]

    solver = Parallel_Parameters_DeepMMnetSolver_T_C_phi(
        student_paths=student_paths, nx=NX, ny=NY
    ).to(device)

    # ---- Independent-parameter alternating optimization ----
    # Three disjoint optimizers: each field's MLP + its own bias are updated
    # only by that field's self-consistency loss.  The output scales are fixed
    # buffers, so the MLPs learn O(1) corrections while the physical magnitude
    # is handled by the fixed scale.
    opt_T = torch.optim.Adam([
        {'params': solver.branch_T.parameters()},
        {'params': [solver.b_T]}
    ], lr=1e-3)
    opt_C = torch.optim.Adam([
        {'params': solver.branch_C.parameters()},
        {'params': [solver.b_C]}
    ], lr=1e-3)
    opt_phi = torch.optim.Adam([
        {'params': solver.branch_phi.parameters()},
        {'params': [solver.b_phi]}
    ], lr=1e-3)

    # ---- Loss weights (tune these to balance the three fields) ----
    # The single-field losses are already normalized by the characteristic
    # physical scale of each field (T_STD, C_STD, PHI_SCALE).  Default weights
    # are 1.0; increase a weight if that field is converging too slowly.
    w_T = 1.0
    w_C = 1.0
    w_phi = 1.0

    # Early stopping thresholds are applied to the WEIGHTED single-field losses.
    early_stop_loss_T = 1e-4
    early_stop_loss_C = 1e-4
    early_stop_loss_phi = 1e-4
    patience = 1000

    best_loss = float('inf')
    best_model_path = 'DeepM_Mnet_phi_T_C_model.pth'
    history = []

    num_epochs = 20000
    log_interval = 100
    epochs_no_improve = 0
    early_stopped = False

    print("=" * 60)
    print("DeepM&Mnet T-C-phi coupling starts...")
    print(f"Grid: {NX} x {NY}, device: {device}")
    print("Optimizer: independent alternating Adam (T -> C -> phi)")
    print(f"Loss weights: w_T={w_T}, w_C={w_C}, w_phi={w_phi}")
    print(f"Early stop thresholds (weighted): loss_T<{early_stop_loss_T:.0e}, "
          f"loss_C<{early_stop_loss_C:.0e}, loss_phi<{early_stop_loss_phi:.0e}, "
          f"patience={patience}")
    print("=" * 60)

    def _step_field(opt, field_idx):
        """Take one Adam step for a single field's parameters."""
        opt.zero_grad()
        T_pred, C_pred, phi_pred = solver()
        loss_T, loss_C, loss_phi, loss_T_raw, loss_C_raw, loss_phi_raw = \
            solver.consistency_loss(T_pred, C_pred, phi_pred, w_T=w_T, w_C=w_C, w_phi=w_phi)
        if field_idx == 0:
            loss_T.backward()
        elif field_idx == 1:
            loss_C.backward()
        else:
            loss_phi.backward()
        opt.step()
        return (loss_T.item(), loss_C.item(), loss_phi.item(),
                loss_T_raw.item(), loss_C_raw.item(), loss_phi_raw.item())

    start_time = time.time()
    for epoch in range(num_epochs):
        # Independent alternating updates
        _step_field(opt_T, 0)
        _step_field(opt_C, 1)
        _step_field(opt_phi, 2)

        with torch.no_grad():
            # Recompute for monitoring
            T_full, C_full, phi_full = solver()
            loss_T_full, loss_C_full, loss_phi_full, loss_T_raw_full, loss_C_raw_full, loss_phi_raw_full = \
                solver.consistency_loss(T_full, C_full, phi_full, w_T=w_T, w_C=w_C, w_phi=w_phi)
            loss_full = loss_T_full + loss_C_full + loss_phi_full

        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:6d} | "
                f"loss_T(w): {loss_T_full.item():.4e} [raw: {loss_T_raw_full.item():.4e}] | "
                f"loss_C(w): {loss_C_full.item():.4e} [raw: {loss_C_raw_full.item():.4e}] | "
                f"loss_phi(w): {loss_phi_full.item():.4e} [raw: {loss_phi_raw_full.item():.4e}] | "
                f"total: {loss_full.item():.4e} | Time: {elapsed:.1f}s"
            )
            history.append([epoch,
                            loss_T_full.item(), loss_C_full.item(), loss_phi_full.item(),
                            loss_T_raw_full.item(), loss_C_raw_full.item(), loss_phi_raw_full.item(),
                            loss_full.item(), elapsed])

        # Best model saving
        if loss_full.item() < best_loss:
            best_loss = loss_full.item()
            torch.save(solver.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping check (based on weighted single-field losses)
        if (loss_T_full.item() < early_stop_loss_T and
            loss_C_full.item() < early_stop_loss_C and
            loss_phi_full.item() < early_stop_loss_phi and
            epochs_no_improve >= patience):
            print(f"\nEarly stopping triggered at epoch {epoch}: "
                  f"all weighted field losses below thresholds and no improvement for {patience} epochs.")
            early_stopped = True
            break

    elapsed_total = time.time() - start_time
    if early_stopped:
        print(f"\nTraining stopped early after {epoch + 1} epochs.")
    else:
        print(f"\nTraining complete ({num_epochs} epochs).")
    print(f"Best total loss: {best_loss:.4e}")
    print(f"Total time: {elapsed_total:.1f}s")

    if history:
        np.save('DeepM_Mnet_phi_T_C_history.npy', np.array(history))
        print("Training history saved to DeepM_Mnet_phi_T_C_history.npy")

    # -------------------- Final evaluation --------------------
    solver.load_state_dict(torch.load(best_model_path))
    solver.eval()
    with torch.no_grad():
        T_pred, C_pred, phi_pred = solver()
        T_np = T_pred.detach().cpu().numpy().squeeze()
        C_np = C_pred.detach().cpu().numpy().squeeze()
        phi_np = phi_pred.detach().cpu().numpy().squeeze()

    # Self-consistency residuals
    phi_norm = (phi_pred - PHI_MEAN) / (PHI_STD + 1e-12)
    T_norm = (T_pred - T_MEAN) / (T_STD + 1e-12)
    C_norm = (C_pred - C_MEAN) / (C_STD + 1e-12)
    with torch.no_grad():
        T_prime = solver.student_phi_2_T(phi_norm).cpu().numpy().squeeze()
        C_prime = solver.student_phi_2_C(phi_norm).cpu().numpy().squeeze() * C_SCALE
        phi_prime = solver.student_T_C_2_phi(torch.cat([T_norm, C_norm], dim=1)).cpu().numpy().squeeze() * PHI_SCALE

    print("\nFinal self-consistency relative errors:")
    print(f"  T: {100 * relative_l2(T_np, T_prime):.4f}%")
    print(f"  C: {100 * relative_l2(C_np, C_prime):.4f}%")
    print(f"  phi: {100 * relative_l2(phi_np, phi_prime):.4f}%")

    # Numerical references for independent verification
    print("\nComputing fully coupled numerical reference...")
    T_ref, C_ref, phi_ref, S_ext, T_errors = compute_numerical_references()
    print(f"Coupled reference converged in {len(T_errors)} iterations, "
          f"final T_err = {T_errors[-1]:.6e}")

    print("\nRelative errors vs. numerical PDE solvers:")
    print(f"  T vs solve_2d_phi_2_T:  {100 * relative_l2(T_np, T_ref):.4f}%")
    print(f"  C vs solve_2d_phi_2_C:  {100 * relative_l2(C_np, C_ref):.4f}%")
    print(f"  phi vs solve_2d_T_C_2_phi: {100 * relative_l2(phi_np, phi_ref):.4f}%")

    # Save predictions
    np.savez('DeepM_Mnet_phi_T_C_predictions.npz',
             T=T_np, C=C_np, phi=phi_np,
             T_ref=T_ref, C_ref=C_ref, phi_ref=phi_ref,
             S_ext=S_ext, T_errors=T_errors)
    print("Predictions saved to DeepM_Mnet_phi_T_C_predictions.npz")

    # Visualize
    plot_fields(T_np, C_np, phi_np, T_ref=T_ref, C_ref=C_ref, phi_ref=phi_ref)
    plot_residuals(T_np, C_np, phi_np, T_prime, C_prime, phi_prime)


if __name__ == "__main__":
    main()
