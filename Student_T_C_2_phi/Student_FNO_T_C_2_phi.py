import torch
import torch.nn as nn
import numpy as np
import time
import os
import csv

print("CUDA available:", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_dtype(torch.float32)

# -------------------- Physics / data params --------------------
L = 2.0         # domain length in x (m)
H = 1.0         # domain height in y (m)
SIGMA = 0.05    # boundary smoothing length
PHI_SCALE = 2.5e7   # normalization scale for phi (matches teacher)

T_FILE = '1_FFT_T_10000_101_201.npy'
C_FILE = '1_FFT_C_10000_101_201.npy'
PHI_FILE = 'phi_pred_10000_201_101.npy'   # teacher predictions (N, nx, ny)

MODEL_PATH = 'Student_FNO_T_C_2_phi_1.pth'
LOG_PATH = 'Student_FNO_T_C_2_phi_training_log_1.csv'


# -------------------- FNO building blocks --------------------
class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. It does FFT, linear transform on Fourier space, and IFFT.
    Uses real 2D FFT (rfft2) to save memory.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep in y
        self.modes2 = modes2  # Number of Fourier modes to keep in x (roughly nx/2 for rfft)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )

    @staticmethod
    def compl_mul2d(input, weights):
        """Complex multiplication in batch: (B, C_in, M1, M2) x (C_in, C_out, M1, M2) -> (B, C_out, M1, M2)"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Real 2D FFT: (B, C, ny, nx) -> (B, C, ny, nx//2+1) complex
        x_ft = torch.fft.rfft2(x)

        # Initialize output in Fourier space
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], torch.view_as_complex(self.weights2)
        )

        # Inverse real 2D FFT back to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOBlock(nn.Module):
    """FNO block: spectral conv + pointwise conv + activation."""
    def __init__(self, width, modes1, modes2):
        super(FNOBlock, self).__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise_conv = nn.Conv2d(width, width, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.pointwise_conv(x)
        return self.activation(x1 + x2)


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for (T, C) -> phi.
    Input:  (B, 2, ny, nx)  [T channel, C channel]
    Output: (B, 1, ny, nx)  [normalized phi]
    """
    def __init__(self, width=64, modes1=16, modes2=16, n_layers=4):
        super(FNO2d, self).__init__()
        self.width = width

        # Lift: 2 (T, C) + 2 (coords) channels -> width
        self.lift = nn.Conv2d(4, width, kernel_size=1)

        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2) for _ in range(n_layers)
        ])

        # Project back to 1 normalized channel
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def _build_coords(self, B, ny, nx, device):
        # Use physical coordinates [0, L] x [0, H]
        x = torch.linspace(0.0, L, nx, device=device).view(1, 1, 1, nx)
        y = torch.linspace(0.0, H, ny, device=device).view(1, 1, ny, 1)
        x = x.expand(B, 1, ny, nx)
        y = y.expand(B, 1, ny, nx)
        return x, y

    def forward(self, x):
        B, _, ny, nx = x.shape
        device = x.device

        x_coord, y_coord = self._build_coords(B, ny, nx, device)
        x = torch.cat([x, x_coord, y_coord], dim=1)  # (B, 4, ny, nx)

        x = self.lift(x)
        for block in self.fno_blocks:
            x = block(x)
        x = self.project(x)

        # Hard enforce vacuum BC: phi = 0 on all four boundaries
        coeff_x = 1.0 - torch.exp(-x_coord / SIGMA) - torch.exp(-(L - x_coord) / SIGMA)
        coeff_y = 1.0 - torch.exp(-y_coord / SIGMA) - torch.exp(-(H - y_coord) / SIGMA)
        coeff = coeff_x * coeff_y
        return coeff * x


# -------------------- Data utilities --------------------
def _match_spatial_shape(arr, target_shape):
    """Ensure arr has the same (ny, nx) spatial shape as target_shape.

    Some saved phi arrays are stored as (N, nx, ny) while T/C are (N, ny, nx).
    """
    if arr.shape[1:] == target_shape[1:]:
        return arr
    if arr.shape[1] == target_shape[2] and arr.shape[2] == target_shape[1]:
        return arr.transpose(0, 2, 1)
    raise ValueError(
        f"Cannot reconcile spatial shape {arr.shape[1:]} with target {target_shape[1:]}"
    )


def load_data():
    """Load T, C, and phi_pred datasets, normalize for stable training."""
    T = np.load(T_FILE)                 # (N, ny, nx)
    C = np.load(C_FILE)                 # (N, ny, nx)
    phi = np.load(PHI_FILE)             # (N, ny, nx) or (N, nx, ny)
    phi = _match_spatial_shape(phi, T.shape)
    assert T.shape == C.shape == phi.shape, \
        f"T shape {T.shape}, C shape {C.shape}, phi shape {phi.shape} must match"

    # Normalize T and C: zero-mean, unit-std per field
    T_mean = T.mean()
    T_std = T.std()
    T_norm = (T - T_mean) / (T_std + 1e-12)

    C_mean = C.mean()
    C_std = C.std()
    C_norm = (C - C_mean) / (C_std + 1e-12)

    # Normalize phi by PHI_SCALE
    phi_norm = phi / PHI_SCALE

    print(f"Loaded T: {T.shape}, C: {C.shape}, phi: {phi.shape}")
    print(f"T normalization: mean={T_mean:.6f}, std={T_std:.6f}")
    print(f"C normalization: mean={C_mean:.6f}, std={C_std:.6f}")
    print(f"phi normalization: scale={PHI_SCALE:.6e}, raw min={phi.min():.6e}, raw max={phi.max():.6e}")

    # Save normalization params so inference can reproduce them
    np.savez('T_C_2_phi_normalization.npz',
             T_mean=T_mean, T_std=T_std,
             C_mean=C_mean, C_std=C_std,
             phi_scale=PHI_SCALE)
    print("Saved normalization params to T_C_2_phi_normalization.npz")

    # Stack T and C as two input channels
    inputs = np.stack([T_norm, C_norm], axis=1)  # (N, 2, ny, nx)
    return inputs, phi_norm, T_mean, T_std


T_MEAN = None
T_STD = None


def split_train_val(inputs, phi, train_ratio=0.9, seed=42):
    """Shuffle and split dataset into train/val."""
    N = inputs.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(N * train_ratio)
    return (inputs[perm[:n_train]], phi[perm[:n_train]],
            inputs[perm[n_train:]], phi[perm[n_train:]],
            perm[:n_train], perm[n_train:])


# -------------------- Training --------------------
def train_student(inputs_train, phi_train, inputs_val, phi_val,
                  num_epochs=10000, batch_size=64,
                  width=64, modes1=16, modes2=16, n_layers=4,
                  lr=1e-3, weight_decay=1e-5,
                  eval_interval=10, patience=200,
                  seed=42):
    inputs_train_t = torch.tensor(inputs_train, dtype=torch.float32, device=device)
    phi_train_t = torch.tensor(phi_train, dtype=torch.float32, device=device).unsqueeze(1)
    inputs_val_t = torch.tensor(inputs_val, dtype=torch.float32, device=device)
    phi_val_t = torch.tensor(phi_val, dtype=torch.float32, device=device).unsqueeze(1)

    model = FNO2d(width=width, modes1=modes1, modes2=modes2, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    N_train = inputs_train.shape[0]

    start_time = time.time()

    with open(LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_mse', 'val_mse', 'val_rel_l2_mean',
                         'val_rel_l2_min', 'val_rel_l2_max', 'lr', 'elapsed_sec'])

    for epoch in range(num_epochs):
        model.train()
        indices = np.random.permutation(N_train)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, N_train, batch_size):
            end = min(start + batch_size, N_train)
            batch_idx = indices[start:end]

            inputs_batch = inputs_train_t[batch_idx]
            phi_true = phi_train_t[batch_idx]

            optimizer.zero_grad()
            phi_pred = model(inputs_batch)
            loss = torch.mean((phi_pred - phi_true) ** 2) * 1e7
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % eval_interval == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_batches = 0
                all_phi_pred = []
                all_phi_true = []
                for start in range(0, phi_val_t.shape[0], batch_size):
                    end = min(start + batch_size, phi_val_t.shape[0])
                    inputs_batch = inputs_val_t[start:end]
                    phi_true = phi_val_t[start:end]
                    phi_pred = model(inputs_batch)
                    loss = torch.mean((phi_pred - phi_true) ** 2) * 1e7
                    val_loss += loss.item()
                    val_batches += 1
                    all_phi_pred.append(phi_pred)
                    all_phi_true.append(phi_true)

                avg_val_loss = val_loss / max(val_batches, 1)

                # Compute relative L2 over full validation set
                phi_pred_full = torch.cat(all_phi_pred, dim=0).cpu().numpy()
                phi_true_full = torch.cat(all_phi_true, dim=0).cpu().numpy()
                phi_pred_flat = phi_pred_full.reshape(phi_val_t.shape[0], -1)
                phi_true_flat = phi_true_full.reshape(phi_val_t.shape[0], -1)
                rel_l2_per_sample = np.linalg.norm(phi_pred_flat - phi_true_flat, axis=1) / \
                                    (np.linalg.norm(phi_true_flat, axis=1) + 1e-12)
                rel_l2_mean = rel_l2_per_sample.mean()
                rel_l2_min = rel_l2_per_sample.min()
                rel_l2_max = rel_l2_per_sample.max()

            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch:6d}] Train MSE: {avg_train_loss:.4e}, "
                f"Val MSE: {avg_val_loss:.4e}, Val RelL2: {rel_l2_mean*100:.4f}% "
                f"[{rel_l2_min*100:.4f}%, {rel_l2_max*100:.4f}%], "
                f"LR: {current_lr:.2e}, Time: {elapsed:.1f}s"
            )

            with open(LOG_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch,
                                 f"{avg_train_loss:.6e}",
                                 f"{avg_val_loss:.6e}",
                                 f"{rel_l2_mean:.6e}",
                                 f"{rel_l2_min:.6e}",
                                 f"{rel_l2_max:.6e}",
                                 f"{current_lr:.6e}",
                                 f"{elapsed:.1f}"])

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  -> Saved best model (val MSE {best_val_loss:.4e})")
            else:
                epochs_no_improve += eval_interval  # count by epochs since last eval

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}: no improvement for {patience} epochs "
                      f"(best epoch {best_epoch}, best val MSE {best_val_loss:.4e})")
                break

    print(f"Training complete. Best validation MSE: {best_val_loss:.4e} at epoch {best_epoch}")
    return model


# -------------------- Main --------------------
def main():
    global T_MEAN, T_STD
    inputs, phi, T_MEAN, T_STD = load_data()

    inputs_train, phi_train, inputs_val, phi_val, train_idx, val_idx = split_train_val(inputs, phi)
    print(f"Train: {inputs_train.shape[0]} instances, Val: {inputs_val.shape[0]} instances")

    # Server training hyperparameters
    num_epochs = 10000
    batch_size = 64
    width = 64
    modes1 = 16
    modes2 = 16
    n_layers = 4

    if os.path.exists(MODEL_PATH):
        print(f"Found existing model {MODEL_PATH}. Skipping training.")
        print("Delete or rename the file to retrain.")
        return

    train_student(
        inputs_train, phi_train, inputs_val, phi_val,
        num_epochs=num_epochs,
        batch_size=batch_size,
        width=width,
        modes1=modes1,
        modes2=modes2,
        n_layers=n_layers
    )

    print(f"Training finished. Model saved to {MODEL_PATH}")
    print(f"Training log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
