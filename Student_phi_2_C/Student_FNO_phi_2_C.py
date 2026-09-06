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
C_inlet = 0.0   # inlet delayed-neutron precursor concentration
SIGMA = 0.05    # boundary smoothing length
C_SCALE = 5e5   # normalization scale for C (matches teacher)
C_INLET_NORM = C_inlet / C_SCALE  # normalized inlet value for hard BC

PHI_FILE = '1_FFT_phi_10000_201_101.npy'
C_FILE = 'C_pred_10000_101_201.npy'   # teacher predictions

MODEL_PATH = 'Student_FNO_phi_2_C_1.pth'
LOG_PATH = 'Student_FNO_phi_2_C_training_log_1.csv'


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
    2D Fourier Neural Operator for phi -> C.
    Input:  (B, 1, ny, nx)
    Output: (B, 1, ny, nx)
    """
    def __init__(self, width=64, modes1=16, modes2=16, n_layers=4):
        super(FNO2d, self).__init__()
        self.width = width

        # Lift: 1 (phi) + 2 (coords) channels -> width
        self.lift = nn.Conv2d(3, width, kernel_size=1)

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
        x = torch.cat([x, x_coord, y_coord], dim=1)  # (B, 3, ny, nx)

        x = self.lift(x)
        for block in self.fno_blocks:
            x = block(x)
        x = self.project(x)

        # Hard enforce inlet Dirichlet BC: C(0, y) = C_inlet
        coeff = 1.0 - torch.exp(-x_coord / SIGMA)
        return coeff * x + C_INLET_NORM


# -------------------- Data utilities --------------------
def load_data():
    """Load phi and C_pred datasets, normalize phi and C for stable training."""
    phi = np.load(PHI_FILE)            # (N, nx, ny) originally
    phi = phi.transpose(0, 2, 1)       # -> (N, ny, nx)
    C = np.load(C_FILE)                # already (N, ny, nx)
    assert phi.shape == C.shape, f"phi shape {phi.shape} != C shape {C.shape}"

    # Normalize phi: zero-mean, unit-std across all samples and spatial points
    phi_mean = phi.mean()
    phi_std = phi.std()
    phi_norm = (phi - phi_mean) / (phi_std + 1e-12)
    print(f"Loaded phi: {phi.shape}, C: {C.shape}")
    print(f"phi normalization: mean={phi_mean:.6e}, std={phi_std:.6e}")

    # Normalize C by C_SCALE to keep network outputs O(1)
    C_norm = C / C_SCALE
    print(f"C normalization: scale={C_SCALE:.6e}, raw min={C.min():.6e}, raw max={C.max():.6e}")

    # Save normalization params so inference can reproduce them
    np.savez('phi_normalization.npz', mean=phi_mean, std=phi_std)
    np.savez('C_normalization.npz', scale=C_SCALE)
    print("Saved phi normalization params to phi_normalization.npz")
    print("Saved C normalization params to C_normalization.npz")

    return phi_norm, C_norm, phi_mean, phi_std


PHI_MEAN = None
PHI_STD = None


def split_train_val(phi, C, train_ratio=0.9, seed=42):
    """Shuffle and split dataset into train/val."""
    N = phi.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(N * train_ratio)
    return (phi[perm[:n_train]], C[perm[:n_train]],
            phi[perm[n_train:]], C[perm[n_train:]],
            perm[:n_train], perm[n_train:])


# -------------------- Training --------------------
def train_student(phi_train, C_train, phi_val, C_val,
                  num_epochs=10000, batch_size=64,
                  width=64, modes1=16, modes2=16, n_layers=4,
                  lr=1e-3, weight_decay=1e-5,
                  eval_interval=10, patience=200,
                  seed=42):
    phi_train_t = torch.tensor(phi_train, dtype=torch.float32, device=device).unsqueeze(1)
    C_train_t = torch.tensor(C_train, dtype=torch.float32, device=device).unsqueeze(1)
    phi_val_t = torch.tensor(phi_val, dtype=torch.float32, device=device).unsqueeze(1)
    C_val_t = torch.tensor(C_val, dtype=torch.float32, device=device).unsqueeze(1)

    model = FNO2d(width=width, modes1=modes1, modes2=modes2, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    N_train = phi_train.shape[0]

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

            phi_batch = phi_train_t[batch_idx]
            C_true = C_train_t[batch_idx]

            optimizer.zero_grad()
            C_pred = model(phi_batch)
            loss = torch.mean((C_pred - C_true) ** 2) * 1e7
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
                all_C_pred = []
                all_C_true = []
                for start in range(0, C_val_t.shape[0], batch_size):
                    end = min(start + batch_size, C_val_t.shape[0])
                    phi_batch = phi_val_t[start:end]
                    C_true = C_val_t[start:end]
                    C_pred = model(phi_batch)
                    loss = torch.mean((C_pred - C_true) ** 2) * 1e7
                    val_loss += loss.item()
                    val_batches += 1
                    all_C_pred.append(C_pred)
                    all_C_true.append(C_true)

                avg_val_loss = val_loss / max(val_batches, 1)

                # Compute relative L2 over full validation set
                C_pred_full = torch.cat(all_C_pred, dim=0).cpu().numpy()
                C_true_full = torch.cat(all_C_true, dim=0).cpu().numpy()
                C_pred_flat = C_pred_full.reshape(C_val_t.shape[0], -1)
                C_true_flat = C_true_full.reshape(C_val_t.shape[0], -1)
                rel_l2_per_sample = np.linalg.norm(C_pred_flat - C_true_flat, axis=1) / \
                                    (np.linalg.norm(C_true_flat, axis=1) + 1e-12)
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
    global PHI_MEAN, PHI_STD
    phi, C, PHI_MEAN, PHI_STD = load_data()

    phi_train, C_train, phi_val, C_val, train_idx, val_idx = split_train_val(phi, C)
    print(f"Train: {phi_train.shape[0]} instances, Val: {phi_val.shape[0]} instances")

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
        phi_train, C_train, phi_val, C_val,
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
