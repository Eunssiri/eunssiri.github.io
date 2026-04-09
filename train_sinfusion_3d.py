"""
SinFusion 3D DDPM Training Script
---------------------------------
- Input: gpm_preprocessed_cropped.npy with shape (N, 2, 192, 40, 32), values in [0, 1]
- Model: 3D U-Net based DDPM (unconditional)
- Output: checkpoints + generated samples

Run:
    python train_sinfusion_3d.py
"""

import os
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Dataset
# =============================================================================

class CroppedGPMDataset_Dynamic(Dataset):
    def __init__(self, path: str):
        arr = np.load(path)
        assert arr.ndim == 5, f"Expected 5D array, got shape {arr.shape}"
        assert arr.shape[1] == 2, f"Expected 2 channels, got {arr.shape[1]}"
        assert arr.shape[2:] == (192, 40, 32), f"Expected volume of shape (192, 40, 32), got {arr.shape[2:]}"
        self.data = torch.from_numpy(arr.astype(np.float32))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # (2, 192, 40, 32) in [0, 1]
        return self.data[idx]


# =============================================================================
# Time Embedding
# =============================================================================

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) integer timesteps
    returns: (B, dim) sinusoidal embedding
    """
    device = t.device
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half_dim, device=device).float() / half_dim
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


# =============================================================================
# 3D U-Net Backbone
# =============================================================================

class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()

        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.InstanceNorm3d(in_ch, affine=True),
            nn.SiLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        )

        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        """
        x:    (B, C, D, H, W)
        t_emb:(B, time_emb_dim)
        """
        h = self.block1(x)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        return h + self.skip(x)


class UNet3D(nn.Module):
    """
    3D U-Net for DDPM epsilon prediction.
    Adjusted for non-cubic volumes: e.g. (192, 40, 32)
    """
    def __init__(self, in_ch=2, base_ch=32, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.enc1 = ResidualBlock3D(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv3d(base_ch, base_ch * 2, kernel_size=4, stride=(2, 2, 2), padding=1)  # 192->96, 40->20, 32->16
        self.enc2 = ResidualBlock3D(base_ch * 2, base_ch * 2, time_emb_dim)
        self.down2 = nn.Conv3d(base_ch * 2, base_ch * 4, kernel_size=4, stride=(2, 2, 2), padding=1)  # 96->48, 20->10, 16->8
        self.enc3 = ResidualBlock3D(base_ch * 4, base_ch * 4, time_emb_dim)

        # Bottleneck
        self.bot1 = ResidualBlock3D(base_ch * 4, base_ch * 4, time_emb_dim)
        self.bot2 = ResidualBlock3D(base_ch * 4, base_ch * 4, time_emb_dim)

        # Decoder
        self.up1 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=4, stride=(2, 2, 2), padding=1)  # 48->96
        self.dec1 = ResidualBlock3D(base_ch * 4, base_ch * 2, time_emb_dim)
        self.up2 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=4, stride=(2, 2, 2), padding=1)  # 96->192
        self.dec2 = ResidualBlock3D(base_ch * 2, base_ch, time_emb_dim)

        self.out_norm = nn.InstanceNorm3d(base_ch, affine=True)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(base_ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        h1 = self.enc1(x, t_emb)                   # (B, B, 192, 40, 32)
        h2 = self.enc2(self.down1(h1), t_emb)      # (B, 2B, 96, 20, 16)
        h3 = self.enc3(self.down2(h2), t_emb)      # (B, 4B, 48, 10, 8)

        # Bottleneck
        hb = self.bot1(h3, t_emb)
        hb = self.bot2(hb, t_emb)

        # Decoder
        u1 = self.up1(hb)                          # (B, 2B, ~96, ~20, ~16)
        u1 = torch.cat(
            [u1[..., :h2.shape[-3], :h2.shape[-2], :h2.shape[-1]], h2],
            dim=1
        )
        u1 = self.dec1(u1, t_emb)

        u2 = self.up2(u1)                          # (B, B, ~192, ~40, ~32)
        u2 = torch.cat(
            [u2[..., :h1.shape[-3], :h1.shape[-2], :h1.shape[-1]], h1],
            dim=1
        )
        u2 = self.dec2(u2, t_emb)

        out = self.out_conv(self.out_act(self.out_norm(u2)))
        return out  # epsilon prediction in [-1,1]


# =============================================================================
# DDPM Core (Cosine Beta + Corrected x0)
# =============================================================================

class DDPM3D(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int, int],
        timesteps: int = 1000,
        s: float = 0.008,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.image_size = image_size

        # Cosine schedule (Nichol & Dhariwal style)
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=1e-8, max=0.999)

        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]],
            dim=0,
        )

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Forward diffusion: q(x_t | x_0)
        x0: (B, 2, D, H, W) in [-1,1]
        t:  (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_omcp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_acp * x0 + sqrt_omcp * noise, noise

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        Compute mean / var of p(x_{t-1} | x_t)
        """
        eps_theta = self.model(x_t, t)

        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_omcp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)

        # Correct x0 estimate
        x0_pred = (x_t - sqrt_omcp * eps_theta) / sqrt_acp
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        betas_t = self.betas[t].view(-1, 1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        acp_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1, 1)
        alphas_t = self.alphas[t].view(-1, 1, 1, 1, 1)

        coef1 = betas_t * torch.sqrt(acp_prev) / (1.0 - alphas_cumprod_t)
        coef2 = (1.0 - acp_prev) * torch.sqrt(alphas_t) / (1.0 - alphas_cumprod_t)
        posterior_mean = coef1 * x0_pred + coef2 * x_t

        posterior_var = self.posterior_variance[t].view(-1, 1, 1, 1, 1)
        posterior_var = posterior_var.clamp(min=1e-20)
        return posterior_mean, posterior_var

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor):
        posterior_mean, posterior_var = self.p_mean_variance(x_t, t)
        # 마지막 step(t=0)이 아니면 noise 추가
        add_noise = (t != 0).any()
        if add_noise:
            noise = torch.randn_like(x_t)
            return posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            return posterior_mean

    @torch.no_grad()
    def sample_hard_with_shape(
        self,
        batch_size: int,
        device: torch.device,
        shape,              # (D, H, W)
        orig,               # (C, D, H, W) or (C, K, J, I) one-hot
        well_ji_list,       # [(j,i), ...]
        t_start_ratio=0.5
    ):
        D, H, W = shape
        if isinstance(orig, np.ndarray):
            orig = torch.from_numpy(orig)

        C = orig.shape[0]
        T = self.timesteps
        t_start = int(T * t_start_ratio)

        x = torch.randn(batch_size, C, D, H, W, device=device)

        orig = orig.to(device).float() * 2.0 - 1.0  # [-1, 1]

        well_mask = torch.zeros((1, 1, D, H, W), device=device)
        well_x0   = torch.zeros((1, C, D, H, W), device=device)

        for (j, i) in well_ji_list:
            well_mask[0, 0, :, j, i] = 1.0
            well_x0[0, :, :, j, i]   = orig[:, :, j, i]

        well_mask = well_mask.repeat(batch_size, 1, 1, 1, 1)
        well_x0   = well_x0.repeat(batch_size, 1, 1, 1, 1)

        noise_fixed = torch.randn_like(well_x0)

        for t in reversed(range(T)):
            t_b = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_b)

            if t <= t_start and t > 0:
                t_prev = t - 1
                t_prev_b = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)

                sqrt_acp_prev = self.sqrt_alphas_cumprod[t_prev_b].view(-1, 1, 1, 1, 1)
                sqrt_om_prev  = self.sqrt_one_minus_alphas_cumprod[t_prev_b].view(-1, 1, 1, 1, 1)

                well_xt_prev = sqrt_acp_prev * well_x0 + sqrt_om_prev * noise_fixed
                x = well_mask * well_xt_prev + (1.0 - well_mask) * x

            if t == 0:
                x = well_mask * well_x0 + (1.0 - well_mask) * x

        x = (x + 1.0) * 0.5
        return x.clamp(0.0, 1.0)




import torch.nn.functional as F

class DDPM3D_hard(DDPM3D):
    """
    DDPM with XY-based Hard Conditioning and output resizing.
    """
    def generate_mask_and_values(self, xy_indices, hard_values_original, target_shape):
        """
        Generates a 3D mask and hard values (broadcast over depth) for Z direction.
        - xy_indices: list of tuples (x, y) for hard conditioning
        - hard_values_original: numpy array or tensor, original values at those positions, shape (2, Z, Y, X)
        - target_shape: final output shape (B, C, D, H, W)
        """
        _, C, D, H, W = target_shape
        mask = torch.zeros((1, C, D, H, W), dtype=torch.float32)
        hard_values = torch.zeros_like(mask)

        # Fill hard data in mask and hard values
        for ch in range(C):
            for x, y in xy_indices:
                mask[0, ch, :, y, x] = 1.0  # entire depth for (x, y)
                hard_values[0, ch, :, y, x] = hard_values_original[ch, :, y, x]

        return mask, hard_values

    @torch.no_grad()
    def p_sample_hard(self, x_t, t, mask, hard_values):
        """
        One denoising step with XY-based hard conditioning.
        """
        posterior_mean, posterior_var = self.p_mean_variance(x_t, t)
        noise = torch.randn_like(x_t) if (t != 0).any() else 0.0
        sample = posterior_mean + torch.sqrt(posterior_var) * noise

        # Overwrite masked regions with hard data
        return sample * (1 - mask) + hard_values * mask

    @torch.no_grad()
    def sample_hard(self, batch_size, device, mask, hard_values):
        """
        Sampling with XY-based hard conditioning.
        """
        D, H, W = self.image_size  # Model output dims (192, 40, 32)
        x = torch.randn(batch_size, 2, D, H, W, device=device)

        # Broadcast mask and hard values
        if mask.shape[0] == 1:
            mask = mask.expand(batch_size, -1, -1, -1, -1).to(device)
            hard_values = hard_values.expand(batch_size, -1, -1, -1, -1).to(device)

        # DDPM sampling loop
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample_hard(x, t_batch, mask, hard_values)

        # Upsample to original dimensions
        x = (x + 1.0) * 0.5  # [-1,1] to [0,1]
        x = F.interpolate(x, size=(200, 50, 36), mode="trilinear", align_corners=False)

        return x.clamp(0.0, 1.0)


# =============================================================================
# Training Loop
# =============================================================================

def train_sinfusion(
    data_path: str = "StratoPy_Ouputs/gpm_preprocessed_cropped.npy",
    outdir: str = "./SinFusion_output_v2",
    epochs: int = 5001,
    batch_size: int = 8,
    timesteps: int = 1000,   # 1000-step cosine schedule
    lr: float = 1e-4,
    device: str = "cuda",
):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset & Loader
    dataset = CroppedGPMDataset_Dynamic(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset size: {len(dataset)} patches")

    # Model & DDPM wrapper
    unet = UNet3D(in_ch=2, base_ch=32, time_emb_dim=256).to(device)
    ddpm = DDPM3D(unet, image_size=(192, 40, 32), timesteps=timesteps).to(device)

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

    for epoch in range(epochs):
        ddpm.train()
        for x0 in loader:
            x0 = x0.to(device)          # (B,2,192,40,32), in [0,1]
            x0 = x0 * 2.0 - 1.0         # map to [-1,1]

            b = x0.shape[0]
            t = torch.randint(0, timesteps, (b,), device=device).long()

            x_t, noise = ddpm.q_sample(x0, t)
            noise_pred = ddpm.model(x_t, t)

            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}/{epochs}] loss = {loss.item():.6f}")

        # Save samples & checkpoint periodically
        if (epoch + 1) % 10 == 0:
            ddpm.eval()
            with torch.no_grad():
                # Save samples as numpy
                samples = ddpm.sample(batch_size=2, device=device)  # [0,1]
                npy_path = os.path.join(outdir, f"samples_epoch{epoch+1}.npy")
                # np.save(npy_path, samples.cpu().numpy())
                print(f"Saved samples to {npy_path}")

                # Save PNG slices
                try:
                    import matplotlib.pyplot as plt
                    v = samples[0].cpu()  # (2,192,40,32)
                    D = v.shape[1]
                    mid = D // 2  # Depth slice index

                    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                    axs[0].imshow(v[0, mid].numpy(), origin="upper", cmap='cividis')
                    axs[0].set_title("Channel 0 (mid slice)")
                    axs[0].axis("off")

                    axs[1].imshow(v[1, mid].numpy(), origin="upper", cmap = 'Oranges')
                    axs[1].set_title("Channel 1 (mid slice)")
                    axs[1].axis("off")

                    png_path = os.path.join(outdir, f"samples_epoch{epoch+1}.png")
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"Saved PNG slice to {png_path}")
                except ImportError:
                    print("Matplotlib not available. Skipping PNG saving.")

                # Save checkpoint
                ckpt_path = os.path.join(outdir, f"sinfusion_3d_epoch{epoch+1}.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": ddpm.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"Checkpoint saved to {ckpt_path}")

