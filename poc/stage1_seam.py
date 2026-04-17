"""
Stage 1 Seam Analysis — MuJEPA PoC
===================================

MAE loss location
-----------------
File: muvit/mae.py, lines 311-312
  loss = F.mse_loss(z[batch_range, idx_mask], patches_normed[batch_range, idx_mask])
Tensor shape at loss site: (M, patch_dim)
  where M = total masked patches across batch, patch_dim = in_channels * prod(patch_size)

Encoder output shape
--------------------
Calling model.encoder(x, bbox) directly returns: (B, N, dim)
  where N = N_per_level * num_levels
  PoC config (patch_size=8, img=64x64, 2 levels): N = (64/8)^2 * 2 = 128

SIGReg (Sketched Isotropic Gaussian Regularization)
----------------------------------------------------
Source: LeJEPA MINIMAL.md (Balestriero & LeCun, arXiv:2511.08544)
Import: NOT exported from the installed lejepa package.
        Must be defined as a standalone class (copied below).

SIGReg input shape: (..., N, D)
  The forward() averages over dim -3 (the N/token axis).
  (B, N, D) 3D input works natively — no reshape needed.

Device fix
----------
MINIMAL.md hardcodes device="cuda" for the random projection matrix.
Fix applied below: device=proj.device (works on CPU and CUDA alike).

Reshape verdict
---------------
MAE loss: (M, patch_dim) at mae.py:311-312
Encoder output: (B, N, dim)
SIGReg input: (..., N, D) — accepts (B, N, dim) directly
Reshape needed: NO
Device fix required: YES (applied below)
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization loss from LeJEPA.

    Source: Balestriero & LeCun, arXiv:2511.08544, MINIMAL.md
    Device fix: random projection matrix uses proj.device instead of "cuda".
    Input: (..., N, D) tensor — works with (B, N, D) encoder output directly.
    Output: scalar loss.
    """

    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


if __name__ == "__main__":
    import muvit   # noqa: F401
    import lejepa  # noqa: F401
    print("import muvit: OK")
    print("import lejepa: OK")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigreg = SIGReg().to(device)
    fake_encoder_output = torch.rand(2, 128, 64, device=device)  # (B=2, N=128, D=64)
    loss = sigreg(fake_encoder_output)
    assert loss.isfinite(), f"Expected finite loss, got {loss.item()}"
    print(f"SIGReg forward (B=2, N=128, D=64): loss={loss.item():.4f}  OK")
