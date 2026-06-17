import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp(h, num_iter=20, eps=1e-6):
    h = torch.exp(h)
    for _ in range(num_iter):
        h = h / (h.sum(dim=-1, keepdim=True) + eps)
        h = h / (h.sum(dim=-2, keepdim=True) + eps)
    return h


class mHC(nn.Module):
    """Manifold constrained hyper-connection module.

    Args:
        num_hc: Number of hyper-connections
    """

    def __init__(self, num_hc: int, d_model: int):
        super().__init__()
        self.num_hc = num_hc
        self.rms_norm = nn.RMSNorm(d_model * num_hc)

        self.a_proj = nn.Linear(num_hc * d_model, num_hc, bias=False)
        self.b_proj = nn.Linear(num_hc * d_model, num_hc * num_hc, bias=False)
        self.c_proj = nn.Linear(num_hc * d_model, num_hc, bias=False)

        self.a_scale = nn.Parameter(torch.ones(num_hc))
        self.b_scale = nn.Parameter(torch.ones(num_hc * num_hc))
        self.c_scale = nn.Parameter(torch.ones(num_hc))

        self.a_shift = nn.Parameter(torch.zeros(num_hc))
        self.b_shift = nn.Parameter(torch.zeros(num_hc, num_hc))
        self.c_shift = nn.Parameter(torch.zeros(num_hc))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor to the block with shape [B, seq_len, num_hc, d_model]

        Returns:
            Tuple of matrices with shapes [B, seq_len, num_hc], [B, seq_len, num_hc, num_hc], [B, seq_len, num_hc]
        """
        b, seq_len, num_hc, d_model = x.shape
        x_vec = self.rms_norm(x.view(b, seq_len, num_hc * d_model))

        a_uncon = self.a_scale * self.a_proj(x_vec) + self.a_shift
        b_uncon = (
            self.b_scale * self.b_proj(x_vec).view(-1, num_hc, num_hc) + self.b_shift
        )
        c_uncon = self.c_scale * self.c_proj(x_vec) + self.c_shift

        a_con = F.sigmoid(a_uncon)
        b_con = sinkhorn_knopp(b_uncon, num_iter=10)
        c_con = 2.0 * F.sigmoid(c_uncon)

        return a_con, b_con, c_con


class AttentionDeepSeekV4(nn.Module):
    def __init__(
        self,
        num_hc: int,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mhc_attn = mHC(num_hc, d_model)
        self.mhc_ffn = mHC(num_hc, d_model)
        self.mha = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.attn_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: NOT CORRECTLY IMPLEMENTED YET.

        Args:
            x:  Input tensor to the block with shape [B, seq_len, num_hc, d_model]
        """
        a_attn, b_attn, c_attn = self.mhc_attn(x, x, x)

        # [B, 1, num_hc] @ [B, num_hc, d_model] -> [B, 1, d_model] -> [B, d_model]
        u = torch.bmm(a_attn.unsqueeze(1), self.attn_norm(x)).squeeze(1)
        attn_out, _ = self.mha(u, u, u)
        x = torch.bmm(b_attn, x) + c_attn.unsqueeze(-1) @ attn_out.unsqueeze(1)

        a_ffn, b_ffn, c_ffn = self.mhc_ffn(x)
        x = b_ffn @ x + c_ffn.unsqueeze(-1) @ self.ffn(a_ffn.unsqueeze(1) @ x)

        return x
