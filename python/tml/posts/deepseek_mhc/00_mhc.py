import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinkhorn_knopp(h: torch.Tensor, num_iter: int = 20, eps: float = 1e-6):
    """Sinkhorn-Knopp algorithm for matrix normalization."""
    h = torch.exp(h)
    for _ in range(num_iter):
        h = h / (h.sum(dim=-1, keepdim=True) + eps)
        h = h / (h.sum(dim=-2, keepdim=True) + eps)
    return h


class ManifoldHCParams(nn.Module):
    """Manifold constrained hyper-connection parameters.

    Args:
        num_hc: Number of hyper-connections
        d_model: Dimension of the model
    """

    def __init__(self, num_hc: int, d_model: int):
        super().__init__()
        self.num_hc = num_hc
        self.rms_norm = nn.RMSNorm(d_model * num_hc)

        self.pre_mix_proj = nn.Linear(num_hc * d_model, num_hc, bias=False)
        self.res_mix_proj = nn.Linear(num_hc * d_model, num_hc * num_hc, bias=False)
        self.post_mix_proj = nn.Linear(num_hc * d_model, num_hc, bias=False)

        self.pre_mix_scale = nn.Parameter(torch.tensor(0.01))
        self.res_mix_scale = nn.Parameter(torch.tensor(0.01))
        self.post_mix_scale = nn.Parameter(torch.tensor(0.01))

        self.pre_mix_shift = nn.Parameter(torch.zeros(num_hc))
        self.res_mix_shift = nn.Parameter(torch.zeros(num_hc, num_hc))
        self.post_mix_shift = nn.Parameter(torch.zeros(num_hc))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Outputs the three components of the mHC block: pre-mix, res-mix, and post-mix.

        Args:
            x: Input tensor to the block with shape [B, seq_len, num_hc, d_model]

        Returns:
            The pre-mix, res-mix, and post-mix tensors with shapes:
            - pre-mix: [B, seq_len, num_hc]
            - res-mix: [B, seq_len, num_hc, num_hc]
            - post-mix: [B, seq_len, num_hc]
        """
        b, seq_len, num_hc, d_model = x.shape
        x_vec = self.rms_norm(x.reshape(b, seq_len, num_hc * d_model))

        pre_mix_uncon = (
            self.pre_mix_scale * self.pre_mix_proj(x_vec) + self.pre_mix_shift
        )
        res_mix_uncon = (
            self.res_mix_scale
            * self.res_mix_proj(x_vec).reshape(b, seq_len, num_hc, num_hc)
            + self.res_mix_shift
        )
        post_mix_uncon = (
            self.post_mix_scale * self.post_mix_proj(x_vec) + self.post_mix_shift
        )

        pre_mix_con = F.sigmoid(pre_mix_uncon)
        res_mix_con = _sinkhorn_knopp(res_mix_uncon)
        post_mix_con = 2.0 * F.sigmoid(post_mix_uncon)

        return pre_mix_con, res_mix_con, post_mix_con


class HyperConnection(nn.Module):
    """Hyper-connection block.

    Args:
        num_hc: Number of hyper-connections
        d_model: Dimension of the model
        block: The block to be applied after the hyper-connection

    Returns:
        The output tensor from the block with shape [B, seq_len, num_hc, d_model]
    """

    def __init__(self, num_hc: int, d_model: int, block: nn.Module):
        super().__init__()
        self.mhc_params = ManifoldHCParams(num_hc, d_model)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hyper-connection block.

        Args:
            x: Input tensor to the block with shape [B, seq_len, num_hc, d_model]

        Returns:
            The output tensor from the block with shape [B, seq_len, num_hc, d_model]
        """
        pre_mix, res_mix, post_mix = self.mhc_params(x)
        block_in = (pre_mix.unsqueeze(2) @ x).squeeze(2)
        block_out = self.block(block_in)  # [B, seq_len, d_model]

        res_out = res_mix @ x
        post_out = post_mix.unsqueeze(-1) * block_out.unsqueeze(2)
        return res_out + post_out


class SelfAttentionBlock(nn.Module):
    """Self-attention block with RMS normalization.

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking.

        Args:
            x: Input tensor to the block with shape [B, seq_len, d_model]
        """
        x = self.norm(x)

        seq_len = x.shape[1]
        mask = torch.triu(
            torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=x.device,
            ),
            diagonal=1,
        )

        y, _ = self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=mask,
        )
        return y


class FFNBlock(nn.Module):
    """Feed-forward block with RMS normalization.

    Args:
        d_model: Dimension of the model
        mlp_ratio: Ratio of the hidden dimension to the model dimension
    """

    def __init__(self, d_model: int, mlp_ratio: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.norm(x))


class TransformerBlockDeepSeekV4(nn.Module):
    """Transformer block with mHC attention and feed-forward network.

    Args:
        num_hc: Number of hyper-connections
        d_model: Dimension of the model
        num_heads: Number of attention heads
        mlp_ratio: Ratio of the hidden dimension to the model dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_hc: int,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.mha_hc = HyperConnection(
            num_hc,
            d_model,
            SelfAttentionBlock(d_model, num_heads, dropout),
        )
        self.ffn_hc = HyperConnection(
            num_hc,
            d_model,
            FFNBlock(d_model, mlp_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:  Input tensor to the block with shape [B, seq_len, num_hc, d_model]
        """
        x = self.mha_hc(x)
        x = self.ffn_hc(x)
        return x


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 5
    num_hc = 3
    d_model = 4
    num_heads = 2
    mlp_ratio = 2

    # Test the ManifoldHCParams module
    mhc_module = ManifoldHCParams(num_hc, d_model)

    x = torch.randn(batch_size, seq_len, num_hc, d_model)
    a, b, c = mhc_module(x)

    assert torch.allclose(
        b.sum(dim=-1), torch.ones(batch_size, seq_len, num_hc), atol=1e-5
    ), "Rows of B do not sum to 1"
    assert torch.allclose(
        b.sum(dim=-2), torch.ones(batch_size, seq_len, num_hc), atol=1e-5
    ), "Columns of B do not sum to 1"

    assert a.shape == (batch_size, seq_len, num_hc), (
        f"Expected shape: [{batch_size}, {seq_len}, {num_hc}], but got: {a.shape}"
    )
    assert b.shape == (batch_size, seq_len, num_hc, num_hc), (
        f"Expected shape: [{batch_size}, {seq_len}, {num_hc}, {num_hc}], but got: {b.shape}"
    )
    assert c.shape == (batch_size, seq_len, num_hc), (
        f"Expected shape: [{batch_size}, {seq_len}, {num_hc}], but got: {c.shape}"
    )

    print("a shape:", a.shape, f"; expected: [{batch_size}, {seq_len}, {num_hc}]")
    print(
        "b shape:",
        b.shape,
        f"; expected: [{batch_size}, {seq_len}, {num_hc}, {num_hc}]",
    )
    print("c shape:", c.shape, f"; expected: [{batch_size}, {seq_len}, {num_hc}]")

    # Test the HyperConnection block
    hyper_connection_block = HyperConnection(
        num_hc, d_model, nn.Linear(d_model, d_model)
    )
    output = hyper_connection_block(x)

    assert output.shape == (batch_size, seq_len, num_hc, d_model), (
        f"Expected shape: [{batch_size}, {seq_len}, {num_hc}, {d_model}], but got: {output.shape}"
    )

    print(
        "output shape:",
        output.shape,
        f"; expected: [{batch_size}, {seq_len}, {num_hc}, {d_model}]",
    )

    # Test the TransformerBlockDeepSeekV4
    transformer_block = TransformerBlockDeepSeekV4(
        num_hc, d_model, num_heads, mlp_ratio, dropout=0.1
    )
    output = transformer_block(x)

    assert output.shape == (batch_size, seq_len, num_hc, d_model), (
        f"Expected shape: [{batch_size}, {seq_len}, {num_hc}, {d_model}], but got: {output.shape}"
    )

    print(
        "transformer output shape:",
        output.shape,
        f"; expected: [{batch_size}, {seq_len}, {num_hc}, {d_model}]",
    )
