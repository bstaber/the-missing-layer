import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)

        hidden_dim = int(mlp_ratio * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model)
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.norm1(x)

        if context is None:
            kv = q
        else:
            kv = self.norm1(context)

        attn_output, _ = self.attn(
            q,
            kv,
            kv,
            need_weights=False,
        )
        x = x + attn_output

        x = x + self.ffn(self.norm2(x))
        return x


class InducedTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_inducing: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.inducing_tokens = nn.Parameter(
            0.02 * torch.randn(1, num_inducing, d_model)
        )
        self.compress_block = TransformerBlock(
            d_model,
            num_heads,
            mlp_ratio,
            dropout,
        )
        self.decompress_block = TransformerBlock(
            d_model,
            num_heads,
            mlp_ratio,
            dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        q = self.inducing_tokens.expand(batch_size, -1, -1)
        source = context if context is not None else x

        z = self.compress_block(q, source)
        out = self.decompress_block(x, z)

        return out


if __name__ == "__main__":
    x = torch.randn(2, 7, 32)
    context = torch.randn(2, 11, 32)

    block = TransformerBlock(32, 4, 4.0)
    y = block(x)
    print(y.shape)

    y = block(x, context)
    print(y.shape)

    block = InducedTransformerBlock(
        d_model=32,
        num_heads=4,
        num_inducing=5,
        mlp_ratio=4.0,
    )

    x = torch.randn(2, 7, 32)
    y = block(x)

    assert y.shape == (2, 7, 32)

    x = torch.randn(2, 10, 20, 32)
    x = x.permute(0, 2, 1, 3).reshape(-1, 10, 32)

    y: torch.Tensor = block(x)

    assert y.shape == (40, 10, 32)

    y = y.view(2, 20, 10, 32)
    y = y.permute(0, 2, 1, 3)
