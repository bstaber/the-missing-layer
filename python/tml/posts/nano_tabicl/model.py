import torch
import torch.nn as nn
from tml.nano_tabicl.blocks import InducedTransformerBlock, TransformerBlock


class TinyTabICL(nn.Module):
    def __init__(
        self,
        out_dim: int,
        d_model: int = 128,
        num_heads_col: int = 8,
        num_heads_row: int = 8,
        num_heads_icl: int = 8,
        num_col_blocks: int = 3,
        num_row_blocks: int = 3,
        num_icl_blocks: int = 3,
        num_inducing: int = 128,
        feature_group_size: int = 3,
    ):
        super().__init__()
        self.feature_group_size = feature_group_size

        self.proj_x = nn.Linear(feature_group_size, d_model)
        self.proj_y = nn.Linear(1, d_model)

        self.col_blocks = nn.ModuleList(
            InducedTransformerBlock(
                d_model=d_model,
                num_heads=num_heads_col,
                num_inducing=num_inducing,
                mlp_ratio=4.0,
                dropout=0.0,
            )
            for _ in range(num_col_blocks)
        )

        self.row_blocks = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads_row,
                mlp_ratio=4.0,
                dropout=0.0,
            )
            for _ in range(num_row_blocks)
        )

        self.icl_blocks = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads_icl,
                mlp_ratio=4.0,
                dropout=0.0,
            )
            for _ in range(num_icl_blocks)
        )

        self.row_cls_tokens = nn.Parameter(
            0.02 * torch.randn(1, 1, num_inducing, d_model)
        )

        self.row_norm = nn.LayerNorm(d_model)

        icl_dim = d_model * num_inducing
        self.out_norm = nn.LayerNorm(icl_dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(icl_dim, icl_dim * 2),
            nn.GELU(),
            nn.Linear(icl_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size, num_rows, num_cols = x.shape
        batch_size, num_train = y.shape

        # Normalization
        x = (x - x[:, :num_train].mean(dim=1, keepdim=True)) / (
            x[:, :num_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8
        )

        # Feature grouping
        idxs = torch.arange(num_cols, dtype=torch.long, device=x.device)
        x = torch.stack(
            [
                x[:, :, (idxs + (2**i - 1)) % num_cols]
                for i in range(self.feature_group_size)
            ],
            dim=-1,
        )

        # Input projections
        x_proj = self.proj_x(x)
        x_proj[:, :num_train] += self.proj_y(y[:, :, None, None])

        # Column blocks
        for block in self.col_blocks:
            x_proj = block(
                x_proj
            )  # missing feature: all rows only attend to training rows

        # Row blocks
        x_proj = torch.cat(
            [self.row_cls_tokens.expand(batch_size, num_rows, -1, -1), x_proj], dim=2
        )
        for block in self.row_blocks[:-1]:
            x_proj = block(x_proj)
        x_proj = self.row_blocks(
            x_proj
        )  # missing feature: need only the cls token values
        x_proj = self.row_norm(x_proj).flatten(-2, -1)

        # ICL blocks
        x_proj[:, :num_train] += self.proj_y_icl(y[:, :, None])
        for block in self.icl_blocks[:-1]:
            x_proj = block(x)  # missing feature: all rows only attend to training rows
        x_proj = self.row_blocks[-1](x_proj[:, num_train:], x_proj[:, :num_train])

        # Out projection
        return self.out_mlp(self.out_norm(x_proj))
