import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    ModelParallelConfig,
    RowParallelLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import init_method_normal, parallel_state
from typing_extensions import Literal

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | rank=%(rank)s | %(message)s",
    stream=sys.stdout,
    force=True,
)


class TensorParallelFFN(nn.Module):
    """Tensor parallel FFN module for our attention block.

    Args:
        in_features: input feature dimension
        hidden_dim: hidden dimension in the FFN
        out_features: output feature dimension
        parallel_config: model parallel config
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        parallel_config: ModelParallelConfig,
    ):
        super().__init__()
        self.up_proj = ColumnParallelLinear(
            input_size=in_features,
            output_size=hidden_dim,
            init_method=init_method_normal(0.02),
            bias=False,
            gather_output=False,
            config=parallel_config,
        )
        self.down_proj = RowParallelLinear(
            input_size=hidden_dim,
            output_size=out_features,
            init_method=init_method_normal(0.02),
            input_is_parallel=True,
            bias=False,
            skip_bias_add=True,
            config=parallel_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the tensor parallel FFN network.

        Args:
            x: input tensor of shape (batch_size, seq_len, in_features)

        Returns:
            output tensor of shape (batch_size, seq_len, out_features)
        """
        x, _ = self.up_proj(x)
        x = F.gelu(x)
        x, _ = self.down_proj(x)
        return x


class TensorParallelMultiHeadAttention(nn.Module):
    """Tensor parallel multi-head attention module.

    Args:
        d_model: hidden dimension of the model
        num_heads: number of attention heads
        parallel_config: model parallel config
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        parallel_config: ModelParallelConfig,
    ):
        super().__init__()
        self.d_model = d_model
        self.global_num_heads = num_heads

        assert num_heads % parallel_config.tensor_model_parallel_size == 0, (
            "num_heads must be divisible by tensor_model_parallel_size"
        )
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.local_num_heads = num_heads // parallel_config.tensor_model_parallel_size
        self.head_dim = d_model // num_heads

        self.q_proj = ColumnParallelLinear(
            input_size=d_model,
            output_size=d_model,
            config=parallel_config,
            init_method=init_method_normal(0.02),
            gather_output=False,
            bias=False,
        )
        self.k_proj = ColumnParallelLinear(
            input_size=d_model,
            output_size=d_model,
            config=parallel_config,
            init_method=init_method_normal(0.02),
            gather_output=False,
            bias=False,
        )
        self.v_proj = ColumnParallelLinear(
            input_size=d_model,
            output_size=d_model,
            config=parallel_config,
            init_method=init_method_normal(0.02),
            gather_output=False,
            bias=False,
        )

        self.attn_out_proj = RowParallelLinear(
            input_size=d_model,
            output_size=d_model,
            config=parallel_config,
            init_method=init_method_normal(0.02),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the tensor parallel MHA.

        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        q, _ = self.q_proj(x)
        k, _ = self.k_proj(x)
        v, _ = self.v_proj(x)

        q = rearrange(q, "b t (h d) -> b h t d", h=self.local_num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.local_num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.local_num_heads)

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        context = rearrange(context, "b h t d -> b t (h d)", h=self.local_num_heads)

        out_attn, _ = self.attn_out_proj(context)
        return out_attn


class TensorParallelAttentionBlock(nn.Module):
    """Tensor parallel attention block.

    Args:
        d_model: hidden dimension of the model
        num_heads: number of attention heads
        mlp_ratio: up/down dimension ratio in FFN branch
        config: model parallel config
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        parallel_config: ModelParallelConfig,
    ):
        super().__init__()
        self.d_model = d_model

        self.mha = TensorParallelMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            parallel_config=parallel_config,
        )

        self.ffn = TensorParallelFFN(
            in_features=d_model,
            hidden_dim=mlp_ratio * d_model,
            out_features=d_model,
            parallel_config=parallel_config,
        )
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the tensor parallel attention block.

        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


def main(backend: Literal["nccl", "gloo"], tp_size: int):
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    logger = logging.LoggerAdapter(logging.getLogger(__name__), {"rank": rank})
    logger.info("Process group initialized")

    if backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        model_parallel_cuda_manual_seed(1234)
    else:
        device = torch.device("cpu")
        torch.manual_seed(1234)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
    )

    config = ModelParallelConfig(use_cpu_initialization=True)
    attn_block = TensorParallelAttentionBlock(
        d_model=768,
        num_heads=12,
        mlp_ratio=4,
        parallel_config=config,
    ).to(device)

    seq_len, batch_size = 128, 32
    x = torch.randn(batch_size, seq_len, 768, device=device)
    dist.broadcast(x, src=0)

    y = attn_block(x)

    logger.info(f"Output shape: {y.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["nccl", "gloo"], default="gloo")
    parser.add_argument("--tp-size", type=int, default=2)
    args = parser.parse_args()

    main(backend=args.backend, tp_size=args.tp_size)
