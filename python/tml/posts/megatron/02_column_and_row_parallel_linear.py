"""Example of using ColumnParallelLinear and RowParallelLinear together in a simple forward pass.

This script initializes a process group, creates both ColumnParallelLinear and RowParallelLinear layers, runs a forward pass through both layers, and logs the output shapes.

To run this script, use the following command:

    python -m torchrun --nproc-per-node=2 02_column_and_row_parallel_linear.py

By default, it uses the "gloo" backend which runs on CPU. If you have multiple GPUs, you can use the "nccl" backend for GPU execution:

    python -m torchrun --nproc-per-node=2 02_column_and_row_parallel_linear.py --backend nccl --tp-size 2
"""

import logging
import os
import sys
from typing import Literal

import torch
import torch.distributed as dist
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import init_method_normal

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | rank=%(rank)s | %(message)s",
    stream=sys.stdout,
    force=True,
)


def main(backend: Literal["nccl", "gloo"], tp_size: int):
    dist.init_process_group(backend=backend)

    use_cpu_initialization = True if backend == "gloo" else False

    try:
        # get global rank
        rank = dist.get_rank()
        logger = logging.LoggerAdapter(logging.getLogger(__name__), {"rank": rank})
        logger.info("Process group initialized")

        # get local rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(f"Local rank: {local_rank}")

        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)

            model_parallel_cuda_manual_seed(1234)
        else:
            device = torch.device("cpu")
            torch.manual_seed(1234)

        # set model parallel cuda rng
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
        )

        # create a single column parallel linear layer
        config = ModelParallelConfig(use_cpu_initialization=use_cpu_initialization)
        layer_column_parallel = ColumnParallelLinear(
            input_size=768,
            output_size=768,
            config=config,
            init_method=init_method_normal(0.2),
            gather_output=False,
        ).to(device)

        layer_row_parallel = RowParallelLinear(
            input_size=768,
            output_size=768,
            config=config,
            init_method=init_method_normal(0.2),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        ).to(device)

        # create a dummy input
        seq_len, batch_size, dim = 128, 32, 768
        x = torch.randn(seq_len, batch_size, dim, device=device)
        y, y_bias = layer_column_parallel(x)
        y, y_bias = layer_row_parallel(y)

        logger.info(f"y shape: {y.shape}")
        logger.info(f"y_bias is None: {y_bias is None}")

    except Exception as e:
        print(e)

    # destroy process group (finalize)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="gloo")
    parser.add_argument("--tp-size", default=2, type=int)
    args = parser.parse_args()

    main(backend=args.backend, tp_size=args.tp_size)
