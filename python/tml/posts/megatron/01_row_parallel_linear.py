"""Example of using Megatron's RowParallelLinear layer in a distributed setting with both CPU and GPU backends.

This script initializes a process group, creates a RowParallelLinear layer, runs a forward and backward pass, and gathers gradients for verification.

To run this script, use the following command:

    python -m torchrun --nproc-per-node=2 01_row_parallel_linear.py

By default, it uses the "gloo" backend which runs on CPU. If you have multiple GPUs, you can use the "nccl" backend for GPU execution:

    python -m torchrun --nproc-per-node=2 01_row_parallel_linear.py --backend nccl --tp-size 2
"""

import logging
import os
import sys
from typing import Literal

import torch
import torch.distributed as dist
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.tensor_parallel import RowParallelLinear
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

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
        )

        config = ModelParallelConfig(use_cpu_initialization=use_cpu_initialization)
        layer = RowParallelLinear(
            input_size=768,
            output_size=768,
            config=config,
            init_method=init_method_normal(0.2),
            bias=False,  # don't create/add bias
            input_is_parallel=False,  # the input is not sharded, set True if input comes from another TP layer
            skip_bias_add=True,  # if bias exists, the layer does not add it immediately; it returns bias separately so caller can fuse operations
        ).to(device)
        logger.info(f"layer.weight shape: {layer.weight.shape}")

        logger.info("Creating dummy input and running forward pass")
        seq_len, batch_size, dim = 128, 32, 768
        x = torch.randn(seq_len, batch_size, dim, device=device)
        dist.broadcast(x, src=0)

        y, y_bias = layer(x)

        logger.info(f"y shape: {y.shape}")
        logger.info(f"y_bias is None: {y_bias is None}")

        loss = (
            y.pow(2).mean()
        )  # each rank has the same y because RowParallelLinear ends with an all_reduce
        loss.backward()

        logger.info(f"layer.weight.grad: {layer.weight.grad.shape}")

        dist.barrier()
        logger.info("Gathering gradients from all TP ranks for verification")
        grad_list = [torch.empty_like(layer.weight.grad) for _ in range(tp_size)]
        dist.all_gather(
            tensor_list=grad_list,
            tensor=layer.weight.grad,
        )
        grad_list = torch.concat(grad_list, dim=1)
        logger.info(f"grad_list shape: {grad_list.shape}")

        logger.info("Gather layer weights from all TP ranks for verification")
        weight_list = [torch.empty_like(layer.weight.data) for _ in range(tp_size)]
        dist.all_gather(
            tensor_list=weight_list,
            tensor=layer.weight.data,
        )
        weight_list = torch.concat(weight_list, dim=1)

        if rank == 0:
            logger.info(
                "Create a reference gradient by running the same forward and backward pass on a non-parallel layer"
            )
            reference_layer = torch.nn.Linear(768, 768, bias=False).to(device)
            reference_layer.weight.data.copy_(weight_list)
            reference_layer.weight.requires_grad = True
            y_ref = reference_layer(x)
            loss_ref = y_ref.pow(2).mean()
            loss_ref.backward()
            logger.info(
                f"reference_layer.weight.grad shape: {reference_layer.weight.grad.shape}"
            )
            logger.info(
                f"Difference between gathered gradients and reference gradient: {(grad_list - reference_layer.weight.grad).abs().max()}"
            )

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
