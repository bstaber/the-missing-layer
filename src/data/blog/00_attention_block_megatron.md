---
author: Brian Staber
pubDatetime: 2026-05-12T20:19:08Z
modDatetime: 2026-05-12T20:19:08Z
title: Building a tensor-parallel transformer block with Megatron-Core
slug: building-a-tensor-parallel-transformer-block-with-megatron-core
featured: false
draft: false
tags:
  - megatron-core
  - tensor parallelism
  - transformer
description: Let's build a tensor-parallel transformer block using Megatron-Core, a library for model parallelism in PyTorch.
---

# Introduction

I wanted to learn more about Megatron-LM and Megatron-Core, two libraries developed by NVIDIA for training large language models with different parallelism strategies. Megatron-LM seems to be focused on LLM training, with several high-level APIs for training well known architectures like GPT or Llama. Megatron-Core, on the other hand, looks more like a lower-level library with more flexibility. I decided to try implementing a simple transformer block using Megatron-Core.

If you don't know what I'm talking about, check out Hugging Face's [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) which has a great overview of the different parallelism techniques and libraries available. Hugging Face also developped [nanotron](https://github.com/huggingface/nanotron) and [picotron](https://github.com/huggingface/picotron), two libraries for training LLMs as well, slightly more accessible than Megatron-LM and Megatron-Core.

## Dependencies

We don't need much to get started: PyTorch, Megatron-Core, and eventually einops for tensor manipulation. You can install them with:

```bash
uv add torch numpy einops megatron-core
```

## Tensor parallelism

In this exercise, I'm solely relying on tensor parallelism (TP). Other parallelism techniques include: data parallelism, pipeline parallelism, sequence parallelism, or expert parallelism; but I'll leave those for another time. Tensor parallelism consists in splitting the weights of a layer across multiple devices, and performing the forward and backward pass in parallel across those devices. This is useful when the model is too large to fit on a single device. For a more detailed explanation of tensor parallelism, check out this [section](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism) of the Ultra-Scale Playbook. At this stage, I'm assuming that you're familiar with the two main types of tensor parallelism: column parallelism and row parallelism.

In an attention block, we have two main components: the multi-head attention (MHA) and the feed-forward network (FFN). Both of these components involve linear layers that can be parallelized with TP. For the MHA, we can split the query, key, and value projections across devices. For the FFN, we can split the up and down projections across devices as well.

There are two useful TP linear layers provided by Megatron-Core: `ColumnParallelLinear` and `RowParallelLinear`. The `ColumnParallelLinear` layer splits the weights across the column dimension, while the `RowParallelLinear` layer splits the weights across the row dimension. They can even be combined: for example, in the MHA, we can use `ColumnParallelLinear` for the query, key, and value projections, and then use `RowParallelLinear` for the output projection. In the FFN, we can use `ColumnParallelLinear` for the up projection and `RowParallelLinear` for the down projection. We will dive into the implementation details in the next section.

# Implementation

We will implement a `TensorParallelMultiHeadAttention` class and a `TensorParallelFFN` class, and then combine them in a `TensorParallelAttentionBlock` class.

## Column parallel linear layer

Megatron-Core provides a `ColumnParallelLinear` layer that splits the weights across the column dimension. It takes care about broadcoasting the input and optionally gathering the output across devices. Here's how we can create a column parallel linear layer:

```python
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.random import init_method_normal

config = ModelParallelConfig()
layer = ColumnParallelLinear(
    input_size=input_size,
    output_size=output_size,
    config=config,
    init_method=init_method_normal(0.02),
    gather_output=True,
)
```

Here, `config` is an object that contains the model parallel configuration. It has several attributes that we left at their default values. The `init_method` is a function that initializes the weights of the layer. And `gather_output` indicates whether we want to gather the output across devices or not. If `gather_output` is `False`, the output remains split across devices (column-wise), which can be useful for intermediate projections like in MHA. If `gather_output` is `True`, the output is gathered on all devices thanks to an `all_gather` operation.

However, you can't run this code as is, because we need to initialiaze and define several things. You can try, but you'll get some nasty errors. We need to define:

- the PyTorch distributed process group
- the device to use for each process
- Megatron-Core's parallel state
- a random seed, required if you're using a GPU (`nccl` backend)

Let's look at a full working example. I'll start by creating the layer, a fake input, and performing a forward pass. Then, I'll add a dummy loss and perform a backward pass, and finally compare the gradients with a non-parallel linear layer. A link to the full code will be provided at the end of the section.

```python
import logging
import os
import sys
from typing import Literal

import torch
import torch.distributed as dist
# [!code highlight:4]
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
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
        # setup logger with rank
        rank = dist.get_rank()
        logger = logging.LoggerAdapter(logging.getLogger(__name__), {"rank": rank})
        logger.info("Process group initialized")

        # get local rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)

            # [!code highlight:1]
            model_parallel_cuda_manual_seed(1234)
        else:
            device = torch.device("cpu")
            torch.manual_seed(1234)

        # set model parallel cuda rng
        # [!code highlight:4]
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
        )

        # create a single column parallel linear layer
        # [!code highlight:8]
        config = ModelParallelConfig(use_cpu_initialization=use_cpu_initialization)
        layer = ColumnParallelLinear(
            input_size=768,
            output_size=768,
            config=config,
            init_method=init_method_normal(0.2),
            gather_output=False,
        ).to(device)
        logger.info(f"layer.weight shape: {layer.weight.shape}")

        seq_len, batch_size, dim = 128, 32, 768
        x = torch.randn(seq_len, batch_size, dim, device=device)
        dist.broadcast(x, src=0) # broadcast the input to all devices

        y, y_bias = layer(x)

        logger.info(f"y shape: {y.shape}")
        logger.info(f"y_bias is None: {y_bias is None}")

    except Exception as e:
        logger.info(e)

    # destroy process group (finalize)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()

if __name__ == "__main__":
    main(backend="gloo", tp_size=2)
```

You can run this example with `torchrun`:

```bash
torchrun --nproc-per-node=2 00_column_parallel_linear.py
```

All the specifics about Megatron-Core are highlighted, the rest is just the usual PyTorch distributed setup. Since I only have a single GPU available on my machine, I use the `gloo` backend, which allows me to run the example on CPU and emulate TP. You can also run it with the `nccl` backend if you have multiple GPUs available. When running this, you should get this kind of output:

```bash
2026-05-13 00:07:09,565 | INFO | rank=1 | layer.weight shape: torch.Size([384, 768])
2026-05-13 00:07:09,565 | INFO | rank=0 | layer.weight shape: torch.Size([384, 768])
2026-05-13 00:07:09,605 | INFO | rank=0 | y shape: torch.Size([128, 32, 384])
2026-05-13 00:07:09,605 | INFO | rank=1 | y shape: torch.Size([128, 32, 384])
2026-05-13 00:07:09,605 | INFO | rank=1 | y_bias is None: True
2026-05-13 00:07:09,605 | INFO | rank=0 | y_bias is None: True
```

As you can see, the weight shape is `[384, 768]` instead of `[768, 768]`, which means that the weights are split across the two devices (column-wise). Each device has half of the output dimension. Note that the weight matrix is transposed inside the `ColumnParallelLinear` layer, and that's why the shape is `[384, 768]` instead of `[768, 384]`.

The output shape of `y` is `[128, 32, 384]` instead of `[128, 32, 768]`, which means that the output is split across the two devices (column-wise) as well. This is expected becaause we set `gather_output=False` in the layer. The bias is `None` because we didn't specify a bias in the layer.

We could go even further: compute a loss, perform a backward pass, and compare the gradients with respect to a non-parallel linear layer.

Let's start by adding a dummy loss and log the gradients shape.

```python
loss = y.pow(2).sum() / (
    seq_len * batch_size * dim
)  # scale to match non-parallel loss magnitude
loss.backward()
logger.info(f"layer.weight.grad: {layer.weight.grad.shape}")
```

You should get this output:

```bash
2026-05-13 00:07:09,650 | INFO | rank=0 | layer.weight.grad: torch.Size([384, 768])
2026-05-13 00:07:09,650 | INFO | rank=1 | layer.weight.grad: torch.Size([384, 768])
```

The gradient shape is the same as the weight shape as expected. Now, let's gather ourselves all the gradients and weights across devices.

```python
dist.barrier()
grad_list = [torch.empty_like(layer.weight.grad) for _ in range(tp_size)]
dist.all_gather(
    tensor_list=grad_list,
    tensor=layer.weight.grad,
)
grad_list = torch.concat(grad_list, dim=0)
logger.info(f"grad_list shape: {grad_list.shape}")

weight_list = [torch.empty_like(layer.weight.data) for _ in range(tp_size)]
dist.all_gather(tensor_list=weight_list, tensor=layer.weight.data)
weight_list = torch.concat(weight_list, dim=0)
```

It should add the following lines to the output:

```bash
2026-05-13 00:07:09,654 | INFO | rank=0 | grad_list shape: torch.Size([768, 768])
2026-05-13 00:07:09,654 | INFO | rank=1 | grad_list shape: torch.Size([768, 768])
```

This means that each device has gathered the full gradient and weight matrices. We will use those weights to create a non-parallel linear layer on the first device, and compare the gradients.

```python
if rank == 0:
    reference_layer = torch.nn.Linear(768, 768, bias=False).to(device)
    reference_layer.weight.data.copy_(weight_list)
    reference_layer.weight.requires_grad = True
    y_ref = reference_layer(x)
    loss_ref = y_ref.pow(2).mean()
    loss_ref.backward()

    logger.info(
        f"Difference between gathered gradients and reference gradient: {(grad_list - reference_layer.weight.grad).abs().max()}"
    )
```

You should get this output:

```bash
2026-05-13 00:07:09,763 | INFO | rank=0 | Difference between gathered gradients and reference gradient: 0.0
```

Yay ! This means that the gradients we got from the column parallel linear layer are the same as the gradients we got from the non-parallel linear layer.

The full code for this example is available [here](https://github.com/bstaber/the-missing-layer/blob/main/python/tml/posts/megatron/00_column_parallel_linear.py).

## Row parallel linear layer

Coming soon!

## Column + row parallel linear layer

Coming soon!

## Tensor parallel transformer block

Coming soon!

## Adding sequence parallelism

Coming soon!
