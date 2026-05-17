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

I wanted to learn more about Megatron-LM and Megatron-Core, two libraries developed by NVIDIA for training large language models with different parallelism strategies. Megatron-LM provides several high-level APIs for training well known architectures like GPT or Llama. Megatron-Core, on the other hand, looks more like a lower-level library with more flexibility. I decided to try implementing a simple transformer block using Megatron-Core.

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

Megatron-Core also provides a `RowParallelLinear` layer that splits the weights across the row dimension. It takes care of scattering the input across devices and optionally all-reducing the output as well. Here's how we can create a row parallel linear layer:

```python
from megatron.core import ModelParallelConfig
from megatron.core.tensor_parallel import RowParallelLinear
from megatron.core.utils import init_method_normal

config = ModelParallelConfig()
layer = RowParallelLinear(
    input_size=768,
    output_size=768,
    config=config,
    init_method=init_method_normal(0.2),
    bias=False,
    input_is_parallel=False,
    skip_bias_add=True,  # if bias exists, the layer does not add it immediately; it returns bias separately so caller can fuse operations
).to(device)
```

Similarly to the `ColumnParallelLinear` layer, we need to specify the model parallel configuration and the weight initialization method. The `input_is_parallel` argument indicates whether the input is already split across devices (column-wise) or not. For instance, if the input comes from a `ColumnParallelLinear` layer with `gather_output=False`, then the input is already parallel and we can set `input_is_parallel=True` to avoid extra communication. The `skip_bias_add` argument indicates whether the layer should add the bias to the output or not. If `skip_bias_add=True`, the layer returns the bias separately, which can be useful for fusing operations like in MHA.

Besides that, the usage is very similar. Here's a full working example that you can run with `torchrun`:

```python
import logging
import os
import sys
from typing import Literal

import torch
import torch.distributed as dist
# [!code highlight:4]
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

            # [!code highlight:1]
            model_parallel_cuda_manual_seed(1234)
        else:
            device = torch.device("cpu")
            torch.manual_seed(1234)

        # [!code highlight:4]
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
        )

        # [!code highlight:10]
        config = ModelParallelConfig(use_cpu_initialization=use_cpu_initialization)
        layer = RowParallelLinear(
            input_size=768,
            output_size=768,
            config=config,
            init_method=init_method_normal(0.2),
            bias=False,
            input_is_parallel=False,
            skip_bias_add=True,
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
    main(backend="gloo", tp_size=2)
```

This script creates a row parallel linear layer, performs a forward and backward pass, gathers the gradients and weights across devices, and compares the gradients with a non-parallel linear layer. The highlighted lines show the key parts of the code that rely on Megatron-Core. You can run it with:

```bash
torchrun --nproc-per-node=2 01_row_parallel_linear.py
```

I get these outputs:

```bash
2026-05-17 14:28:09,063 | INFO | rank=0 | Process group initialized
2026-05-17 14:28:09,063 | INFO | rank=1 | Process group initialized
2026-05-17 14:28:09,063 | INFO | rank=1 | Local rank: 1
2026-05-17 14:28:09,064 | INFO | rank=0 | Local rank: 0
2026-05-17 14:28:09,124 | INFO | rank=0 | layer.weight shape: torch.Size([768, 384])
2026-05-17 14:28:09,124 | INFO | rank=0 | Creating dummy input and running forward pass
2026-05-17 14:28:09,124 | INFO | rank=1 | layer.weight shape: torch.Size([768, 384])
2026-05-17 14:28:09,124 | INFO | rank=1 | Creating dummy input and running forward pass
2026-05-17 14:28:09,180 | INFO | rank=1 | y shape: torch.Size([128, 32, 768])
2026-05-17 14:28:09,180 | INFO | rank=0 | y shape: torch.Size([128, 32, 768])
2026-05-17 14:28:09,180 | INFO | rank=0 | y_bias is None: True
2026-05-17 14:28:09,180 | INFO | rank=1 | y_bias is None: True
2026-05-17 14:28:09,263 | INFO | rank=1 | layer.weight.grad: torch.Size([768, 384])
2026-05-17 14:28:09,263 | INFO | rank=0 | layer.weight.grad: torch.Size([768, 384])
2026-05-17 14:28:09,264 | INFO | rank=1 | Gathering gradients from all TP ranks for verification
2026-05-17 14:28:09,264 | INFO | rank=0 | Gathering gradients from all TP ranks for verification
2026-05-17 14:28:09,266 | INFO | rank=1 | grad_list shape: torch.Size([768, 768])
2026-05-17 14:28:09,266 | INFO | rank=1 | Gather layer weights from all TP ranks for verification
2026-05-17 14:28:09,266 | INFO | rank=0 | grad_list shape: torch.Size([768, 768])
2026-05-17 14:28:09,266 | INFO | rank=0 | Gather layer weights from all TP ranks for verification
2026-05-17 14:28:09,268 | INFO | rank=0 | Create a reference gradient by running the same forward and backward pass on a non-parallel layer
2026-05-17 14:28:09,367 | INFO | rank=0 | reference_layer.weight.grad shape: torch.Size([768, 768])
2026-05-17 14:28:09,368 | INFO | rank=0 | Difference between gathered gradients and reference gradient: 6.984919309616089e-10
```

Everything seems to be working as planned. The weight shape is `[768, 384]` instead of `[768, 768]`, which means that the weights are split row-wise (remember that the weight matrix is transposed inside the layer). The output shape of `y` is `[128, 32, 768]` instead of `[128, 32, 384]`, which means that the output is not split across devices, as expected since `RowParallelLinear` ends with an all-reduce. The gradient shape is the same as the weight shape as expected. Finally, the difference between the gathered gradients and the reference gradient is very close to zero, which means that the row parallel linear layer is computing the correct gradients.

The full code for this example is also available [here](https://github.com/bstaber/the-missing-layer/blob/main/python/tml/posts/megatron/01_row_parallel_linear.py).

## Column + row parallel linear layer

Now that we know how to create column and row parallel linear layers, we can combine them to create the kind of projections we need in the MHA and FFN of a transformer block. But before that, let's see how we can create a simple linear layer that is both column and row parallel. Let's create a column parallel linear layer with `gather_output=False`, followed by a row parallel linear layer with `input_is_parallel=True` and `skip_bias_add=True`.

```python
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
```

Then, we can perform a forward pass through both layers:

```python
seq_len, batch_size, dim = 128, 32, 768
x = torch.randn(seq_len, batch_size, dim, device=device)

y, y_bias = layer_column_parallel(x)
y, y_bias = layer_row_parallel(y)
```

The rest is the same as before: define the model parallel configuration, initialize the process group, set the random seed, etc. Here's an example of a full script that combines both layers:

```python
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
    main(backend="gloo", tp_size=2)
```

When I run this with `torchrun --nproc-per-node=2`, I get the following output:

```bash
2026-05-17 14:47:53,936 | INFO | rank=1 | Process group initialized
2026-05-17 14:47:53,936 | INFO | rank=0 | Process group initialized
2026-05-17 14:47:53,936 | INFO | rank=0 | Local rank: 0
2026-05-17 14:47:53,936 | INFO | rank=1 | Local rank: 1
2026-05-17 14:47:54,091 | INFO | rank=1 | y shape: torch.Size([128, 32, 768])
2026-05-17 14:47:54,091 | INFO | rank=0 | y shape: torch.Size([128, 32, 768])
2026-05-17 14:47:54,092 | INFO | rank=0 | y_bias is None: True
2026-05-17 14:47:54,092 | INFO | rank=1 | y_bias is None: True
```

Everything seems to be working as expected. The output shape of `y` is `[128, 32, 768]`, which means that the output is not split across devices, as expected since the row parallel linear layer ends with an all-reduce!

## Tensor parallel transformer block

We know have the main ingredients to build a tensor parallel transformer block. Let's a focus on a simple transformer block made of a MHA and a FFN, and implement a `TensorParallelAttentionBlock` class that combines our `TensorParallelMultiHeadAttention` and `TensorParallelFFN` classes. The final architecture will be quite similar to a standard transformer block:

```python
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

        # [!code highlight:5]
        self.mha = TensorParallelMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            parallel_config=parallel_config,
        )

        # [!code highlight:6]
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
```

Compared to a standard block, we need to pass the model parallel configuration, the rest is pretty much the same.

### Tensor parallel multi-head attention

The tensor parallel multi-head attention essentially needs to replace the linear layers in the query, key, value, and output projections with column and row parallel linear layers. The attention computation itself remains the same.

```python
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

        # [!code highlight:1]
        self.local_num_heads = num_heads // parallel_config.tensor_model_parallel_size
        self.head_dim = d_model // num_heads

        # [!code highlight:8]
        self.q_proj = ColumnParallelLinear(
            input_size=d_model,
            output_size=d_model,
            config=parallel_config,
            init_method=init_method_normal(0.02),
            gather_output=False,
            bias=False,
        )

        # [!code highlight:8]
        self.k_proj = ColumnParallelLinear(
            input_size=d_model,
            output_size=d_model,
            config=parallel_config,
            init_method=init_method_normal(0.02),
            gather_output=False,
            bias=False,
        )

        # [!code highlight:8]
        self.v_proj = ColumnParallelLinear(
            input_size=d_model,
            output_size=d_model,
            config=parallel_config,
            init_method=init_method_normal(0.02),
            gather_output=False,
            bias=False,
        )

        # [!code highlight:9]
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
```

Let's go through the implementation details.

- We first compute the local number of heads by dividing the global number of heads by the tensor model parallel size. This is because each device will only compute a subset of the attention heads.
- We then create the query, key, and value projections using `ColumnParallelLinear` layers with `gather_output=False`. This means that the output of these projections will be split across devices (column-wise), which is what we want for the attention computation.
- Finally, we create the output projection using a `RowParallelLinear` layer with `input_is_parallel=True` and `skip_bias_add=True`. This means that the input to this layer is already split across devices (column-wise), and the layer will perform an all-reduce at the end to gather the output across devices.

The main trick here is to remember to set the `gather_output` and `input_is_parallel` arguments correctly to avoid unnecessary communication, and to be aware that using tensor parallelism in MHA corresponds to splitting the attention heads across devices.

### Tensor parallel feed-forward network

Now that we've done this, implementing a tensor parallel FFN is quite straightforward. We can basically do the same thing: use a `ColumnParallelLinear` layer for the up projection and a `RowParallelLinear` layer for the down projection.

```python
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
        # [!code highlight:8]
        self.up_proj = ColumnParallelLinear(
            input_size=in_features,
            output_size=hidden_dim,
            init_method=init_method_normal(0.02),
            bias=False,
            gather_output=False,
            config=parallel_config,
        )

        # [!code highlight:8]
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
```

### Example of usage

That's it. Here's an example of usage:

```python
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
    main(backend="gloo", tp_size=2)
```

I get the following output:

```bash
2026-05-17 15:05:11,081 | INFO | rank=0 | Process group initialized
2026-05-17 15:05:11,081 | INFO | rank=1 | Process group initialized
2026-05-17 15:05:11,563 | INFO | rank=1 | Output shape: torch.Size([32, 128, 768])
2026-05-17 15:05:11,563 | INFO | rank=0 | Output shape: torch.Size([32, 128, 768])
```

So at the end, the output shape seems correct. The output is not split across devices because the row parallel linear layers end with an all-reduce.

## Adding sequence parallelism

It is also possible to add sequence parallelism to the attention block. It's useful because there are components in the tranformer block that can be parallelized across the sequence dimension. If you look at the layer normalization layers, they are applied independently to each token, which means that we can split the sequence across devices and perform the normalization in parallel. This is very well explained in the [Ultra-Scale playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=sequence_parallelism). Megatron-Core model parallel configuration has a key `sequence_parallel` which seems to enable this kind of parallelism. I haven't tried yet and I'll leave it for a future experiment.
