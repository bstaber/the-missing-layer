---
author: Brian Staber
pubDatetime: 2026-07-11T12:16:08Z
modDatetime: 2026-07-11T12:16:08Z
title: Triton kernel for physics-based attention
slug: triton-kernel-for-physics-based-attention
featured: false
draft: false
tags:
  - attention
  - transformer
  - physics
  - triton
description: Using triton to implement Transolver's physics attention
---

<div class="not-prose my-6 rounded-lg border border-yellow-400 bg-yellow-50 px-4 py-3 text-yellow-900 dark:border-yellow-500/60 dark:bg-yellow-950/40 dark:text-yellow-100">
  <strong>Work in progress.</strong> This post is still being written and may change. Working my way through some Triton tutorials and implementing a kernel for Transolver's physics attention.
</div>

# Introduction

The goal of this note is to learn more about Triton by implementing a kernel for Transolver's physics-attention. At the time of writing, I barely know CUDA or Triton, and I've only come across a few Triton kernels in the context of scientific machine learning.

My understanding of Triton is that it's a higher-level programming model for writing GPU kernels. Given an algorithm that we want to parallelize on a GPU, we could implement it either as a CUDA kernel or as a Triton kernel. CUDA is thread-centric: we describe what each thread should compute and organize threads into blocks. Triton, on the other hand, is program-centric (or tile-centric): we think in terms of how to partition the computation into tiles and how each tile should be processed. In practice, a Triton program instance typically computes an entire tile rather than a single output element. For instance, in a vector addition kernel, a CUDA kernel would describe how to compute one output element per thread, while a Triton kernel would describe how to compute an entire tile of output elements per program instance.

Both approaches target the same GPU execution model. Whether we write CUDA or Triton, the generated code is eventually executed by the same hardware: streaming multiprocessors (SMs), warps, registers, shared memory, caches, and so on. Triton therefore provides a higher-level way of expressing GPU programs, but it does not completely remove the need to understand how GPUs execute code. I feel like that understanding the execution model remains essential for writing efficient Triton kernels, even if many low-level details are handled by the compiler.

## Basics about GPU architecture

### Streaming Multiprocessors

I think it's useful to know a few basics about GPU architecture before using Triton and at least: SMs, warps, register, and cache. Your GPU is made of many **Streaming Multiprocessors** (SMs) that can run many threads simultaneously. Each SM has some compute units, including:

- CUDA cores that take care of ordinary arithmetic instructions like `a + b`, `a * b`, etc.
- Tensor cores that are specialized in things like matrix multiplications (`A @ B`), GEMM, and other linear algebra operations.
- RT Cores which accelerate ray tracing operations. Won't be used by Triton.

The SM also contains one or more warp schedulers. A **warp** is a group of 32 threads that execute the same instruction simultaneously. The warp scheduler decides which ready warp to execute next, allowing to hide memory latency by switching between warps.

Each SM also contains fast on-chip memory:

- **Registers** which are private to each thread and are the fastest memory on the GPU
- **Shared memory** which is shared by all threads belonging to the same program

All SMs share a larger **L2 cache**, which sits in front of the GPU's global memory (**DRAM**), where tensors are ultimately stored. On my consumer GPU (RTX 4080 Super Ada Lovelace):

- There are **80 Streaming Multiprocessors (SMs)**
- Each SM has **65,536 32-bit registers** (256 KB of register storage)
- Each SM provides up to **101,376 bytes (~99 KB)** of shared memory
- All SMs share a **64 MB L2 cache**
- The GPU has **16 GB of GDDR6X DRAM** (global memory)

You can check that with `nvidia-smi -q`, `torch.cuda.get_device_properties(0)` or with Triton helpers functions.

Summary:

- **What is a SM**: An execution unit of the GPU. Each SM contains compute units (CUDA Cores, Tensor Cores, ...), warp schedulers, registers, and shared memory. A Triton program is scheduled as a unit onto one SM.
- **What is a warp**: A group of 32 threads that execute the same instruction simultaneously
- **Where do tensors live**: Tensors are stored in global memory (DRAM). During execution, their data may be cached in L2 cache or copied into shared memory and registers for faster access.
- **What is a Triton program**: A Triton program is Triton's unit of work. One program typically computes one tile of the output. Programs are independent and are scheduled onto SMs by the GPU runtime. More on that in the next section.
- **Why can multiple programs run on the same SM**: An SM has enough registers and shared memory to host several programs simultaneously. While one program is waiting for data from memory, the SM can execute warps from another ready program, helping hide memory latency.

### Kernels in Triton and CUDA

A Triton program is Triton's unit of work. It is often compared to a CUDA thread block because both are scheduled as units onto an SM, although the programming models are different. At runtime, each program is scheduled onto an SM, where it is executed by warps of GPU threads. **A Triton kernel is itself a collection of programs**. The number of programs is determined when launching the kernel (the grid). Each program processes a **tile** of data. The size of this tile is controlled by one or more compile-time parameters such as `BLOCK_SIZE`.

If you're new to CUDA and Triton, I feel that some terminology can be confusing. For instance, both have the notion of grid. In CUDA, the grid is a collection of blocks, and each block is a collection of threads. In Triton, the grid is a collection of programs, and each program is responsible for computing one tile of the computation (typically a tile of the output tensor).

Suppose that we want to compute a vector addition `C = A + B`. 

In a naive CUDA implementation, we would launch a grid of blocks, where each block contains a number of threads. Each thread would compute one element of the output vector `C`. For example, if we have 256 threads per block and 1024 elements in the vectors, we would launch 4 blocks (1024 / 256 = 4). Each thread would compute one element of `C` based on its thread index (that might be a super naive CUDA implementation though).

```markdown
Grid
├── Block 0
│   ├── Thread 0   → C[0]
│   ├── Thread 1   → C[1]
│   ├── ...
│   └── Thread 255 → C[255]
│
├── Block 1
│   ├── Thread 0   → C[256]
│   ├── Thread 1   → C[257]
│   ├── ...
│   └── Thread 255 → C[511]
│
├── Block 2
│   └── ...
│
└── Block 3
    └── ...
```

In the CUDA implementation, you would see something like this:

```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}
```

You can see that the CUDA kernel describes how each thread computes one element of `C`

In a Triton implementation, we would launch a grid of programs, where each program computes a tile of the output vector `C`. For example, if we have a `BLOCK_SIZE` of 256, each program would compute 256 elements of `C`. If we have 1024 elements in the vectors, we would launch 4 programs (1024 / 256 = 4). Each program would compute a tile of `C` based on its program index.

```markdown
Grid
└── Programs
    ├── Program 0 → computes C[0:256]
    ├── Program 1 → computes C[256:512]
    ├── Program 2 → computes C[512:768]
    └── Program 3 → computes C[768:1024]
```

In the Triton implementation, you would see something like this:

```python
@triton.jit
def vector_add_kernel(A_ptr, B_ptr, C_ptr, n_elements, BLOCK_SIZE=256):
    program_id = tl.program_id(0)

    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    
    C = A + B
    tl.store(C_ptr + offsets, C, mask=mask)
```

Here, `program_id` tells us which program instance we are in. Instead of computing one scalar index `i`, the program computes a whole vector of indices offsets, corresponding to a tile of `C`. Notice that there is no `threadIdx` or `blockIdx` in Triton

Notice that both kernels perform exactly the same computation and process the same chunks of the vector. The difference is the level of abstraction. In CUDA, we explicitly describe how individual threads cooperate to process those chunks. In Triton, we directly describe how one program processes a chunk (or tile), leaving the compiler to map that computation onto GPU threads and warps.

# Triton tutorials

There are several tutorials available on the Triton [website](https://triton-lang.org/main/getting-started/tutorials/index.html). It starts with a gentle vector addition, but it quickly jumps to fused softmax and matrix multiplication which I found not that easy for a beginner. I decided to introduce some intermediate examples such as copying a tensor and computing a dot product.

## Copy a tensor

Making a kernel that copies a tensor essentially essentially consists in loading a tensor and copy it back into some output tensor.

Here's what this simple example tries to achieve:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(
    x_ptr,
    output_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    ...

def copy_tensor(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    copy_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)

    return output

if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)

    size = 98432
    x = torch.rand(size, device=DEVICE)

    output_torch = x.clone()
    output_triton = copy_tensor(x)

    print(torch.allclose(output_torch, output_triton))
```

Let's go through the code:

- The function `copy_kernel` decorated with `@triton.jit` is our triton kernel. We will define in this function how each chunk of the input tensor is copied into the output tensor (recall that triton is tile-centric). The kernel needs to get the pointers of the arrays it's going to work with. Here, we need the pointer of the input array, `x_ptr`, and the pointer of the output array, `output_ptr`.

- The `copy_tensor` function is a helper function that launches our kernel over a grid of blocks. We define a one-dimensional grid thanks to a tuple with a single element: ```(triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)```. In our example, `n_elements = 98432` is not a multiple of `BLOCK_SIZE = 1024`, so we get a grid with `97` blocks which corresponds to `99328`, or in other terms, the last block has `896` elements that we don't need (`97 * 1024 - 98432 = 896`). We will deal these extra elements inside the kernel function. The grid is defined as a lambda function that can access all the input arguments given to the kernel through a dict `meta`. Finally, using this grid, we can launch the kernel function. Note that although we pass the PyTorch tensors `x` and `output`, Triton automatically extracts the underlying device pointers from the PyTorch tensors.

- In the main body, we basically get the device on which we're running, apply our `copy_tensor` function, and check that it works as intended.

Let's implement the kernel function. The main idea is the following. Each program launched by the our one-dimensional grid will copy one chunk of the input tensor into the output tensor. We define the chunk being processed by the program in terms of the program id and the chosen block size. For instance, the 'first' program in the grid will process the elements `0, ..., 1023`, the second program will process `1024, ..., 2047`, the k-th program will process the elements `k * 1024, ..., (k + 1) * 1024 - 1`, and so on.

```python
@triton.jit
def copy_kernel(
    x_ptr,
    output_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask)
    tl.store(output_ptr + offsets, x, mask)
```

As mentioned earlier, the last program is assigned a block of 1024 elements, but only the first `128` correspond to valid tensor elements. The remaining `896` are masked out. In order to avoid reading or writing past the end of the tensor, we define a mask that can be passed to triton's `load` and `store` functions. Every program executes exactly the same code. The only thing that changes from one program to another is its `program_id`.

That's it for this simple copy kernel function.

## Dot product

The second example I'd like to go through is the dot product. In Triton's official documentation they start with the [vector addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) which is quite similar. As in the previous example, we need to implement a Triton kernel, a `dot_product` helper function and check that it works as intended. The script does take the form:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def dot_product_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    ...

def dot_product(
    x: torch.Tensor,
    y: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    assert x.ndim == 1 and y.ndim == 1
    assert x.shape == y.shape
    assert x.device == y.device
    assert x.dtype == y.dtype
    assert x.is_contiguous() and y.is_contiguous()
    assert x.dtype == torch.float32

    n_elements = x.numel()
    num_programs = triton.cdiv(n_elements, block_size)
    partial_dot_products = torch.empty(num_programs, dtype=x.dtype, device=x.device)

    grid = (num_programs,)

    dot_product_kernel[grid](
        x,
        y,
        partials,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    
    return partial_dot_products.sum()
```

In the kernel function, we essentially need to load the appropriate chunks of `x` and `y`, and compute their dot product. Hence, each entry in the output tensor will contain the dot products between the individual chunks of `x` and `y`. That's why we still need to return `output.sum()` in our helper function, and why the output tensor's length is equal to the number of programs. In other words, this implementation uses Triton for the first reduction stage and PyTorch for the final reduction.
```python
@triton.jit
def dot_product_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask, other=0.0)

    dot_product = tl.sum(x * y, axis=0)

    tl.store(output_ptr + pid, dot_product)
```

Most of the kernel consists in loading `x` and `y`. Then we can Triton's built-in `sum` function to compute the dot product between the chunks of `x` and `y`. Finally, we store the scalar result into the output tensor using the simple offset given by the program id. This function is not a complete GPU-only dot-product implementation:

- Each Triton program reduces one tile into one partial scalar.
- The partial scalars are stored in GPU memory.
- `partial_dot_products.sum()` launches a PyTorch reduction to obtain the final scalar.

A later extension could implement the second reduction in Triton or use an atomic addition, although atomics introduce their own performance and numerical considerations

<div class="not-prose my-6 rounded-lg border border-yellow-400 bg-yellow-50 px-4 py-3 text-yellow-900 dark:border-yellow-500/60 dark:bg-yellow-950/40 dark:text-yellow-100">
  Work in progress: we will dive into fused softmax and matrix multiplication soon, and then implement Transolver's physics attention kernel.
</div>