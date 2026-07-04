---
author: Brian Staber
pubDatetime: 2026-06-29T21:25:52Z
modDatetime: 2026-06-29T21:25:52Z
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

# Introduction

The goal of this note is to learn more about Triton by implementing a kernel for Transolver's physics-attention. At the time of writing, I barely know CUDA or Triton, and I've only come across a few Triton kernels in the context of scientific machine learning.

My understanding of Triton is that it's a higher-level programming model for writing GPU kernels. Given an algorithm that we want to parallelize on a GPU, we could implement it either as a CUDA kernel or as a Triton kernel. CUDA is thread-centric: we describe what each thread should compute and organize threads into blocks. Triton, on the other hand, is program-centric (or tile-centric): we think in terms of how to partition the computation into tiles and how each tile should be processed. In practice, a Triton program instance typically computes an entire tile rather than a single output element. For instance, in a vector addition kernel, a CUDA kernel would describe how to compute one output element per thread, while a Triton kernel would describe how to compute an entire tile of output elements per program instance.

Both approaches target the same GPU execution model. Whether we write CUDA or Triton, the generated code is eventually executed by the same hardware: streaming multiprocessors (SMs), warps, registers, shared memory, caches, and so on. Triton therefore provides a higher-level way of expressing GPU programs, but it does not completely remove the need to understand how GPUs execute code. I feel like that understanding the execution model remains essential for writing efficient Triton kernels, even if many low-level details are handled by the compiler.

## Basics about GPU architecture

### Streaming Multiprocessors

I think it's useful to know a few basics about GPU architecture before using Triton. Your GPU is made of many **Streaming Multiprocessors** (SMs) that can run many threads simultaneously. Each SM has some compute units, including:

- CUDA cores that take care of ordinary arithmetic instructions like `a + b`, `a * b`, etc.
- Tensor cores that are specialized in things like matrix multiplications (`A @ B`), GEMM, and other linear algebra operations.
- RT Cores which accelerate ray tracing operations. Won't be used by Triton.

The SM also contains one or more warp schedulers. A **warp** is a group of 32 threads that execute the same instruction simultaneously. The warp scheduler decides which ready warp to execute next, allowing to hide memory latency by switching between warps.

Each SM also contains fast on-chip memory:

- **Registers** which are private to each thread and are the fastest memory on the GPU
- **Shared memory** which is shared by all threads belonging to the same program

All SMs share a larger **L2 cache**, which sits in front of the GPU's global memory (**DRAM**), where tensors are ultimately stored.

### Example

On my consumer GPU (RTX 4080 Super Ada Lovelace):

- There are **80 Streaming Multiprocessors (SMs)**
- Each SM has **65,536 32-bit registers** (256 KB of register storage)
- Each SM provides up to **101,376 bytes (~99 KB)** of shared memory
- All SMs share a **64 MB L2 cache**
- The GPU has **16 GB of GDDR6X DRAM** (global memory)

You can check that with `nvidia-smi -q`, `torch.cuda.get_device_properties(0)` or with Triton helpers functions.

### Triton programs

A Triton program is Triton's unit of work, analogous to a CUDA thread block. At runtime, each program is scheduled onto an SM, where it is executed by warps of GPU threads. **A Triton kernel is itself a collection of programs**. The number of programs is determined when launching the kernel (the grid). Each program processes a **tile** of data. The size of this tile is controlled by one or more compile-time parameters such as `BLOCK_SIZE`.

If you're new to CUDA and Triton, I feel that some terminology can be confusing. For instance, both have the notion of grid. In CUDA, the grid is a collection of blocks, and each block is a collection of threads. In Triton, the grid is a collection of programs, and each program is responsible for computing one tile of the computation (typically a tile of the output tensor).

### Let's wrap it up

- **What is a SM**: An execution unit of the GPU. Each SM contains compute units (CUDA Cores, Tensor Cores, ...), warp schedulers, registers, and shared memory. A Triton program is scheduled as a unit onto one SM.
- **What is a warp**: A group of 32 threads that execute the same instruction simultaneously
- **Where do tensors live**: Tensors are stored in global memory (DRAM). During execution, their data may be cached in L2 cache or copied into shared memory and registers for faster access.
- **What is a Triton program**: A Triton program is Triton's unit of work. One program typically computes one tile of the output. Programs are independent and are scheduled onto SMs by the GPU runtime.
- **Why can multiple programs run on the same SM**: An SM has enough registers and shared memory to host several programs simultaneously. While one program is waiting for data from memory, the SM can execute warps from another ready program, helping hide memory latency.

### Triton tutorials

First tutorial: I was confused by the mask. It's useful because the last because n_elements usually is not an exact multiple of BLOCK_SIZE. So the last program might have a block that goes beyond the size of the arrays. Example: triton.cdiv(98432, 1024) = 97, and so the last program has block_start = 96 * 1024 = 98304 and offsets = 98304 + [0, 1, ..., 1023]. So this offset go from 98304 to 99327, but the valid indices are only: 0 to 98431. So you need the mask.