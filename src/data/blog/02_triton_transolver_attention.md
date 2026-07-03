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

The goal of this note ...

## Basics about GPU architecture

### Streaming Multiprocessors

I think it's useful to know a few basics about GPU architecture before using Triton. Your GPU is made of many **Streaming Multiprocessors** (SMs) that can run many threads simultaneously. Each SM has some compute units, including:

- CUDA cores that take care of ordinary arithmetic instructions like `a + b`, `a * b`, etc.
- Tensor cores that are specialized for matrix multiplications (`A @ B`), GEMM, and other linear algebra operations.
- RT Cores which accelerate ray tracing operations. Won't be used by Triton.

The SM also contains one or more warp schedulers. A **warp** is a group of 32 threads that execute the same instruction simultaneously. The warp scheduler decides which ready warp to execute next, allowing to hide memory latency by switching between warps.

Each SM also contains fast on-chip memory:

- **Registers** which are private to each thread and are the fastest memory on the GPU
- **Shared memory** which is shared by the threads of a program running on the SM

All SMs share a larger **L2 cache**, which sits in front of the GPU's global memory (**DRAM**), where tensors are ultimately stored.

### Example

On my consumer GPU (RTX 4080 Super):

- There are **80 Streaming Multiprocessors (SMs)**
- Each SM has **65,536 32-bit registers** (256 KB of register storage)
- Each SM provides up to **101,376 bytes (~99 KB)** of shared memory
- All SMs share a **64 MB L2 cache**
- The GPU has **16 GB of GDDR6X DRAM** (global memory)

You can check that with `nvidia-smi -q`, `torch.cuda.get_device_properties(0)` or with Triton helpers functions.

### Triton programs

A Triton **program** executes on a single SM, and a SM can host several Triton programs depending on the available resources such as **registers** and **shared memory**. **A Triton kernel is itself a collection of programs**. The number of programs is determined when launching the kernel (the grid). Each program typically processes a **block** of data. The size of this block is controlled by one or more compile-time parameters such as `BLOCK_SIZE`.


### Let's wrap it up

- **What is a SM**: An execution unit of the GPU. Each SM contains compute units (CUDA Cores, Tensor Cores, ...), warp schedulers, registers, and shared memory. A Triton program executes on a single SM.
- **What is a warp**: A group of 32 threads that execute the same instruction simultaneously
- **Where do tensors live**: Tensors are stored in global memory (DRAM). During execution, their data may be cached in L2 cache or copied into shared memory and registers for faster access.
- **What is a Triton program**: A Triton program is the basic unit of execution in Triton. A Triton kernel is a collection of independent programs. The number of programs is determined when launching the kernel (the grid).
- **Why can multiple programs run on the same SM**: An SM has enough registers and shared memory to host several programs simultaneously. While one program is waiting for data from memory, the SM can execute warps from another ready program, helping hide memory latency.

## Triton tutorials

First tutorial: I was confused by the mask. It's useful because the last because n_elements usually is not an exact multiple of BLOCK_SIZE. So the last program might have a block that goes beyond the size of the arrays. Example: triton.cdiv(98432, 1024) = 97, and so the last program has block_start = 96 * 1024 = 98304 and offsets = 98304 + [0, 1, ..., 1023]. So this offset go from 98304 to 99327, but the valid indices are only: 0 to 98431. So you need the mask.