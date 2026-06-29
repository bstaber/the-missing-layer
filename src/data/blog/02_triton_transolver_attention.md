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

# Triton tutorials

First tutorial: I was confused by the mask. It's useful because the last because n_elements usually is not an exact multiple of BLOCK_SIZE. So the last program might have a block that goes beyond the size of the arrays. Example: triton.cdiv(98432, 1024) = 97, and so the last program has block_start = 96 * 1024 = 98304 and offsets = 98304 + [0, 1, ..., 1023]. So this offset go from 98304 to 99327, but the valid indices are only: 0 to 98431. So you need the mask.