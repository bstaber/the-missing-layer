---
author: Brian Staber
pubDatetime: 2026-07-20T21:19:45Z
modDatetime: 2026-07-20T21:19:45Z
title: Implementing a super tiny squeezy TabICL
slug: tiny-tabicl
featured: false
draft: false
tags:
  - tabular foundation models
  - tabicl
  - transformer
description: Understanding the building blocks of TabICL v2
---

# Introduction
Tabular foundation models (TFMs) are attracting a lot of attention (no pun intended). In this post, I want to understand what happens under the hood of TabICLv2, an open-source TFM developed by the SODA team at Inria. Other models, such as TabPFN from Prior Labs, follow the same broad idea: pretrain a Transformer on millions of synthetic tabular prediction tasks, then use in-context learning to solve a new task without updating the model's weights.

At inference time, the model receives the labeled training rows as context and predicts the labels or target values of unseen rows. The model conditions on an entire training set, but it does not run gradient descent or fit new parameters for that dataset. TabICLv2 supports classification and regression and handles numerical and categorical features, missing values, and outliers.

My goal here is not to reproduce the full system. I will implement a small version of its three-stage architecture: a column encoder, a row encoder, and a dataset-wise Transformer that performs in-context learning. I will then train it on a controlled synthetic prior and test whether it actually learns to make better predictions as the context set grows.

I used the official [NanoTabICL](https://github.com/soda-inria/nanotabicl/) and [TabICL](https://github.com/soda-inria/tabicl) implementations to understand what's going on. 

# Architecture

I think that the architecture can be divided into the following main steps:

1. Embedding step: it includes normalization, feature grouping, linear embedding of the inputs and training outputs for ICL
2. Column attention: several blocks of self-attentions with induced points applied to the columns
3. Row attention: several blocks of self-attentions applied to the rows (concatenated with extra CLS tokens)
4. ICL blocks: several self-attention blocks again but those perform ICL
5. Final output projection

There are additional details such as RoPE, multiple CLS tokens, QASSMax (a scalable softmax).

It was follows, I will focus on regression.

## Feature grouping

After normalization, feature grouping is performed. Before this step, we have a tensor `x` with shape `x.shape = (batch, rows, columns)`. Feature grouping adds a new dimension: `(batch, rows, columns, feature_group_size)`. Each scalar feature value is replaced by a small vector containing values gathered from several columns of the same row.

For instance, assume that `x.shape = (1, 100, 5)` and `feature_group_size = 3`. Let `[x_0, x_1, x_2, x_3, x_4]` be a given row. Feature grouping produces

$$
\begin{bmatrix}
x_0 & x_1 & x_3\\
x_1 & x_2 & x_4\\
x_2 & x_3 & x_0\\
x_3 & x_4 & x_1\\
x_4 & x_0 & x_2\\
\end{bmatrix}
$$

Each column is represented by a vector of three values. Thanks to this, the first linear embedding layer receives information from multiple columns instead of embedding every column independently.

For a group size $n_g$, the offsets are $o_i = 2^i - 1$ for $i = 0, \dots, n_g - 1$. In our case, with $n_g = 3$, this gives three offsets $[0, 1, 3]$. For each offset $o$, the column indices are computed as `(arange(n_cols) + offset) % n_cols`. In our case, this gives:

- Offset $0$: `(arange(5) + 0) % 5` -> `[0, 1, 2, 3, 4]`
- Offset $1$: `(arange(5) + 1) % 5` -> `[1, 2, 3, 4, 0]`
- Offset $2$: `(arange(5) + 2) % 5` -> `[3, 4, 0, 1, 2]`.

By stacking the three slices `x[..., [0, 1, 2, 3, 4]]`, `x[..., [1, 2, 3, 4, 0]]`, and `x[..., [3, 4, 0, 1, 2]]`, you get the final tensor with shape `(batch, rows, columns, 3)`.

After feature grouping, a linear layer projects each feature group into the model’s embedding space. This operation is applied to the entire input tensor x, which contains both training and test rows. The labels y, which are available only for the training rows, are embedded separately and added to the corresponding training-row embeddings. The test-row embeddings receive no label information.

## Column attention with inducing points

At this stage, x has shape `x.shape = (batch, rows, columns, d_model)` where `d_model` is the embedding dimension of the model. I'm going to use the same name `x` at each step of the architecture by the way. 

The aim of column attention is to apply attention in each column independently. In tabular problems, the number of rows can be large and performing vanilla attention would be quadratic in the number of rows. To circumvent this issue, the authors leverage self-attention with inducing points. 

It consists in introducing $M << n_\mathrm{rows}$ learnable latent queries of dimensions $d_model$. It works in two steps: compression and decompression. 

The input tensor is first reshaped into `(batch * columns, rows, d_model)` so that each column of each dataset becomes an independent sequence. Attention is then performed across the rows within each sequence.

During the compression step, the $M$ inducing points act as querieswhile the row representations act as keys and values:

$$
H = \mathrm{Attention}(I, X, X)
$$

where $I$ denotes the inducing points. This produces a compressed representation with shape `(batch * columns, M, d_model)`. 

During decompression, the original row representations act as queries, while the compressed representations act as keys and values:

$$
X' = \mathrm{Attention}(X, H, H)
$$

The result has shape `(batch * columns, rows, d_model)` and is rearranged back to `(batch, rows, columns, d_model)`.

<div class="not-prose my-6 rounded-lg border border-yellow-400 bg-yellow-50 px-4 py-3 text-yellow-900 dark:border-yellow-500/60 dark:bg-yellow-950/40 dark:text-yellow-100">
  <strong>Side note.</strong> I think that this kind of block is also called ISAB (Induced Set Attention Block), introduced the 
  <a
  href="https://proceedings.mlr.press/v97/lee19d/lee19d.pdf"
  target="_blank"
  rel="noopener noreferrer"
  class="underline"
>
  Set Transformer paper.
</a>
   The learnable latent queries also exist in the 
  <a
    href="https://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf"
    target="_blank"
    rel="noopener noreferrer"
    class="underline"
  >Perceiver paper</a> architecture introduced by DeepMind. But the Perceiver block the Perceiver applies several self-attention layers in latent space while ISAB immediately projects information back to the original space.
</div>

## Row attention 

TBD.

## ICL attention

TBD.

# Implementation

TBD.

# Training this tiny TabICL with my own prior data

TBD.