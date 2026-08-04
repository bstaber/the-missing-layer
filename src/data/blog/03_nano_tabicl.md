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
Tabular foundation models (TFMs) are attracting a lot of attention (no pun intended). In this post, I want to understand what happens under the hood of [TabICLv2](https://www.alphaxiv.org/abs/2602.11139), an open-source TFM developed by the SODA team at Inria. Other models, such as TabPFN from Prior Labs, follow the same broad idea: pretrain a Transformer on millions of synthetic tabular prediction tasks, then use in-context learning to solve a new task without updating the model's weights.

At inference time, the model receives the labeled training rows as context and predicts the labels or target values of unseen rows. The model conditions on an entire training set, but it does not run gradient descent or fit new parameters for that dataset. TabICLv2 supports classification and regression and handles numerical and categorical features, missing values, and outliers.

My goal here is not to reproduce the full system. I will implement a small version of its three-stage architecture: a column encoder, a row encoder, and a dataset-wise Transformer that performs in-context learning. I will then train it on a controlled synthetic prior and test whether it actually learns to make better predictions as the context set grows.

I used the official [NanoTabICL](https://github.com/soda-inria/nanotabicl/) and [TabICL](https://github.com/soda-inria/tabicl) implementations to understand what's going on. 

# Formal definition

TabICL and TabPFN are Prior-Fitted Networks (PFNs). Prior-Fitted Networks were, to the best of my knowledge, introduced by [Transformers Can Do Bayesian Inference](https://openreview.net/forum?id=KSugKcbNf9). It was further developped in [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://openreview.net/forum?id=cp5PvcI6w8_), and [TabICL: A Tabular Foundation Model for In-Context Learning](https://arxiv.org/abs/2505.19307).

The key idea is to train a transformer on millions of synthetic datasets sampled from a prior distribution over tabular tasks. Assume that datasets are generated as 

$$
f \sim p(f), \quad \mathcal{D} = \{(x_i, y_i)\}_{i=1}^N, \quad y_i = f(x_i) + \epsilon\,.
$$

The quantity we care about when predicting a new point is the posterior predictive distribution (PPD)

$$
p(y^{\star} | x^{\star}, \mathcal{D}) = \int p(y^{\star} | x^{\star}, f) p(f | \mathcal{D}) df\,,
$$

where $p(f | \mathcal{D})$ is the posterior distribution over functions given the training data. The PPD is intractable in general, but we can approximate it with a transformer trained to predict $y^{\star}$ given $x^{\star}$ and $\mathcal{D}$. The transformer learns to perform Bayesian inference by conditioning on the training set and producing predictions for new inputs.

For more details, I recommend reading the original papers and the recent technical reports.

# Architecture

I think that the architecture can be divided into the following main steps:

1. Embedding step: it includes normalization, feature grouping, linear embedding of the inputs and training outputs for ICL
2. Column attention: several blocks of self-attentions with induced points applied to the columns
3. Row attention: several blocks of self-attentions applied to the rows (concatenated with extra CLS tokens)
4. ICL blocks: several self-attention blocks again but those perform ICL
5. Final output projection

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
  <strong>Side note.</strong> I think that this kind of block is also called ISAB (Induced Set Attention Block), introduced in the 
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
  >Perceiver paper</a> architecture introduced by DeepMind. Unlike ISAB, the Perceiver applies several self-attention layers in the latent space before optionally projecting the information back to the inputs (Perceiver IO).
</div>

## Row attention 

After several column attention blocks, the features pass through several row attention blocks. Row attention is essentially multi-head self-attention applied to each row independently. At this stage, the tensor has a shape `(batch, rows, columns, d_model)` and it is reshaped into `(batch * rows, columns, d_model), making the columns the sequence dimension.

Unlike column attention, no inducing points are used. Since tabular datasets contain far fewer columns than rows, the quadratic complexity of self-attention remains manageable. This allows the model to directly capture interactions between the features of each individual sample. After the attention and feed-forward layers, the tensor is reshaped back to `(batch, rows, columns, d_model)` before being passed to the next block.

The row attention block is thus identical to a standard Transformer encoder block (multi-head self-attention + residual connections + layer normalization + feed-forward network), except that it operates independently on each row after reshaping.

Before entering the first row attention block, four learnable `[CLS]` tokens are prepended to every row. These tokens are concatenated with the feature representations produced by the column transformer, yielding a sequence of columns + 4 tokens for each row. The row attention blocks jointly process the feature and `[CLS]` tokens, allowing the latter to aggregate information from all features through self-attention.

After the final row attention block, only the four `[CLS]` representations are kept and concatenated, producing a tensor of shape `(batch, rows, 4 * d_model)`, which is passed to the ICL transformer.

## ICL attention

After the row attention stage, every row is represented by a single embedding summarizing all its features. The goal of the ICL stage is to allow test examples to retrieve useful information from the labeled training examples.

The feature tensor has hape `(batch, rows, d_model)`. Unlike the previous attention blocks, the rows are not processed independently, instead, attention is applied across the rows themselves. And since the labels of the test rows are unknown, information should only flow from the training rows toward the test rows. In other words, the keys and values are restricted to the training rows, and during the final ICL block, only the test rows are queried.

## Addtional perks

That's it for the principal architecture: feature grouping, induced column attention, row attention, and dataset-wise ICL attention. But there are other details worth mentioning: 
- **QASSMax** (Query-Aware Scalable Softmax). Used in the column and ICL transformer blocks, QASSMax modifies the query vectors with a learnable transformation before computing attention. The transformation depends on both the context size and the query content, allowing the model to adapt the effective attention temperature and maintain sharp attention distributions when processing datasets of varying sizes.
- **Multiple row-level `[CLS]` tokens**: I mentioned this in the above sections but it's worth emphasizing again that instead of using a single `[CLS]` token to aggregate the feature representations of a row, TabICLv2 prepends four learnable `[CLS]` tokens to each row before the row transformer.
- **Labels injected twice**: I didn't really mention this clearly in the above section, I think. Training labels are incorporated at two stages of the network. They are first embedded and added to the input feature embeddings before the column and row transformers, allowing the row representations to encode feature-label relationships. The labels are then embedded again before the ICL transformer so that dataset-level attention has direct access to the supervision signal when retrieving relevant training examples.
- **Column attention reads only the training rows**: The column transformer is applied exclusively to the labeled training rows. Since this stage performs attention across the row dimension, excluding the test rows reduces computation while still allowing them to retrieve information from the encoded training set during the subsequent ICL stage. I think that this was not the case in the first version of TabICL.
- **The final ICL block is asymmetric**: The last ICL block is implemented as cross-attention rather than self-attention: only the test-row representations are used as queries, while the training-row representations provide the keys and values. Since the model only needs to produce predictions for the test rows, this avoids unnecessary computation on the training rows.

# Implementation

The aim is obtain a model that we can create as follows:

```python
model = TinyTabICL(out_dim=1, d_model=128)
```

Given a batch of datasets, the model can be called with:

```python
y_pred = model(X, y)
```

where `X` contains the train and test rows, while `y` only contain the train rows. The model uses the labeled training rows as context and returns predictions for the test rows. More precisely:

- `X`:      `(batch_size, num_train + num_test, num_features)`
- `y`:      `(batch_size, num_train)`
- `y_pred`: `(batch_size, num_test, out_dim)`

For scalar regression, we set out_dim=1. The model can be implemented as an nn.Module with the following interface:

```python
class TinyTabICL(nn.Module):
    """Tiny squeezy TabICL model."""

    def __init__(
        self,
        out_dim: int,
        d_model: int = 128,
        num_heads_col: int = 8,
        num_heads_row: int = 8,
        num_heads_icl: int = 8,
        num_col_blocks: int = 3,
        num_row_blocks: int = 3,
        num_icl_blocks: int = 3,
        num_inducing: int = 128,
        num_cls_cols: int = 4,
        feature_group_size: int = 3,
    ):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        ...
```

The main hyperparameter is d_model, which controls the dimension of the feature representations throughout the column and row transformers. For each stage (column attention, row attention, and ICL attention) we specify both the number of Transformer blocks and the number of attention heads.

The remaining parameters control architecture-specific details:
- `num_inducing` is the number of learnable inducing tokens used by the column transformer
- `num_cls_cols` is the number of learnable `[CLS]` tokens prepended to each row before row attention. 
- `feature_group_size` determines how many feature values are grouped together before the initial linear projection.

## Forward pass

I would like to go through the forward method step by step, sequentially. Let's start with this:

```python
def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batch_size, num_rows, num_cols = x.shape
    y_batch_size, num_train = y.shape

    if y_batch_size != batch_size:
        raise ValueError(
            f"Batch-size mismatch: x has {batch_size}, y has {y_batch_size}."
        )

    # input preprocessing
    ...

    # input embedding
    ...

    # column attention
    ...

    # row attention
    ...

    # ICL attention
    ...

    # output projection
    ...
```

The complete implementation is given in the sequel of this section.

### Input preprocessing

The input preprocessing step consists of normalizing the features and applying the feature-grouping strategy. I took both operations directly from the NanoTabICL implementation.

The features are standardized using statistics computed from the training rows only:

```python
x = (x - x[:, :num_train].mean(dim=1, keepdim=True)) / (
    x[:, :num_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8
)
```

The mean and standard deviation are computed independently for every dataset and every feature.

We then apply feature grouping:

```python
idxs = torch.arange(num_cols, dtype=torch.long, device=x.device)
x = torch.stack(
    [
        x[:, :, (idxs + (2**i - 1)) % num_cols]
        for i in range(self.feature_group_size)
    ],
    dim=-1,
)
```

Before this operation, x has shape `(batch_size, num_rows, num_cols)`. After feature grouping, its shape becomes `(batch_size, num_rows, num_cols, feature_group_size)`.

### Input embedding

Each feature group is then projected into the model embedding space:

```python
x_proj = self.proj_x(x)
```

The resulting tensor has shape `(batch, rows, columns, d_model)`. The labels are embedded separately and added only to the training-row representations:

```python
x_proj[:, :num_train] += self.proj_y(y[:, :, None, None])
```

The two singleton dimensions are added so that the label embedding can be broadcast across all columns of the corresponding row. The test rows receive no label information.

### Column attention

Column attention processes each column as an independent sequence over the rows. We therefore move the column dimension next to the batch dimension and merge both dimensions:

```python
x_proj = x_proj.permute(0, 2, 1, 3).reshape(
    batch_size * num_cols,
    num_rows,
    -1,
)
```

The shape changes from `(batch, rows, columns, d_model)` to `(batch * columns, rows, d_model)`. The induced Transformer blocks are then applied:

```python
for block in self.col_blocks:
    train_context = x_proj[:, :num_train]
    x_proj = block(x_proj, train_context)
```

All rows are used as queries, but the inducing tokens summarize only the training rows. Consequently, the test rows can retrieve information from the training context without contributing to it.

After the column blocks, we restore the original table layout:

```python
x_proj = x_proj.reshape(
    batch_size,
    num_cols,
    num_rows,
    -1,
).permute(0, 2, 1, 3)
```

The tensor once again has shape `(batch, rows, columns, d_model)`.

### Row attention

Before applying row attention, we prepend the learnable [CLS] tokens to every row:

```python
x_proj = torch.cat(
    [self.row_cls_tokens.expand(batch_size, num_rows, -1, -1), x_proj],
    dim=2,
)
```

These tokens are inserted along the column dimension and will be used to summarize the feature representations of each row. We then merge the batch and row dimensions so that each row becomes an independent sequence:

```python
num_tokens = self.num_cls_cols + num_cols
x_proj = x_proj.reshape(
    batch_size * num_rows,
    num_tokens,
    -1,
)
```

The intermediate row blocks apply ordinary self-attention to all feature and [CLS] tokens:

```python
for block in self.row_blocks[:-1]:
    x_proj = block(x_proj)
```

In the final row block, only the [CLS] tokens are used as queries:

```python
cls_queries = x_proj[:, : self.num_cls_cols]
x_proj = self.row_blocks[-1](cls_queries, x_proj)
```

Finally, we restore the table dimensions and concatenate the [CLS] representations:

```python
x_proj = x_proj.reshape(
    batch_size,
    num_rows,
    self.num_cls_cols,
    -1,
)

x_proj = self.row_norm(x_proj).flatten(-2, -1)
```

### ICL attention

Before the ICL Transformer, the training labels are embedded a second time, now directly in the ICL embedding space:

```python
x_proj[:, :num_train] += self.proj_y_icl(y[:, :, None])
```

The intermediate ICL blocks update every row, but only the training rows are used as keys and values:

```python
for block in self.icl_blocks[:-1]:
    x_proj = block(
        x_proj,
        x_proj[:, :num_train],
    )
```

The final ICL block is asymmetric. Only the test rows are used as queries, while the training rows provide the context:

```python
x_proj = self.icl_blocks[-1](
    x_proj[:, num_train:],
    x_proj[:, :num_train],
)
```

The final normalization and MLP map these representations to predictions:

```python
return self.out_mlp(self.out_norm(x_proj))
```

## Complete implementation

The complete tiny squeezy TabICL v2 model is shown below. You can find all the steps discussed above in the forward method.

This version omits some implementation details from TabICLv2, such as QASSMax, RoPE and the classification-specific embeddings, in order to focus on the core architecture.

<details>
<summary>TinyTabICL model implementation (click to expand)</summary>

```python
import torch
import torch.nn as nn

class TinyTabICL(nn.Module):
    """Tiny squeezy TabICL model."""

    def __init__(
        self,
        out_dim: int,
        d_model: int = 128,
        num_heads_col: int = 8,
        num_heads_row: int = 8,
        num_heads_icl: int = 8,
        num_col_blocks: int = 3,
        num_row_blocks: int = 3,
        num_icl_blocks: int = 3,
        num_inducing: int = 128,
        num_cls_cols: int = 4,
        feature_group_size: int = 3,
    ):
        super().__init__()
        self.feature_group_size = feature_group_size
        self.num_cls_cols = num_cls_cols

        icl_dim = d_model * num_cls_cols

        self.proj_x = nn.Linear(feature_group_size, d_model)

        # Labels are injected both before the column transformer and before ICL
        self.proj_y = nn.Linear(1, d_model)
        self.proj_y_icl = nn.Linear(1, icl_dim)

        self.col_blocks = nn.ModuleList(
            InducedTransformerBlock(
                d_model=d_model,
                num_heads=num_heads_col,
                num_inducing=num_inducing,
                mlp_ratio=4.0,
                dropout=0.0,
            )
            for _ in range(num_col_blocks)
        )

        self.row_blocks = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads_row,
                mlp_ratio=4.0,
                dropout=0.0,
            )
            for _ in range(num_row_blocks)
        )

        self.icl_blocks = nn.ModuleList(
            TransformerBlock(
                d_model=icl_dim,
                num_heads=num_heads_icl,
                mlp_ratio=4.0,
                dropout=0.0,
            )
            for _ in range(num_icl_blocks)
        )

        # These tokens will summarize the features of each row, the paper uses 4 CLS tokens
        self.row_cls_tokens = nn.Parameter(
            0.02 * torch.randn(1, 1, num_cls_cols, d_model)
        )

        self.row_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(icl_dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(icl_dim, icl_dim * 2),
            nn.GELU(),
            nn.Linear(icl_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size, num_rows, num_cols = x.shape
        y_batch_size, num_train = y.shape

        if y_batch_size != batch_size:
            raise ValueError(
                f"Batch-size mismatch: x has {batch_size}, y has {y_batch_size}."
            )

        # Normalization, use only the training rows for statistics (ofc)
        x = (x - x[:, :num_train].mean(dim=1, keepdim=True)) / (
            x[:, :num_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8
        )

        # Feature grouping
        idxs = torch.arange(num_cols, dtype=torch.long, device=x.device)
        x = torch.stack(
            [
                x[:, :, (idxs + (2**i - 1)) % num_cols]
                for i in range(self.feature_group_size)
            ],
            dim=-1,
        )

        # Embedding and label injection
        x_proj = self.proj_x(x)
        x_proj[:, :num_train] += self.proj_y(y[:, :, None, None])

        # Column attention
        ## Reshape for column attention
        x_proj = x_proj.permute(0, 2, 1, 3).reshape(
            batch_size * num_cols,
            num_rows,
            -1,
        )

        for block in self.col_blocks:
            # All rows are queried, but the inducing tokens summarize training rows only
            train_context = x_proj[:, :num_train]
            x_proj = block(x_proj, train_context)

        ## Reshape back
        x_proj = x_proj.reshape(
            batch_size,
            num_cols,
            num_rows,
            -1,
        ).permute(0, 2, 1, 3)

        # Row attention
        ## Add learnable tokens that will summarize each row
        x_proj = torch.cat(
            [self.row_cls_tokens.expand(batch_size, num_rows, -1, -1), x_proj],
            dim=2,
        )

        ## Reshape for row attention
        num_tokens = self.num_cls_cols + num_cols
        x_proj = x_proj.reshape(
            batch_size * num_rows,
            num_tokens,
            -1,
        )

        for block in self.row_blocks[:-1]:
            x_proj = block(x_proj)

        ## In the last block, only the CLS tokens need updated representations
        cls_queries = x_proj[:, : self.num_cls_cols]
        x_proj = self.row_blocks[-1](cls_queries, x_proj)

        ## Reshape back
        x_proj = x_proj.reshape(
            batch_size,
            num_rows,
            self.num_cls_cols,
            -1,
        )

        ## Concatenate the CLS tokens into one fixed-size row representation
        x_proj = self.row_norm(x_proj).flatten(-2, -1)

        # ICL attention
        ## Inject labels again at the dimension used by the ICL transformer
        x_proj[:, :num_train] += self.proj_y_icl(y[:, :, None])

        for block in self.icl_blocks[:-1]:
            # Every row is queried, but only training rows provide keys and values
            x_proj = block(
                x_proj,
                x_proj[:, :num_train],
            )

        ## The final block only computes representations for the test rows
        x_proj = self.icl_blocks[-1](
            x_proj[:, num_train:],
            x_proj[:, :num_train],
        )

        # Output projection
        return self.out_mlp(self.out_norm(x_proj))
```
</details>

<details>
<summary>Transformer block implementation (click to expand)</summary>

```python
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Classic transformer block with optional cross attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)

        hidden_dim = int(mlp_ratio * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model)
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.norm1(x)

        if context is None:
            kv = q
        else:
            kv = self.norm1(context)

        attn_output, _ = self.attn(
            q,
            kv,
            kv,
            need_weights=False,
        )
        x = x + attn_output

        x = x + self.ffn(self.norm2(x))
        return x
```
</details>

<details>
<summary>Induced Transformer block implementation (click to expand)</summary>

```python
class InducedTransformerBlock(nn.Module):
    """Transformer block with induced tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_inducing: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Learnable inducing tokens / queries that summarize the input sequence
        self.inducing_tokens = nn.Parameter(
            0.02 * torch.randn(1, num_inducing, d_model)
        )

        # Transformer block that compresses the input sequence by attending to the inducing tokens
        self.compress_block = TransformerBlock(
            d_model,
            num_heads,
            mlp_ratio,
            dropout,
        )

        # Transformer block that decompresses to the original sequence length
        self.decompress_block = TransformerBlock(
            d_model,
            num_heads,
            mlp_ratio,
            dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the induced transformer block."""
        batch_size = x.shape[0]

        q = self.inducing_tokens.expand(batch_size, -1, -1)
        source = context if context is not None else x

        # q: (batch_size, num_inducing, d_model)
        # source: (batch_size, seq_len, d_model)
        # z: (batch_size, num_inducing, d_model)
        z = self.compress_block(q, source)

        # z: (batch_size, num_inducing, d_model)
        # x: (batch_size, seq_len, d_model)
        # out: (batch_size, seq_len, d_model)
        out = self.decompress_block(x, z)

        return out
```
</details>

We can now train this model on some prior data.

# Training this tiny TabICL with my own prior data

TBD.