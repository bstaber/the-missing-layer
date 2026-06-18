---
author: Brian Staber
pubDatetime: 2026-06-18T21:11:39Z
modDatetime: 2026-06-18T21:11:39Z
title: DeepSeekV4 Manifold constrained hyper-connections
slug: manifold-constrained-hyper-connections
featured: false
draft: false
tags:
  - hyper connections
  - transformer
description: Implementing manifold constrained hyper-connections in transformer architectures.
---

# Introduction

The goal of this note is to give a shot at implementing the manifold constrained hyper-connections introduced in [DeepSeekV4](https://www.alphaxiv.org/abs/deepseek-v4).

In an attention block within a transformer architecture, we have a residual connection that connects the input of the block to its output. This can be represented as:

$$
\begin{aligned}
x_{\ell+1} = x_\ell + F(x_\ell)
\end{aligned}
$$

where $x_\ell$ is the input to the block, $F$ is the function representing the operations within the block (e.g., multi-head attention, feed-forward network), and $x_{\ell+1}$ is the output of the block.

Hyper-connection can be seen as a generalization of this residual connection, where we have multiple connections from the input to the output, potentially with different weights. Let $n_{h_c}$ be the number of hyper-connections, and let $X_\ell$ be the matrix of inputs at layer $\ell$ with dimensions $n_{h_c} \times d$, where $d$ is the usual dimension of the features. The matrix $X_\ell$ can we be written as:

$$
\begin{aligned}
X_{\ell} = [X_{\ell, 1}, \dots, X_{\ell,n_{h_c}}]^T \in \mathbb{R}^{n_{h_c} \times d} 
\end{aligned}
$$

where $X_{\ell, i}$ represents the $i$-th connection. In the hyper-connection framework, the output of the attention block is defined as:

$$
\begin{aligned}
X_{\ell+1} = B_\ell X_\ell + C_\ell F(A_\ell X_\ell) \in \mathbb{R}^{n_{h_c} \times d}
\end{aligned}
$$

Here, $A_\ell \in \mathbb{R}^{1 \times n_{h_c}}$, $B_\ell \in \mathbb{R}^{n_{h_c} \times n_{h_c}}$, and $C_\ell \in \mathbb{R}^{n_{h_c} \times 1}$ are learnable weight matrices that determine how the inputs are combined and how the function $F$ is applied to the inputs. Given the dimensions of these matrices, we can see that:

- $A_\ell X_\ell$ results in a $1 \times d$ vector, which is the input to the function $F$
- $F(A_\ell X_\ell)$ also results in a $1 \times d$ vector, which is then multiplied by $C_\ell$ to produce a $n_{h_c} \times d$ matrix
- $B_\ell X_\ell$ results in a $n_{h_c} \times d$ matrix, which is added to the output of the second term to produce the final output $X_{\ell+1}$

DeepSeek reported that while hyper-connections can improve model performance, they found that the model could become unstable during training. That's why they introduced a constrained version, called manifold constrained hyper-connections (mHC).

# Manifold constrained hyper-connections

The idea of mHC is to constrain the mapping matrix $B_\ell$ so that its spectral norm is bounded by $1$. It is achieved by constructing $B_\ell$ such that it belongs to the set $\mathcal{M}$ given by:

$$ 
\begin{aligned}
& \mathcal{M} = \{ B \in \mathbb{R}^{n \times n}\,|\, B\mathbf{1}_n = \mathbf{1}_n,\, \mathbf{1}_n^T B = \mathbf{1}_n^T,\, B \geq 0 \}\,. 
\end{aligned}
$$

They say that this set is stable under multiplication, which means that we can stack multiple layers of mHC without worrying about the spectral norm growing too large. In addition, the input and output transforms $A_\ell$ and $C_\ell$ are also constrained to be non-negative.

## Parameterization of the matrices

In practice, the matrices $A_\ell$, $B_\ell$, and $C_\ell$ are obtained by transforming unconstrained matrices $\tilde{A}_\ell$, $\tilde{B}_\ell$, and $\tilde{C}_\ell$ using the following transformations:

$$
\begin{aligned}
A_\ell &= \sigma(\tilde{A}_\ell) \\
B_\ell &= \mathrm{SinkhornKnopp}(\tilde{B}_\ell) \\
C_\ell &= 2 \sigma(\tilde{C}_\ell)
\end{aligned}
$$

where $\sigma$ is the sigmoid function, and $\mathrm{SinkhornKnopp}$ denotes the Sinkhorn-Knopp algorithm, which is used to project $\tilde{B}_\ell$ onto the set $\mathcal{M}$. The Sinkhorn-Knopp algorithm simply consists in doing: $B \leftarrow B / (B\mathbf{1}_n)$ and $B \leftarrow B / (\mathbf{1}_n^T B)$ iteratively until convergence (row and column normalizations).

The unconstrained matrices are defined as learnable parameters of the model, decomposed into input-dependent and input-independent components as follows. Let first $\hat{X}_\ell$ be the normalized input defined as

$$
\hat{X}_\ell = \mathrm{RMSNorm(\mathrm{Vec}(X_\ell))} \in \mathbb{R}^{1 \times n_{h_c}d}
$$

where it is recalled that $X_\ell \in \mathbb{R}^{n_{h_c} \times d}$. Then, let $W_{\ell, A} \in \mathbb{R}^{n_{h_c}d \times n_{h_c}}$, $W_{\ell, B} \in \mathbb{R}^{n_{h_c}d \times n_{h_c}^2}$, and $W_{\ell, C} \in \mathbb{R}^{n_{h_c}d \times n_{h_c}}$ be learnable weight matrices. Let also $S_{\ell, A} \in \mathbb{R}^{1 \times n_{h_c}}$, $S_{\ell, B} \in \mathbb{R}^{n_{h_c} \times n_{h_c}}$, and $S_{\ell, C} \in \mathbb{R}^{n_{h_c} \times 1}$ be learnable bias matrices. Finally, let $\alpha_\ell$, $\beta_\ell$, and $\gamma_\ell$ be learnable scalar parameters. The unconstrained matrices are then defined as:

$$
\begin{aligned}
\tilde{A}_\ell &= \alpha_\ell \cdot (\hat{X}_\ell W_{\ell, A}) + S_{\ell, A} \\
\tilde{B}_\ell &= \beta_\ell \cdot \mathrm{Reshape}(\hat{X}_\ell W_{\ell, B}) + S_{\ell, B} \\
\tilde{C}_\ell &= \gamma_\ell \cdot (\hat{X}_\ell W_{\ell, C})^T + S_{\ell, C}
\end{aligned}
$$

where $\mathrm{Reshape}$ is the operation that reshapes matrix product into the appropriate dimensions.

At this stage, we have everything we need to attempt a PyTorch implementation.

# Implementation

In this exercise, I'd like to implement a transformer block that uses manifold constrained hyper-connections. For simplicity, I'm going to use a simple MHA within the block instead of a more complex attention mechanism like the one used in DeepSeek V4. 

My implementation consists of two main elements:

- A `ManifoldHCParams` module that computes the matrices $A_\ell$, $B_\ell$, and $C_\ell$ given the input $X_\ell$.
- A `HyperConnection` module that implements the hyper-connection operation using the matrices computed by `ManifoldHCParams`.  

The final transformer block takes the following form:

```python
class TransformerBlockDeepSeekV4(nn.Module):
    """Transformer block with mHC attention and feed-forward network."""

    def __init__(
        self,
        num_hc: int,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.mha_hc = HyperConnection(
            num_hc,
            d_model,
            SelfAttentionBlock(d_model, num_heads, dropout),
        )
        self.ffn_hc = HyperConnection(
            num_hc,
            d_model,
            FFNBlock(d_model, mlp_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:  Input tensor to the block with shape [B, seq_len, num_hc, d_model]
        """
        x = self.mha_hc(x)
        x = self.ffn_hc(x)
        return x
```

The `HyperConnection` module is initialized with the number of hyper-connections, the model dimension, and a block (either MHA or FFN). The mixing matrices $A_\ell$, $B_\ell$, and $C_\ell$ are computed within the `HyperConnection` module using the `ManifoldHCParams` module.

## The `ManifoldHCParams` module

The goal of this module is to compute the matrices $A_\ell$, $B_\ell$, and $C_\ell$ given the input $X_\ell$. We first compute the unconstrained matrices, and the we apply the appropriate transformations to obtain the constrained matrices. The module is defined as follows:

```python
class ManifoldHCParams(nn.Module):
    """Manifold constrained hyper-connection parameters."""

    def __init__(self, num_hc: int, d_model: int):
        super().__init__()
        self.num_hc = num_hc
        self.rms_norm = nn.RMSNorm(d_model * num_hc)

        self.pre_mix_proj = nn.Linear(num_hc * d_model, num_hc, bias=False)
        self.res_mix_proj = nn.Linear(num_hc * d_model, num_hc * num_hc, bias=False)
        self.post_mix_proj = nn.Linear(num_hc * d_model, num_hc, bias=False)

        self.pre_mix_scale = nn.Parameter(torch.tensor(0.01))
        self.res_mix_scale = nn.Parameter(torch.tensor(0.01))
        self.post_mix_scale = nn.Parameter(torch.tensor(0.01))

        self.pre_mix_shift = nn.Parameter(torch.zeros(num_hc))
        self.res_mix_shift = nn.Parameter(torch.zeros(num_hc, num_hc))
        self.post_mix_shift = nn.Parameter(torch.zeros(num_hc))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Outputs the three components of the mHC block: pre-mix, res-mix, and post-mix.

        Args:
            x: Input tensor to the block with shape [B, seq_len, num_hc, d_model]

        Returns:
            The pre-mix, res-mix, and post-mix tensors with shapes:
            - pre-mix: [B, seq_len, num_hc]
            - res-mix: [B, seq_len, num_hc, num_hc]
            - post-mix: [B, seq_len, num_hc]
        """
        b, seq_len, num_hc, d_model = x.shape
        x_vec = self.rms_norm(x.reshape(b, seq_len, num_hc * d_model))

        pre_mix_uncon = (
            self.pre_mix_scale * self.pre_mix_proj(x_vec) + self.pre_mix_shift
        )
        res_mix_uncon = (
            self.res_mix_scale
            * self.res_mix_proj(x_vec).reshape(b, seq_len, num_hc, num_hc)
            + self.res_mix_shift
        )
        post_mix_uncon = (
            self.post_mix_scale * self.post_mix_proj(x_vec) + self.post_mix_shift
        )

        pre_mix_con = F.sigmoid(pre_mix_uncon)
        res_mix_con = _sinkhorn_knopp(res_mix_uncon)
        post_mix_con = 2.0 * F.sigmoid(post_mix_uncon)

        return pre_mix_con, res_mix_con, post_mix_con
```

Let's break down the components of this module:
- For clarity, the matrices $A_\ell$, $B_\ell$, and $C_\ell$ are referred to as pre-mix, res-mix, and post-mix respectively.
- In the initialization, we define the linear projections ($W_{\ell, A}$, $W_{\ell, B}$, $W_{\ell, C}$) and all the learnable parameters ($\alpha_\ell$, $\beta_\ell$, $\gamma_\ell$, $S_{\ell, A}$, $S_{\ell, B}$, $S_{\ell, C}$), referred to as scales and shifts. The forward method computes the unconstrained matrices and then applies the appropriate transformations to obtain the constrained matrices.
- In the forward method, we first reshape the input $X_\ell$ into a vector and apply RMS normalization.
- We then compute the unconstrained matrices using the linear projections and the learnable parameters. Finally, we apply the appropriate transformations to obtain the constrained matrices.

The `_sinkhorn_knopp` function is a helper function that implements the Sinkhorn-Knopp algorithm to project the res-mix matrix onto the set $\mathcal{M}$:

```python
def _sinkhorn_knopp(h: torch.Tensor, num_iter: int = 20, eps: float = 1e-6):
    """Sinkhorn-Knopp algorithm for matrix normalization."""
    h = torch.exp(h)
    for _ in range(num_iter):
        h = h / (h.sum(dim=-1, keepdim=True) + eps)
        h = h / (h.sum(dim=-2, keepdim=True) + eps)
    return h
```

## The `HyperConnection` module

The goal of this module is simply to implement to update rule:

$$
\begin{aligned}
X_{\ell+1} = B_\ell X_\ell + C_\ell F(A_\ell X_\ell) \in \mathbb{R}^{n_{h_c} \times d}
\end{aligned}
$$

The only tricky part is to make sure that pre-mix and post-mix are applied correctly to the input and output of the block. Here, I rely on `unsqueeze` and `@` to perform the appropriate matrix multiplications. The module is defined as follows:

```python
class HyperConnection(nn.Module):
    """Hyper-connection block."""

    def __init__(self, num_hc: int, d_model: int, block: nn.Module):
        super().__init__()
        self.mhc_params = ManifoldHCParams(num_hc, d_model)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hyper-connection block.

        Args:
            x: Input tensor to the block with shape [B, seq_len, num_hc, d_model]

        Returns:
            The output tensor from the block with shape [B, seq_len, num_hc, d_model]
        """
        pre_mix, res_mix, post_mix = self.mhc_params(x)
        block_in = (pre_mix.unsqueeze(2) @ x).squeeze(2)
        block_out = self.block(block_in)  # [B, seq_len, d_model]

        res_out = res_mix @ x
        post_out = post_mix.unsqueeze(-1) * block_out.unsqueeze(2)
        return res_out + post_out
```

## Putting it all together

That's it, the `TransformerBlockDeepSeekV4` is now complete. For completeness, here are the definitions of the `SelfAttentionBlock` and `FFNBlock` modules used in the transformer block:

```python
class SelfAttentionBlock(nn.Module):
    """Self-attention block with RMS normalization."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking.

        Args:
            x: Input tensor to the block with shape [B, seq_len, d_model]
        """
        x = self.norm(x)

        seq_len = x.shape[1]
        mask = torch.triu(
            torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=x.device,
            ),
            diagonal=1,
        )

        y, _ = self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=mask,
        )
        return y


class FFNBlock(nn.Module):
    """Feed-forward block with RMS normalization."""

    def __init__(self, d_model: int, mlp_ratio: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.norm(x))
```

I ran the following tests to make sure that the implementation is correct in terms of shape and constraints:

```python
if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 5
    num_hc = 3
    d_model = 4
    num_heads = 2
    mlp_ratio = 2

    # Test the ManifoldHCParams module
    mhc_module = ManifoldHCParams(num_hc, d_model)

    x = torch.randn(batch_size, seq_len, num_hc, d_model)
    a, b, c = mhc_module(x)

    assert torch.allclose(
        b.sum(dim=-1), torch.ones(batch_size, seq_len, num_hc), atol=1e-5
    ), "Rows of B do not sum to 1"
    assert torch.allclose(
        b.sum(dim=-2), torch.ones(batch_size, seq_len, num_hc), atol=1e-5
    ), "Columns of B do not sum to 1"

    assert a.shape == (batch_size, seq_len, num_hc), (
        f"Expected shape: [{batch_size}, {seq_len}, {num_hc}], but got: {a.shape}"
    )
    assert b.shape == (batch_size, seq_len, num_hc, num_hc), (
        f"Expected shape: [{batch_size}, {seq_len}, {num_hc}, {num_hc}], but got: {b.shape}"
    )
    assert c.shape == (batch_size, seq_len, num_hc), (
        f"Expected shape: [{batch_size}, {seq_len}, {num_hc}], but got: {c.shape}"
    )

    print("a shape:", a.shape, f"; expected: [{batch_size}, {seq_len}, {num_hc}]")
    print(
        "b shape:",
        b.shape,
        f"; expected: [{batch_size}, {seq_len}, {num_hc}, {num_hc}]",
    )
    print("c shape:", c.shape, f"; expected: [{batch_size}, {seq_len}, {num_hc}]")

    # Test the HyperConnection block
    hyper_connection_block = HyperConnection(
        num_hc, d_model, nn.Linear(d_model, d_model)
    )
    output = hyper_connection_block(x)

    assert output.shape == (batch_size, seq_len, num_hc, d_model), (
        f"Expected shape: [{batch_size}, {seq_len}, {num_hc}, {d_model}], but got: {output.shape}"
    )

    print(
        "output shape:",
        output.shape,
        f"; expected: [{batch_size}, {seq_len}, {num_hc}, {d_model}]",
    )

    # Test the TransformerBlockDeepSeekV4
    transformer_block = TransformerBlockDeepSeekV4(
        num_hc, d_model, num_heads, mlp_ratio, dropout=0.1
    )
    output = transformer_block(x)

    assert output.shape == (batch_size, seq_len, num_hc, d_model), (
        f"Expected shape: [{batch_size}, {seq_len}, {num_hc}, {d_model}], but got: {output.shape}"
    )

    print(
        "transformer output shape:",
        output.shape,
        f"; expected: [{batch_size}, {seq_len}, {num_hc}, {d_model}]",
    )
```