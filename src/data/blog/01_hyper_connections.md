---
author: Brian Staber
pubDatetime: 2026-05-20T20:09:15Z
modDatetime: 2026-05-20T20:09:15Z
title: Manifold constrained hyper-connections
slug: manifold-constrained-hyper-connections
featured: false
draft: false
tags:
  - hyper connections
  - transformer
description: Implementing manifold constrained hyper-connections in transformer architectures.
---

# Introduction

In an attention block within a transformer architecture, we have a residual connection that connects the input of the block to its output. This can be represented as:

$$ x*{\ell+1} = x*\ell + F(x\_\ell) $$

where $x_\ell$ is the input to the block, $F$ is the function representing the operations within the block (e.g., multi-head attention, feed-forward network), and $x_{\ell+1}$ is the output of the block.

Hyper-connection can be seen as a generalization of this residual connection, where we have multiple connections from the input to the output, potentially with different weights. Let $n_{h_c}$ be the number of hyper-connections, and let $X_\ell$ be the matrix of inputs at layer $\ell$ with dimensions $n_{h_c} \times d$, where $d$ is the usual dimension of the features. The matrix $X_\ell$ can we be written as:

$$ X*{\ell} = [X*{\ell, 1}, \dots, X*{\ell,n*{h*c}}]^T \in \mathbb{R}^{n*{h_c} \times d} $$

where $X_{\ell, i}$ represents the $i$-th connection. In the hyper-connection framework, the output of the attention block is defined as:

$$ X*{\ell+1} = B*\ell X*\ell + C*\ell F(A*\ell X*\ell) \in \mathbb{R}^{n\_{h_c} \times d} $$

Here, $A_\ell \in \mathbb{R}^{1 \times n_{h_c}}$, $B_\ell \in \mathbb{R}^{n_{h_c} \times n_{h_c}}$, and $C_\ell \in \mathbb{R}^{n_{h_c} \times 1}$ are learnable weight matrices that determine how the inputs are combined and how the function $F$ is applied to the inputs. Given the dimensions of these matrices, we can see that:

- $A_\ell X_\ell$ results in a $1 \times d$ vector, which is the input to the function $F$
- $F(A_\ell X_\ell)$ also results in a $1 \times d$ vector, which is then multiplied by $C_\ell$ to produce a $n_{h_c} \times d$ matrix
- $B_\ell X_\ell$ results in a $n_{h_c} \times d$ matrix, which is added to the output of the second term to produce the final output $X_{\ell+1}$

DeepSeek reported that while hyper-connections can improve model performance, they found that the model could become unstable during training. That's why they introduced a constrained version, called manifold constrained hyper-connections (mHC).

# Manifold constrained hyper-connections

The idea of mHC is to constrain the mapping matrix $B_\ell$ so that its spectral norm is bounded by $1$. It is achieved by constructing $B_\ell$ such that it belongs to the set $\mathcal{M}$ given by:

$$ \mathcal{M} = \{ B \in \mathbb{R}^{n \times n}\,|\, B\mathbf{1}\_n = \mathbf{1}\_n,\, \mathbf{1}\_n^T B = \mathbf{1}\_n^T,\, B \geq 0 \} $$.

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

## Implementation

In coming.
