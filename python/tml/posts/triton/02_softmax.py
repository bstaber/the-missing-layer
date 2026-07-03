"""Official example: softmax function implemented in Triton."""

import torch
import triton
import triton.language as tl


def softmax_kernel(
    x_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    for row in tl.range(pid, n_rows, row_step):
        print(row)


def softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    y = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    softmax_kernel[grid](x, y, n_rows, BLOCK_SIZE=BLOCK_SIZE)
