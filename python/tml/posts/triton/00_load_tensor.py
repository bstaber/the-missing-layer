"""Simple example of loading a tensor using Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def load_tensor_kernel(
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


def load_tensor(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    load_tensor_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)

    return output


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)

    size = 98432
    x = torch.rand(size, device=DEVICE)

    output_torch = x.clone()
    output_triton = load_tensor(x)
    print(torch.allclose(output_torch, output_triton))
