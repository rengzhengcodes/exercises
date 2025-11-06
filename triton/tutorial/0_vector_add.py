import torch

import triton
import triton.language as tl

# Pick a device with PyTorch
DEVICE = torch.device("cuda", torch.cuda.current_device())

@triton.jit
def add_kernel(
    x_ptr, y_ptr, o_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    """
    Conducts two vector element-wise additions across vectors of the same size
    and writes it to a new object.

    Parameters
    ----------
    x_ptr:
        *Pointer* to first input vector.
    y_ptr:
        *Pointer* to second input vector.
    o_ptr:
        *Pointer* to output vector.
    n_elements:
        Size of the vectors.
    BLOCK_SIZE:
        Number of elements each program should process.
        NOTE: `constexpr` so it can be used as a shape value.
    """
    # Identifies the current program among the multiple concurrent ones.
    pid = tl.program_id(axis=0)    # Launches a 1D grid with axis = 0.
    # The program will process inputs offset from the initial data by breaking
    # the n_elements into chunks of `BLOCK_SIZE`.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load the x,y vectors from DRAM, masking out any extra elements in case the
    # input is not a multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x+y
    # Write x+y back to DRAM.
    tl.store(o_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Helper function to `add_kernel` to allocate the o-tensor and enqueue the
    kernel with the appropriate grid/block sizes.
    """
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements: int = output.numel()

    # The SPMD launch grid denotes the number of kernel instances to run in parallel.
    # Analagous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters)
    grid: Callable = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its
    #   first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a
    #   callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been
    # called, the kernel is still running asynchronously at this point.
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')
