# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions for FlashLA debugging and development.
"""

import cutlass
from cutlass import cute


@cute.jit
def print_tensor_2d(tensor: cute.Tensor):
    """
    Print a 2D tensor with 6 decimal places.
    
    Args:
        tensor: A 2D cute.Tensor to print
    """
    rows = cute.size(tensor, mode=[0])
    cols = cute.size(tensor, mode=[1])
    
    cute.printf("---------- tensor [%d x %d] ----------\n", rows, cols)
    
    for i in cutlass.range_constexpr(rows):
        cute.printf("[")
        for j in cutlass.range_constexpr(cols):
            if j > 0:
                cute.printf(", ")
            cute.printf("%10.6f", tensor[i, j].to(cutlass.Float32))
        cute.printf("]\n")
    
    cute.printf("----------------------------------\n")


@cute.jit
def print_tensor(tensor: cute.Tensor):
    """
    Print a 2D tensor with 6 decimal places.
    For higher dimension tensors, flattens and prints as 2D.
    
    Args:
        tensor: A cute.Tensor to print (assumes 2D indexing)
    """
    rows = cute.size(tensor, mode=[0])
    cols = cute.size(tensor, mode=[1])
    
    cute.printf("---------- tensor [%d x %d] ----------\n", rows, cols)
    
    for i in cutlass.range_constexpr(rows):
        cute.printf("[")
        for j in cutlass.range_constexpr(cols):
            if j > 0:
                cute.printf(", ")
            cute.printf("%10.6f", tensor[i, j].to(cutlass.Float32))
        cute.printf("]\n")
    
    cute.printf("----------------------------------\n")


@cute.jit
def print_tensor_flat(tensor: cute.Tensor):
    """
    Print all elements of a tensor in flat order with 6 decimal places.
    
    Args:
        tensor: A cute.Tensor to print
    """
    total_size = cute.size(tensor)
    
    cute.printf("---------- tensor [flat, size=%d] ----------\n", total_size)
    
    for i in cutlass.range_constexpr(total_size):
        cute.printf("[%d] = %10.6f\n", i, tensor.flat_ref(i).to(cutlass.Float32))
    
    cute.printf("----------------------------------\n")


@cute.jit
def print_tensor_partial(tensor: cute.Tensor, max_rows: int, max_cols: int):
    """
    Print a partial view of a 2D tensor (first max_rows x max_cols elements).
    Useful for large tensors where printing everything is impractical.
    
    Args:
        tensor: A 2D cute.Tensor to print
        max_rows: Maximum number of rows to print
        max_cols: Maximum number of columns to print
    """
    rows = cute.size(tensor, mode=[0])
    cols = cute.size(tensor, mode=[1])
    
    print_rows = rows if rows < max_rows else max_rows
    print_cols = cols if cols < max_cols else max_cols
    
    cute.printf("---------- tensor [%d x %d] (showing %d x %d) ----------\n", 
               rows, cols, print_rows, print_cols)
    
    for i in cutlass.range_constexpr(print_rows):
        cute.printf("[")
        for j in cutlass.range_constexpr(print_cols):
            if j > 0:
                cute.printf(", ")
            cute.printf("%10.6f", tensor[i, j].to(cutlass.Float32))
        if print_cols < cols:
            cute.printf(", ...")
        cute.printf("]\n")
    
    if print_rows < rows:
        cute.printf("... (%d more rows)\n", rows - print_rows)
    
    cute.printf("----------------------------------\n")
