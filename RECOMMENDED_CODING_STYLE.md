# Coding Style Guide

This document outlines the coding style and conventions used in this project, which is primarily written in CUDA C++ with heavy use of CUTLASS and CUTE libraries.

## Table of Contents
1. [General Principles](#general-principles)
2. [File Organization](#file-organization)
3. [Naming Conventions](#naming-conventions)
4. [Formatting](#formatting)
5. [CUDA-Specific Conventions](#cuda-specific-conventions)
6. [Comments](#comments)
7. [Templates and Generics](#templates-and-generics)
8. [Error Handling](#error-handling)

## General Principles

- Follow modern C++ standards (C++14/17) combined with CUDA extensions
- Prioritize performance and readability
- Use CUTLASS and CUTE libraries for tensor operations
- Maintain consistency with NVIDIA's CUTLASS codebase style

## File Organization

- Use hierarchical directory structure: `csrc/sm{xx,90,100}/decode|prefill/dense|sparse/`
- Group related files in subdirectories (e.g., `instantiations/`, `common/`)
- Separate header files (`.h`, `.cuh`) from implementation files (`.cu`, `.cpp`)
- Use `#include` guards or `#pragma once` in headers

## Naming Conventions

### Variables
- Use `snake_case` for local variables and function parameters
- Use descriptive names that indicate purpose
- Examples:
  ```cpp
  int batch_idx = blockIdx.x;
  int warp_idx = threadIdx.x / 32;
  float max_lse = -INFINITY;
  ```

### Tensor Variables (CUTE Library)
- For CUTE `Tensor` objects, use abbreviated mixed case following CUTLASS/CUTE conventions
- Format: lowercase prefix + PascalCase suffix
- Prefix indicates memory space: `g` for global, `s` for shared, `r` for register
- Suffix indicates data type or purpose (single letter or abbreviation)
- Examples:
  ```cpp
  Tensor gA = make_tensor(...);          // Global tensor A
  Tensor gLse = make_tensor(...);        // Global LSE tensor
  Tensor sB = make_tensor(...);          // Shared tensor B
  Tensor rC = make_tensor(...);          // Register tensor C
  ```
- This style prioritizes brevity in mathematical code while maintaining readability

### Functions
- Use `snake_case` for function names
- Kernel functions: `fwd_combine_kernel`
- Host functions: `run_fwd_combine_kernel`
- Template instantiations: `run_fwd_combine_kernel<ElementT>`

### Types and Classes
- Use `PascalCase` for type names
- Template parameters: `ElementT`, `HEAD_DIM_V`
- Structs: `CombineParams`, `DenseAttnDecodeParams`

### Namespaces
- Use nested namespaces: `namespace smxx::decode { ... }`
- Avoid `using namespace` in header files
- Use `using namespace cute;` in implementation files when appropriate

### Constants and Macros
- Use `SCREAMING_SNAKE_CASE` for constants
- CUDA-specific: `FLASH_DEVICE_ASSERT`, `CUTLASS_PRAGMA_UNROLL`
- Mathematical constants: `M_LOG2E`, `CUDART_L2E_F`

## Formatting

### Indentation
- Use 4 spaces for indentation (no tabs)
- Align continuation lines with the opening parenthesis or bracket

### Braces
- Use K&R style (opening brace on same line)
- Always use braces for control structures, even for single statements
- Examples:
  ```cpp
  if (condition) {
      return;
  }

  for (int i = 0; i < n; ++i) {
      // statements
  }
  ```

### Line Length
- Limit lines to 120 characters
- Break long lines at logical points (operators, commas)

### Spacing
- One space around binary operators: `a + b`
- No space after unary operators: `-x`
- Space after keywords: `if (`, `for (`
- No space before semicolons
- No space after C-style casts: `(int)value` not `(int) value`

### Template Declarations
- Always break template declarations onto new lines
- Example:
  ```cpp
  template<typename ElementT, int HEAD_DIM_V, int BLOCK_SIZE_M>
  __global__ void kernel_function(...)
  ```

## CUDA-Specific Conventions

### Kernel Launch
- Use `__launch_bounds__(NUM_THREADS)` for kernel declarations
- Use `__grid_constant__` for constant parameters
- Example:
  ```cpp
  template<typename ElementT, ...>
  __global__ void __launch_bounds__(NUM_THREADS)
  fwd_combine_kernel(__grid_constant__ const CombineParams params)
  ```

### Thread Management
- Use `threadIdx`, `blockIdx`, `blockDim` for indexing
- Calculate `warp_idx = threadIdx.x / 32`, `lane_idx = threadIdx.x % 32`
- Use `__syncwarp()` for warp synchronization

### Memory Access
- Use `__ldg()` for read-only global memory access (when compatible)
- Use `float4` for vectorized loads/stores
- Prefer shared memory for intra-block communication

### Pragmas
- Use `CUTLASS_PRAGMA_UNROLL` for loop unrolling hints
- Use `#pragma unroll` sparingly, prefer compiler hints

## Comments

### Style
- Use `//` for single-line comments
- Use `/* */` for multi-line comments only when necessary
- Place comments above the code they describe

### Content
- Explain complex algorithms and optimizations
- Document kernel grid/block dimensions: `// grid_shape: [batch_size, s_q, h_q/BLOCK_SIZE_M]`
- Note performance-critical sections
- Warn about limitations: `// NOTE: We don't use __ldg here since it is incompatible with PDL`

## Templates and Generics

### Template Parameters
- Use descriptive names: `typename ElementT`, `int HEAD_DIM_V`
- Order: types first, then integers, then booleans
- Example:
  ```cpp
  template<typename ElementT, int HEAD_DIM_V, int BLOCK_SIZE_M, int MAX_SPLITS, int NUM_THREADS>
  ```

### Template Instantiations
- Explicitly instantiate in `.cu` files
- Group instantiations at end of file
- Example:
  ```cpp
  template void run_fwd_combine_kernel<cutlass::bfloat16_t>(CombineParams &params);
  template void run_fwd_combine_kernel<cutlass::half_t>(CombineParams &params);
  ```

## Error Handling

- Use `FLASH_DEVICE_ASSERT` for device-side assertions
- Use `CUTLASS_CHECK` for CUTLASS operations
- Handle edge cases gracefully (e.g., `my_num_splits == 1` early return)
- Validate inputs at kernel launch time

## Tools

### Code Formatting
- Use `clang-format` with the provided `.clang-format` configuration
- Run before committing: `clang-format -i <file>`

### Static Analysis
- Use `clang-tidy` with the provided `.clang-tidy` configuration
- Run on modified files: `clang-tidy <file> -- -I<include_paths>`
- Configure IDE integration for real-time checking
- Checks enforce naming conventions, code quality, and modernization rules
- Some CUDA-specific patterns are intentionally excluded to avoid false positives

### Recommended Workflow
1. Write or modify code
2. Run `clang-format -i` to fix formatting
3. Run `clang-tidy` to check for issues
4. Fix warnings before committing

This style guide should be followed for all new code contributions. For existing code that doesn't conform, consider refactoring during related changes.