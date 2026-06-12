"""
Persistent GPU binary cache for CuTe DSL kernels.

Saves compiled kernel binaries (ELF objects with embedded cubin) to disk,
enabling zero-compilation restarts across Python sessions.

Usage:
    from cula.ops.binary_cache import get_or_compile

    # Automatically compiles on first call, loads from disk on subsequent calls
    compiled = get_or_compile(my_kernel, *example_tensors,
                              name="my_kernel", stream=stream, **kwargs)
    compiled(*actual_tensors, **actual_kwargs)

How it works:
    - Uses `cute.compile()` + `dump_to_object()` to produce a self-contained .o file
    - Saves to `~/.cache/cutlass_binaries/{name}_{hash}.o`
    - On cache hit, uses `cute.runtime.load_module()` for instant (zero-compilation) load
    - Hash is computed from kernel class name, attributes, and argument shapes/dtypes
"""

import os
import hashlib
import json

import cutlass.cute as cute

_BINARY_CACHE_DIR = os.path.expanduser("~/.cache/cutlass_binaries")
os.makedirs(_BINARY_CACHE_DIR, exist_ok=True)


def _compute_kernel_hash(kernel_instance, example_args, example_kwargs):
    """Compute a stable hash for the kernel + its argument shapes/types.
    Excludes runtime objects (stream) that vary between sessions."""
    h = hashlib.sha256()
    # Kernel class and constructor arguments
    h.update(type(kernel_instance).__name__.encode())
    for attr in sorted(vars(kernel_instance).keys()):
        val = str(getattr(kernel_instance, attr)).encode()
        h.update(attr.encode() + b":" + val + b";")
    # Argument shapes and dtypes
    for arg in example_args:
        if hasattr(arg, 'shape'):
            h.update(str(arg.shape).encode())
            h.update(str(arg.dtype).encode())
        elif hasattr(arg, '__cache_key__'):
            h.update(str(arg.__cache_key__).encode())
    # Stable keyword arguments (exclude runtime objects like stream)
    for key in sorted(example_kwargs.keys()):
        if key in ('stream',):
            continue
        h.update(key.encode() + b":" + str(example_kwargs[key]).encode() + b";")
    return h.hexdigest()[:16]


def _get_cache_path(name, kernel_hash):
    """Get the path for a cached kernel binary."""
    safe_name = name.replace("/", "_").replace("\\", "_").replace(".", "_")
    return os.path.join(_BINARY_CACHE_DIR, f"{safe_name}_{kernel_hash}.o")


def _compile_and_save(kernel_instance, example_args, example_kwargs, name, kernel_hash):
    """Compile kernel and save to disk."""
    compiled = cute.compile(kernel_instance, *example_args, **example_kwargs)

    cache_path = _get_cache_path(name, kernel_hash)
    obj_bytes = compiled.dump_to_object(name)
    with open(cache_path, "wb") as f:
        f.write(obj_bytes)

    meta = {
        "name": name,
        "kernel_hash": kernel_hash,
        "kernel_class": type(kernel_instance).__name__,
        "attrs": {k: str(v) for k, v in vars(kernel_instance).items()},
    }
    with open(cache_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[binary_cache] Saved compiled kernel to {cache_path}")
    return compiled


def _load_cached(name, kernel_hash):
    """Load a cached kernel binary from disk."""
    from cutlass.cute.runtime import load_module

    cache_path = _get_cache_path(name, kernel_hash)
    if not os.path.exists(cache_path):
        return None

    try:
        mod = load_module(cache_path)
        compiled = mod[name]
        print(f"[binary_cache] Loaded cached kernel from {cache_path}")
        return compiled
    except Exception as e:
        print(f"[binary_cache] Failed to load cached kernel: {e}, will recompile")
        return None


def get_or_compile(kernel_instance, *example_args, name=None, **example_kwargs):
    """
    Get a compiled kernel, using disk cache if available.

    On first call for a given kernel+arguments combination, compiles the kernel
    and saves the GPU binary to disk. On subsequent calls (even in new Python
    sessions), loads the cached binary instantly with zero compilation.

    Args:
        kernel_instance: A class instance with @cute.jit on __call__.
        *example_args: Example input tensors (CuTe tensors from from_dlpack).
        name: Unique name for this kernel variant (e.g., "kda_qk_sm80").
        **example_kwargs: Additional kwargs passed to the kernel function
            (e.g., stream, problem_size, decay, scale).

    Returns:
        A callable compiled function. Call it with actual tensor arguments
        and keyword arguments (same signature as the original @cute.jit function).
    """
    if name is None:
        name = type(kernel_instance).__name__

    kernel_hash = _compute_kernel_hash(kernel_instance, example_args, example_kwargs)

    cached = _load_cached(name, kernel_hash)
    if cached is not None:
        return cached

    return _compile_and_save(kernel_instance, example_args, example_kwargs, name, kernel_hash)
